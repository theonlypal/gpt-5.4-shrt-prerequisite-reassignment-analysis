#!/usr/bin/env python3
"""Frozen prerequisite-reassignment analysis; consumes independent classifications."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit
from scipy.stats import chi2, fisher_exact, norm

CELLS = ("E", "F", "G", "H")
CLASSES = ("T1", "T0", "HE", "SHART", "SHAMS", "MX", "OT", "V0", "V1", "NV", "REFUSAL", "SAFETY", "TOOL", "ERROR")
N_PER_CELL = 200
MODEL = "gpt-5.4-2026-03-05"
ALPHA = .05
INTEGRITY_COUNTERS = ("classification_disagreements", "hash_failures", "protocol_deviations", "model_substitutions", "schedule_deviations")
V0_SIGNS = np.array([-1., 1., 1., -1.])
FIRTH_SIGNS = -V0_SIGNS
CI_METHOD = "signed sum of Bonferroni-Wilson 98.75% cell intervals; nominal simultaneous 95%"
CONTRAST_CI_METHOD = "signed sum of two Bonferroni-Wilson 97.5% cell intervals; nominal 95% per contrast"
INTERPRETATIONS = {
    "SUPPORTED": "V0 versus non-V0 continuation tracks prerequisite reassignment between the two named Arabic strings under this frozen GPT-5.4 construction.",
    "NOT SUPPORTED": "The complete, protocol-valid experiment did not meet every preregistered crossover support criterion.",
    "INCONCLUSIVE": "Incomplete evaluation, missing evidence, or protocol/integrity or numerical failure prevents the preregistered decision."
}


def validate_counts(successes, ns, size=None):
    y, n = np.asarray(successes, dtype=float), np.asarray(ns, dtype=float)
    if y.ndim != 1 or y.shape != n.shape or (size is not None and y.shape != (size,)):
        raise ValueError("aligned cell count vectors required")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(n)):
        raise ValueError("finite counts required")
    if np.any(y != np.floor(y)) or np.any(n != np.floor(n)) or np.any(n <= 0) or np.any(y < 0) or np.any(y > n):
        raise ValueError("nonempty cells and integer binomial counts required")
    return y, n


def wilson_interval(successes, n, alpha=ALPHA):
    if not (0 <= successes <= n) or n < 0 or not 0 < alpha < 1:
        raise ValueError("invalid binomial counts or alpha")
    if n == 0:
        return [None, None]
    z = float(norm.ppf(1 - alpha / 2))
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0., center - radius), min(1., center + radius)]


def signed_interval(successes, ns, signs):
    y, n = validate_counts(successes, ns)
    signs = np.asarray(signs, dtype=float)
    if signs.shape != y.shape or not np.all(np.isin(signs, [-1., 1.])):
        raise ValueError("one positive or negative sign required per cell")
    intervals = [wilson_interval(int(a), int(b), ALPHA / len(y)) for a, b in zip(y, n)]
    low = sum(s * bound[0 if s > 0 else 1] for s, bound in zip(signs, intervals))
    high = sum(s * bound[1 if s > 0 else 0] for s, bound in zip(signs, intervals))
    return float(np.dot(signs, y / n)), [float(low), float(high)], intervals


def interaction_estimate(successes, ns, signs=V0_SIGNS):
    y, n = validate_counts(successes, ns, 4)
    estimate, interval, cell_intervals = signed_interval(y, n, signs)
    contrasts = {}
    for label, indices, contrast_signs in (("delta_C", [0, 1], [-1, 1]), ("delta_S", [2, 3], [1, -1])):
        value, bounds, intervals = signed_interval(y[indices], n[indices], contrast_signs)
        contrasts[label] = {"estimate": value, "ci95": bounds, "ci_method": CONTRAST_CI_METHOD,
                            "cell_intervals_used": {CELLS[i]: bound for i, bound in zip(indices, intervals)}}
    return {
        "cell_counts": dict(zip(CELLS, map(int, y))), "cell_n": dict(zip(CELLS, map(int, n))),
        "cell_proportions": dict(zip(CELLS, map(float, y / n))),
        "delta_C": contrasts["delta_C"]["estimate"], "delta_S": contrasts["delta_S"]["estimate"],
        "contrasts": contrasts, "interaction": estimate, "I_REASSIGN": estimate,
        "ci95": interval, "ci_method": CI_METHOD,
        "cell_intervals_used": dict(zip(CELLS, cell_intervals)),
    }


def penalized_loglik(eta, successes, ns):
    """Full-model Jeffreys-penalized saturated likelihood, constants omitted.

    det(X'WX)=det(X)^2*product(w_cell) for these four full-rank design rows.
    The same four-parameter penalty remains in every constrained profile fit.
    """
    eta, y, n = map(lambda value: np.asarray(value, dtype=float), (eta, successes, ns))
    return float(np.sum((y + .5) * eta - (n + 1) * np.logaddexp(0, eta)))


def profile_penalized_loglik(coefficient, successes, ns):
    """Maximize l* at eta_E-eta_F-eta_G+eta_H = coefficient."""
    y, n = validate_counts(successes, ns, 4)
    full_eta = np.log((y + .5) / (n - y + .5))
    # Free eta_E, eta_F, eta_G; eta_H = coefficient-eta_E+eta_F+eta_G.
    transform = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 1., 1.]])
    offset = np.array([0., 0., 0., float(coefficient)])

    def objective(free):
        return -penalized_loglik(transform @ free + offset, y, n)

    def gradient(free):
        return transform.T @ ((n + 1) * expit(transform @ free + offset) - (y + .5))

    projected = full_eta + FIRTH_SIGNS * (coefficient - np.dot(FIRTH_SIGNS, full_eta)) / 4
    fit = minimize(objective, projected[:3], jac=gradient, method="BFGS", options={"gtol": 1e-10, "maxiter": 500})
    free = fit.x.copy()
    for _ in range(12):
        score = gradient(free)
        if np.max(np.abs(score)) <= 1e-9:
            break
        probability = expit(transform @ free + offset)
        weights = (n + 1) * probability * (1 - probability)
        hessian = transform.T @ (weights[:, None] * transform)
        step = np.linalg.solve(hessian, score)
        scale, current = 1., objective(free)
        for _ in range(30):
            candidate = free - scale * step
            if objective(candidate) <= current + 1e-10:
                free = candidate
                break
            scale *= .5
        else:
            break
    score_norm = float(np.max(np.abs(gradient(free))))
    if not math.isfinite(float(objective(free))) or score_norm > 1e-6:
        raise ArithmeticError("Firth profile did not converge: score_inf=%r; %s" % (score_norm, fit.message))
    if abs(float(np.dot(FIRTH_SIGNS, transform @ free + offset)) - coefficient) > 1e-8:
        raise ArithmeticError("Firth profile constraint mismatch")
    return -float(objective(free))


def firth_interaction(successes, ns, alpha=ALPHA):
    y, n = validate_counts(successes, ns, 4)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    p = (y + .5) / (n + 1)
    eta = np.log((y + .5) / (n - y + .5))
    estimate = float(np.dot(FIRTH_SIGNS, eta))
    maximum = penalized_loglik(eta, y, n)
    threshold = float(chi2.ppf(1 - alpha, 1))

    def lr_at(value):
        return max(0., 2 * (maximum - profile_penalized_loglik(value, y, n)))

    def endpoint(direction):
        distance = 1.
        for _ in range(12):
            edge = estimate + direction * distance
            if lr_at(edge) >= threshold:
                return float(brentq(lambda value: lr_at(value) - threshold, min(edge, estimate), max(edge, estimate), xtol=1e-8))
            distance *= 2
        raise ArithmeticError("Firth profile confidence limit was not bracketed")

    interval = [endpoint(-1), endpoint(1)]
    lr = lr_at(0.)
    return {
        "model": "V0 ~ named_prerequisite + user_input + named_prerequisite:user_input",
        "coding": {"named_prerequisite": {"SHART": 0, "SHAMS": 1}, "user_input": {"SHART": 0, "SHAMS": 1}},
        "coefficient": estimate, "predicted_direction": "negative",
        "coefficient_definition": "eta_E-eta_F-eta_G+eta_H",
        "se": float(np.sqrt(np.sum(1 / (n * p * (1 - p))))),
        "se_method": "ordinary binomial Fisher information evaluated at Firth estimate; profile CI is authoritative",
        "profile_ci95": interval, "penalized_lr": lr, "df": 1,
        "p_two_sided": float(chi2.sf(lr, 1)),
        "fitted_cell_probabilities": dict(zip(CELLS, map(float, p))), "converged": True,
        "method": "Firth saturated 2x2 Jeffreys penalty; profile retains the full-model penalty; chi-square(1) calibration",
    }


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("p values must be finite probabilities")
    order, result, running = np.argsort(p, kind="stable"), np.zeros(len(p)), 0.
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        result[index] = min(1., running)
    return list(map(float, result))


def pairwise_tests(counts, totals):
    records = []
    for first, second in (("E", "F"), ("H", "G")):
        a, b = counts[first]["V0"], counts[second]["V0"]
        table = [[a, totals[first] - a], [b, totals[second] - b]]
        statistic, pvalue = fisher_exact(table, alternative="two-sided")
        records.append({"outcome": "V0", "comparison": first + "_vs_" + second, "table": table,
                        "odds_ratio": float(statistic) if math.isfinite(statistic) else None,
                        "odds_ratio_nonfinite": None if math.isfinite(statistic) else str(statistic),
                        "p_raw": float(pvalue), "alternative": "two-sided"})
    for record, adjusted in zip(records, holm_adjust([record["p_raw"] for record in records])):
        record["p_holm_family2"] = adjusted
    return {"family": "two rule-stratified V0 comparisons E_vs_F and H_vs_G", "family_size": 2, "adjustment": "Holm", "tests": records}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def counters_from_report(report):
    counters, result = report.get("counts", {}), {}
    for key in INTEGRITY_COUNTERS:
        value = report.get(key, counters.get(key))
        result[key] = len(value) if isinstance(value, list) else value
    return result


def decision_for(totals, counts, integrity_pass, primary, firth, tests, statistical_error=None):
    if not integrity_pass or any(totals[c] != N_PER_CELL for c in CELLS) or any(counts[c]["ERROR"] for c in CELLS) or statistical_error:
        return "INCONCLUSIVE"
    if primary is None or firth is None or tests is None:
        return "INCONCLUSIVE"
    contrasts = [primary["contrasts"][name] for name in ("delta_C", "delta_S")]
    tests_by_name = {test["comparison"]: test for test in tests["tests"]}
    if set(tests_by_name) != {"E_vs_F", "H_vs_G"} or len(tests["tests"]) != 2:
        return "INCONCLUSIVE"
    supported = (primary["interaction"] > 0
                 and all(contrast["estimate"] > 0 and contrast["ci95"][0] > 0 for contrast in contrasts)
                 and all(test["p_holm_family2"] < ALPHA for test in tests_by_name.values())
                 and firth["coefficient"] < 0 and firth["profile_ci95"][1] < 0
                 and firth["p_two_sided"] < ALPHA and firth["converged"] is True)
    return "SUPPORTED" if supported else "NOT SUPPORTED"


def analyze(root):
    root = Path(root).resolve()
    output = root / "results"
    output.mkdir(exist_ok=True)
    issues = []

    def read_required(path):
        try:
            return load_json(path)
        except (OSError, ValueError) as exc:
            issues.append("unreadable required evidence %s: %s" % (path.name, exc))
            return {}

    verifier = read_required(output / "verifier_report.json")
    integrity = read_required(output / "protocol_integrity.json")
    manifest = read_required(root / "PRE_RUN_MANIFEST.json")
    try:
        with (output / "classification.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        issues.append("unreadable classification.csv: " + str(exc))
        rows = []
    counts, slots = {cell: Counter() for cell in CELLS}, []
    for row in rows:
        cell, klass, slot = row.get("cell"), row.get("class"), row.get("slot_id")
        if cell not in CELLS or klass not in CLASSES or not slot:
            issues.append("invalid classification row: " + repr((slot, cell, klass)))
            continue
        slots.append(slot)
        counts[cell][klass] += 1
    if len(set(slots)) != len(slots):
        issues.append("duplicate classification slot_id")
    totals = {cell: sum(counts[cell].values()) for cell in CELLS}
    counters, integrity_counters = counters_from_report(verifier), counters_from_report(integrity)
    for key in INTEGRITY_COUNTERS:
        available = [value for value in (counters[key], integrity_counters[key]) if value is not None]
        counters[key] = max(available) if available else None
    returned = verifier.get("returned_models", integrity.get("returned_models", []))
    if returned != [MODEL]:
        issues.append("returned model evidence does not contain exclusively the required snapshot")
    integrity_pass = (verifier.get("status") == "PASS" and integrity.get("status") == "PASS"
                      and all(counters[key] == 0 for key in INTEGRITY_COUNTERS) and not issues)
    summaries = []
    for cell in CELLS:
        for klass in CLASSES:
            n, k = totals[cell], counts[cell][klass]
            low, high = wilson_interval(k, n)
            summaries.append({"cell": cell, "class": klass, "count": k, "n": n, "expected_n": N_PER_CELL,
                              "proportion": k / n if n else None, "wilson95_low": low, "wilson95_high": high})
    with (output / "cell_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    interactions = {"primary_V0": None, "firth_primary": None, "statistical_error": None}
    tests = {"family": "two rule-stratified V0 comparisons E_vs_F and H_vs_G", "family_size": 2, "adjustment": "Holm", "tests": []}
    if all(totals[cell] > 0 for cell in CELLS):
        ns, ys = [totals[cell] for cell in CELLS], [counts[cell]["V0"] for cell in CELLS]
        try:
            interactions["primary_V0"] = interaction_estimate(ys, ns)
            tests = pairwise_tests(counts, totals)
            interactions["firth_primary"] = firth_interaction(ys, ns)
        except (ArithmeticError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            interactions["statistical_error"] = str(exc)
    else:
        interactions["statistical_error"] = "one or more cells have no recorded outcomes"
    decision = decision_for(totals, counts, integrity_pass, interactions["primary_V0"], interactions["firth_primary"], tests, interactions["statistical_error"])
    pooled = {}
    for name, cells in (("match", ("E", "H")), ("mismatch", ("F", "G"))):
        k, n = sum(counts[cell]["V0"] for cell in cells), sum(totals[cell] for cell in cells)
        pooled[name] = {"cells": list(cells), "V0_count": k, "recorded_n": n, "scheduled_n": 400,
                        "proportion": k / n if n else None, "wilson_ci95": wilson_interval(k, n)}
    result = {
        "title": "Prerequisite Reassignment Crossover in GPT-5.4",
        "requested_model": MODEL, "returned_models": returned,
        "decision": decision, "interpretation": INTERPRETATIONS[decision],
        "metadata": manifest, "expected_per_cell": N_PER_CELL, "recorded_per_cell": totals,
        "definitive_model_outcomes_per_cell": {cell: totals[cell] - counts[cell]["ERROR"] for cell in CELLS},
        "classification_counts": {cell: {klass: counts[cell][klass] for klass in CLASSES} for cell in CELLS},
        "integrity_pass": integrity_pass, "integrity_counts": counters,
        "analysis_input_issues": issues, "interactions": interactions, "statistical_tests": tests, "pooled_V0": pooled,
        "denominator_rule": "Retain all recorded terminal slots, including ERROR. Expected denominator is 200 per cell; missing slots are not imputed and incomplete data cannot support the confirmatory claim."
    }
    write_json(output / "interaction_analysis.json", interactions)
    write_json(output / "statistical_tests.json", tests)
    write_json(output / "RESULTS.json", result)
    (output / "RESULTS.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def display(value):
    if value is None:
        return "not estimable"
    if isinstance(value, float):
        return "%.10g" % value
    if isinstance(value, list):
        return "[" + ", ".join(display(item) for item in value) + "]"
    return str(value)


def render_markdown(result):
    meta = result["metadata"]
    lines = ["# " + result["title"], "", result["decision"], "", result["interpretation"], "",
             "Requested model: `" + result["requested_model"] + "`. Returned: " + display(result["returned_models"]) + ".", ""]
    for label, key in (("Runner", "runner_repository_url"), ("Frozen runner commit", "runner_frozen_commit"),
                       ("Frozen runner tag", "runner_frozen_tag"), ("Analysis", "analysis_repository_url"),
                       ("Preregistered analysis commit", "analysis_preregistered_commit"), ("Schedule SHA-256", "schedule_sha256")):
        lines.extend([label + ": `" + str(meta.get(key, "MISSING")) + "`", ""])
    lines.extend(["| Cell | Named prerequisite | Input | Definitive / scheduled | Recorded |", "| --- | --- | --- | ---: | ---: |"])
    labels = {"E": ("شَرْط", "شَرْط"), "F": ("شَرْط", "شَمْس"), "G": ("شَمْس", "شَرْط"), "H": ("شَمْس", "شَمْس")}
    for cell in CELLS:
        lines.append("| %s | %s | %s | %d/200 | %d |" % (cell, *labels[cell], result["definitive_model_outcomes_per_cell"][cell], result["recorded_per_cell"][cell]))
    lines.extend(["", "| Class | E | F | G | H |", "| --- | ---: | ---: | ---: | ---: |"])
    for klass in CLASSES:
        lines.append("| " + klass + " | " + " | ".join(str(result["classification_counts"][cell][klass]) for cell in CELLS) + " |")
    lines.extend(["", "All-class proportions and Wilson 95% intervals: `cell_summary.csv`.", ""])
    for name in ("mismatch", "match"):
        pool = result["pooled_V0"][name]
        lines.append("P(V0|%s): %d/%d recorded (%d scheduled); proportion %s." % (name, pool["V0_count"], pool["recorded_n"], pool["scheduled_n"], display(pool["proportion"])))
    primary = result["interactions"]["primary_V0"] or {}
    firth = result["interactions"]["firth_primary"] or {}
    lines.extend(["", "| Estimate | Value | 95% interval |", "| --- | ---: | --- |"])
    for name, label in (("delta_C", "ΔC = pF − pE"), ("delta_S", "ΔS = pG − pH")):
        contrast = primary.get("contrasts", {}).get(name, {})
        lines.append("| %s | %s | %s |" % (label, display(contrast.get("estimate")), display(contrast.get("ci95"))))
    lines.append("| I_REASSIGN = pF + pG − pE − pH | %s | %s |" % (display(primary.get("interaction")), display(primary.get("ci95"))))
    lines.append("| Firth named prerequisite × user input | %s | %s |" % (display(firth.get("coefficient")), display(firth.get("profile_ci95"))))
    lines.extend(["", "Contrast intervals: " + CONTRAST_CI_METHOD + ". Interaction interval: " + CI_METHOD + ".", "",
                  "Firth predicted sign: negative. Penalized likelihood-ratio statistic (df=1): %s; two-sided p: %s." % (display(firth.get("penalized_lr")), display(firth.get("p_two_sided"))), "",
                  "| Two-sided Fisher comparison | Raw p | Holm p (two tests) |", "| --- | ---: | ---: |"])
    for test in result["statistical_tests"]["tests"]:
        lines.append("| %s | %s | %s |" % (test["comparison"], display(test["p_raw"]), display(test["p_holm_family2"])))
    lines.extend(["", "Integrity: " + ("PASS" if result["integrity_pass"] else "FAIL") + ".", ""])
    for key in INTEGRITY_COUNTERS:
        lines.append(key.replace("_", " ") + ": " + display(result["integrity_counts"][key]))
    if result["analysis_input_issues"]:
        lines.extend(["", "Input issues: " + "; ".join(result["analysis_input_issues"])])
    if result["interactions"]["statistical_error"]:
        lines.extend(["", "Statistical status: " + result["interactions"]["statistical_error"]])
    lines.extend(["", "This result concerns the two named strings and the frozen prompts; semantic generalization and an internal gating mechanism are not identified.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    print(analyze(parser.parse_args().root)["decision"])
