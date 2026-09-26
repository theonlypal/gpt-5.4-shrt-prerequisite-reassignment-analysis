import copy
import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import chi2, fisher_exact

from analysis import (CELLS, CLASSES, FIRTH_SIGNS, INTEGRITY_COUNTERS, MODEL, V0_SIGNS,
                      analyze, decision_for, firth_interaction, holm_adjust,
                      interaction_estimate, pairwise_tests, penalized_loglik,
                      profile_penalized_loglik, wilson_interval)


class StatisticsTests(unittest.TestCase):
    def test_wilson_bounds(self):
        low, high = wilson_interval(0, 200)
        self.assertAlmostEqual(low, 0)
        self.assertAlmostEqual(high, 0.01884532637726658)
        self.assertAlmostEqual(wilson_interval(200, 200)[0], 1 - high)
        self.assertEqual(wilson_interval(0, 0), [None, None])

    def test_crossover_estimand_and_ci_construction(self):
        y, n = [10, 160, 180, 20], [200] * 4
        result = interaction_estimate(y, n)
        self.assertAlmostEqual(result["delta_C"], .75)
        self.assertAlmostEqual(result["delta_S"], .8)
        self.assertAlmostEqual(result["I_REASSIGN"], 1.55)
        bounds = [wilson_interval(k, 200, alpha=.0125) for k in y]
        expected_low = -bounds[0][1] + bounds[1][0] + bounds[2][0] - bounds[3][1]
        expected_high = -bounds[0][0] + bounds[1][1] + bounds[2][1] - bounds[3][0]
        np.testing.assert_allclose(result["ci95"], [expected_low, expected_high])
        c_bounds = [wilson_interval(k, 200, alpha=.025) for k in y[:2]]
        np.testing.assert_allclose(result["contrasts"]["delta_C"]["ci95"],
                                   [c_bounds[1][0] - c_bounds[0][1], c_bounds[1][1] - c_bounds[0][0]])
        maximum = interaction_estimate([0, 200, 200, 0], n)
        self.assertEqual(maximum["interaction"], 2)
        self.assertLess(maximum["ci95"][0], 2)
        self.assertAlmostEqual(maximum["ci95"][1], 2)

    def test_holm_exact_two_test_family_and_original_order(self):
        np.testing.assert_allclose(holm_adjust([.01, .04]), [.02, .04])
        np.testing.assert_allclose(holm_adjust([.04, .01]), [.04, .02])
        np.testing.assert_allclose(holm_adjust([.03, .03]), [.06, .06])
        counts = {cell: Counter(V0=k) for cell, k in zip(CELLS, [10, 50, 170, 90])}
        result = pairwise_tests(counts, dict.fromkeys(CELLS, 200))
        self.assertEqual(result["family_size"], 2)
        self.assertEqual([test["comparison"] for test in result["tests"]], ["E_vs_F", "H_vs_G"])
        expected_tables = [[[10, 190], [50, 150]], [[90, 110], [170, 30]]]
        pvalues = [fisher_exact(table, alternative="two-sided").pvalue for table in expected_tables]
        for test, table, raw, adjusted in zip(result["tests"], expected_tables, pvalues, holm_adjust(pvalues)):
            self.assertEqual(test["table"], table)
            self.assertEqual(test["p_raw"], raw)
            self.assertEqual(test["p_holm_family2"], adjusted)

    def test_firth_coefficient_is_named_prerequisite_by_user_input(self):
        x = np.array([[1., 0., 0., 0.], [1., 0., 1., 0.], [1., 1., 0., 0.], [1., 1., 1., 1.]])
        y, n = np.array([5, 180, 170, 30.]), np.array([200.] * 4)
        eta = np.log((y + .5) / (n - y + .5))
        beta = np.linalg.solve(x, eta)
        fitted = firth_interaction(y, n)
        self.assertAlmostEqual(fitted["coefficient"], beta[3])
        self.assertLess(beta[3], 0)
        np.testing.assert_array_equal(FIRTH_SIGNS, -V0_SIGNS)

    def test_full_matrix_jeffreys_identity(self):
        x = np.array([[1., 0., 0., 0.], [1., 0., 1., 0.], [1., 1., 0., 0.], [1., 1., 1., 1.]])
        n, y = np.array([170, 180, 190, 200.]), np.array([20, 125, 2, 0.])
        differences = []
        for beta in ([.1, .2, -.3, 1.], [-3, 2, 1, -2], [4, -3, 5, -6]):
            eta = x @ np.array(beta)
            p = expit(eta)
            sign, logdet = np.linalg.slogdet(x.T @ np.diag(n * p * (1 - p)) @ x)
            self.assertEqual(sign, 1)
            direct = np.sum(y * eta - n * np.logaddexp(0, eta)) + .5 * logdet
            differences.append(direct - penalized_loglik(eta, y, n))
        np.testing.assert_allclose(differences, differences[0], atol=1e-8)

    def test_profile_endpoints_are_full_model_likelihood_cutoff(self):
        y, n = [2, 150, 170, 10], [200] * 4
        fitted = firth_interaction(y, n)
        eta = np.log((np.array(y) + .5) / (np.array(n) - np.array(y) + .5))
        maximum = penalized_loglik(eta, y, n)
        self.assertAlmostEqual(profile_penalized_loglik(fitted["coefficient"], y, n), maximum, places=7)
        for endpoint in fitted["profile_ci95"]:
            lr = 2 * (maximum - profile_penalized_loglik(endpoint, y, n))
            self.assertAlmostEqual(lr, chi2.ppf(.95, 1), places=5)

    def test_invalid_counts_fail_closed(self):
        for y, n in (([0, 1, 2], [200] * 3), ([0, 1, 2, 201], [200] * 4),
                     ([0, 1, 2, np.nan], [200] * 4), ([0, 1, 2, .5], [200] * 4),
                     ([0, 1, 2, 0], [200, 200, 200, 0])):
            with self.assertRaises(ValueError):
                firth_interaction(y, n)

    def supported_inputs(self):
        totals = dict.fromkeys(CELLS, 200)
        counts = {cell: Counter(V0=200 if cell in ("F", "G") else 0) for cell in CELLS}
        primary = {"interaction": 2., "contrasts": {
            "delta_C": {"estimate": 1., "ci95": [.9, 1.]},
            "delta_S": {"estimate": 1., "ci95": [.9, 1.]}}}
        firth = {"coefficient": -20., "profile_ci95": [-25., -15.], "p_two_sided": .001, "converged": True}
        tests = {"tests": [{"comparison": name, "p_holm_family2": .001} for name in ("E_vs_F", "H_vs_G")]}
        return totals, counts, True, primary, firth, tests

    def test_every_support_gate_is_required(self):
        original = self.supported_inputs()
        self.assertEqual(decision_for(*original), "SUPPORTED")
        mutations = [
            (3, lambda value: value.update(interaction=0.)),
            (3, lambda value: value["contrasts"]["delta_C"].update(estimate=0.)),
            (3, lambda value: value["contrasts"]["delta_S"].update(estimate=-.1)),
            (3, lambda value: value["contrasts"]["delta_C"].update(ci95=[0., 1.])),
            (3, lambda value: value["contrasts"]["delta_S"].update(ci95=[-.1, 1.])),
            (4, lambda value: value.update(coefficient=1.)),
            (4, lambda value: value.update(profile_ci95=[-1., 0.])),
            (4, lambda value: value.update(p_two_sided=.05)),
            (4, lambda value: value.update(converged=False)),
            (5, lambda value: value["tests"][0].update(p_holm_family2=.05)),
            (5, lambda value: value["tests"][1].update(p_holm_family2=.05)),
        ]
        for index, mutate in mutations:
            with self.subTest(index=index, mutate=mutate):
                changed = copy.deepcopy(original)
                mutate(changed[index])
                self.assertEqual(decision_for(*changed), "NOT SUPPORTED")

    def test_integrity_and_completion_failures_are_inconclusive(self):
        inputs = list(self.supported_inputs())
        inputs[2] = False
        self.assertEqual(decision_for(*inputs), "INCONCLUSIVE")
        for n in (199, 201):
            inputs = list(self.supported_inputs())
            inputs[0]["E"] = n
            self.assertEqual(decision_for(*inputs), "INCONCLUSIVE")
        inputs = list(self.supported_inputs())
        inputs[1]["F"]["ERROR"] = 1
        self.assertEqual(decision_for(*inputs), "INCONCLUSIVE")
        self.assertEqual(decision_for(*self.supported_inputs(), statistical_error="synthetic failure"), "INCONCLUSIVE")

    def write_synthetic_inputs(self, root):
        results = root / "results"
        results.mkdir()
        report = {"status": "PASS", "returned_models": [MODEL]}
        report.update({key: [] for key in INTEGRITY_COUNTERS})
        for name in ("verifier_report.json", "protocol_integrity.json"):
            (results / name).write_text(json.dumps(report), encoding="utf-8")
        (root / "PRE_RUN_MANIFEST.json").write_text(json.dumps({"runner_frozen_commit": "SYNTHETIC", "analysis_preregistered_commit": "SYNTHETIC"}), encoding="utf-8")
        with (results / "classification.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["slot_id", "cell", "class", "rule_present", "prerequisite_matched", "attempt_number", "definitive_path"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for cell in CELLS:
                for index in range(200):
                    writer.writerow({"slot_id": "%s-%03d" % (cell, index), "cell": cell,
                                     "class": "V0" if cell in ("F", "G") else "HE",
                                     "rule_present": 1, "prerequisite_matched": int(cell in ("E", "H")),
                                     "attempt_number": 1, "definitive_path": "SYNTHETIC"})
        return results, report

    def test_complete_synthetic_pipeline_and_integrity_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, report = self.write_synthetic_inputs(root)
            result = analyze(root)
            self.assertEqual(result["decision"], "SUPPORTED")
            self.assertEqual(result["classification_counts"]["G"]["V0"], 200)
            self.assertEqual(result["integrity_counts"], dict.fromkeys(INTEGRITY_COUNTERS, 0))
            self.assertEqual(result["pooled_V0"]["match"]["V0_count"], 0)
            self.assertEqual(result["pooled_V0"]["mismatch"]["V0_count"], 400)
            self.assertEqual(result["pooled_V0"]["mismatch"]["recorded_n"], 400)
            with (results / "cell_summary.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 56)
            markdown = (results / "RESULTS.md").read_text(encoding="utf-8")
            for klass in CLASSES:
                self.assertIn("| " + klass + " |", markdown)
            for filename in ("RESULTS.json", "interaction_analysis.json", "statistical_tests.json"):
                json.loads((results / filename).read_text(encoding="utf-8"))
            report["status"], report["hash_failures"] = "FAIL", ["SYNTHETIC deliberate mutation"]
            (results / "protocol_integrity.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(analyze(root)["decision"], "INCONCLUSIVE")

    def test_missing_evidence_and_wrong_model_are_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(analyze(root)["decision"], "INCONCLUSIVE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, report = self.write_synthetic_inputs(root)
            report["returned_models"] = ["gpt-5.4"]
            (results / "verifier_report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(analyze(root)["decision"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
