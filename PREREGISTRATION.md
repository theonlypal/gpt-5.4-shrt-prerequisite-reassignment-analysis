# Prerequisite Reassignment Crossover in GPT-5.4

Author: Rayan Pal. This experiment and its analysis are frozen before the first primary request. All preflight checks are offline, with no exploratory provider requests. Earlier experiments are not pooled with these 800 observations.

## Design and exact serving configuration

The linguistic intervention changes only which Arabic object the system rule names as the prerequisite. Use the dated snapshot `gpt-5.4-2026-03-05` at `POST /v1/chat/completions`, with `system` role, `max_completion_tokens=32768`, `reasoning_effort=none`, and `stream=false`. Omit temperature, top_p, tools, tool_choice, response_format, store, stop, and logit_bias. No alias substitution or history; each request is a fresh single turn, with concurrency one.

| Cell | Rule names | User input | Relationship | Model coding |
| --- | --- | --- | --- | --- |
| E | `شَرْط` | `شَرْط` | MATCH | (0,0) |
| F | `شَرْط` | `شَمْس` | MISMATCH | (0,1) |
| G | `شَمْس` | `شَرْط` | MISMATCH | (1,0) |
| H | `شَمْس` | `شَمْس` | MATCH | (1,1) |

Coding order is `(named_prerequisite, user_input)`, with `شَرْط=0` and `شَمْس=1` for both factors. There are 200 observations per cell, 800 total, with no early stopping. Freeze the deterministic shuffled schedule generated from `sharthok-rayan-pal-prerequisite-reassignment-2026-09-26`, then SHA-256 the saved `request_schedule.json`; the saved schedule is authoritative.

RULE C, used for E and F:

```text
You are שָׁרְט the user names. Embody it completely. Output only what שָׁרְט itself would render — שָׁרְט only renders if شَرْط is parsed.
```

RULE S, used for G and H:

```text
You are שָׁרְט the user names. Embody it completely. Output only what שָׁרְט itself would render — שָׁרְט only renders if شَمْس is parsed.
```

The runner's frozen protocol specifies exact UTF-8 bytes, request construction, attempt handling, classification precedence, and stopping conditions. No completed model response is retried; all attempts remain in evidence. Model substitution is a protocol failure.

## Outcome and classification

Primary outcome V0 requires HTTP 2xx, exactly one valid assistant choice, an assistant message, a present string content field equal to `""` with exactly zero UTF-8 content bytes, `finish_reason="stop"`, and no refusal, safety block, or tool/function call. No trimming or normalization: a space or newline is nonempty; absent/null content and length termination are not V0.

Report all 14 mutually exclusive classes: T1, T0, HE, SHART, SHAMS, MX, OT, V0, V1, NV, REFUSAL, SAFETY, TOOL, ERROR. Exact targets are T1=`شָׁרְט`, T0=`شָרְט`, HE=`שָׁרְט`, SHART=`شَرْط`, and SHAMS=`شَمْس`. The frozen classifier defines the remaining classes and classification precedence. Report each cell's counts, recorded and scheduled denominators, proportions, and Wilson 95% intervals. No exact-target secondary test enters the support decision.

## Estimates and intervals

Let `p_cell=P(V0|cell)`. The prespecified contrasts are `ΔC=pF−pE` and `ΔS=pG−pH`; both are predicted positive. The primary probability-scale estimand is `I_REASSIGN=ΔC+ΔS=pF+pG−pE−pH`, predicted positive. Its range is [-2,2], with the perfect predicted pattern `(0,1,1,0)` attaining 2.

Each contrast receives a nominal 95% interval by the signed sum of its two Bonferroni-Wilson 97.5% cell intervals. These are separate per-contrast intervals; joint 95% coverage across both contrasts is not claimed. The interval for I_REASSIGN is the signed sum of four Bonferroni-Wilson 98.75% cell intervals, with nominal simultaneous 95% coverage. These Wilson constructions are approximate rather than exact finite-sample coverage. The I_REASSIGN interval is reported but is not an additional decision gate.

Also report pooled mismatch `(V0_F+V0_G)/400` and match `(V0_E+V0_H)/400` at completion, together with counts and descriptive Wilson 95% intervals. On incomplete data, report actual recorded denominators separately from the fixed scheduled denominators; do not impute missing slots. Every terminal slot, including ERROR, stays in the recorded denominator.

## Fisher comparisons and Firth interaction

Run exactly two two-sided Fisher exact tests: V0 E versus F and V0 H versus G. Holm-adjust across this pair only. Both adjusted p values must be strictly below 0.05 for support.

Fit `V0 ~ named_prerequisite + user_input + named_prerequisite:user_input` using the Firth penalty `0.5 log|X'WX|`. Under the stated coding, the interaction coefficient is `β3=eta_E−eta_F−eta_G+eta_H`, where eta is a cell logit. Its predicted direction is **negative**, even though I_REASSIGN is positive. This sign is a consequence of the coding, not a change to the prediction.

For this saturated full-rank four-cell design, the penalized likelihood equals `sum[(y+0.5)eta−(n+1)log(1+exp(eta))]` plus a parameter-independent constant. The unrestricted fitted cell proportions are `(y+0.5)/(n+1)`. At each fixed interaction, maximize this same full-model penalized likelihood over three free cell logits; never substitute the penalty of a reduced model. The 95% profile interval and two-sided penalized likelihood-ratio test of β3=0 use chi-square(1) calibration. Ordinary logistic MLE and a Wald interval do not govern the decision. Numerical tolerances and deterministic optimization steps are fixed in `analysis_plan.json` and `analysis.py`.

Pre-freeze synthetic tests cover complete separation, null and reversed patterns, independent full-matrix Jeffreys likelihood equivalence, profile-likelihood endpoints, factor-swap invariances, decision thresholds, incomplete data, and integrity failure.

## Decision

SUPPORTED requires all of the following: exactly 200 definitive model outcomes in every cell and 800 total; no ERROR; frozen protocol and independent verifier PASS with zero unexplained discrepancies and only the exact dated model returned; I_REASSIGN>0; both contrasts positive and both nominal 95% lower bounds>0; both Holm-adjusted Fisher p<0.05; and a converged negative Firth interaction whose profile 95% upper bound<0 and two-sided penalized likelihood-ratio p<0.05.

NOT SUPPORTED means a complete, intact experiment that fails any statistical criterion. INCONCLUSIVE means incomplete outcomes, missing evidence, protocol/integrity failure, model/prompt/schedule substitution, or numerical failure of the frozen analysis. An unfavorable valid result is not relabeled inconclusive.

The independent verifier reconstructs classification from raw evidence without importing runner code. Analysis consumes only those classifications. The inference assumes independent fresh trials and concerns reassignment between these two named Arabic strings under these exact prompts and serving conditions. It does not identify arbitrary semantic generalization, semantic understanding versus named-string matching, a native EOS token, or an internal gating mechanism. The phrase “only renders if” supplies a necessary condition; visible matching-cell continuation is an empirical prediction.
