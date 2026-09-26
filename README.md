# Prerequisite Reassignment Crossover

An 800-request confirmatory experiment on `gpt-5.4-2026-03-05`: which of two Arabic strings licenses continuation when the system rule reassigns the prerequisite?

**SUPPORTED.** Reassigning the prerequisite reversed which input normally returned zero-byte, normal-stop content.

| Cell | Named prerequisite | Input | V0 | Visible |
| --- | --- | --- | ---: | ---: |
| E | `شَرْط` | `شَرْط` | 0/200 | 200/200 |
| F | `شَرْط` | `شَمْس` | 200/200 | 0/200 |
| G | `شَمْس` | `شَرْط` | 198/200 | 2/200 |
| H | `شَمْس` | `شَمْس` | 0/200 | 200/200 |

Mismatch: **398/400 V0**. Match: **0/400 V0**, all 400 visible. `I_REASSIGN = 1.99`. Both preregistered contrasts and the negative Firth crossover interaction meet the frozen support criteria.

All 800 attempts returned HTTP 200, `finish_reason=stop`, and the exact requested model ID. Zero retries, transport errors, exclusions, or verifier discrepancies. Both G exceptions remain in the record: slot-0016 returned `شَرْط`; slot-0126 returned `شَمس`.

Full 14-class breakdown and statistics: [results/RESULTS.md](results/RESULTS.md). Raw provider bytes: [raw/attempts](raw/attempts). [Download the complete evidence bundle](https://github.com/theonlypal/gpt-5.4-shrt-prerequisite-reassignment-analysis/releases/download/v1.0.0-results/gpt-5.4-shrt-prerequisite-reassignment-results.zip).

Runner and exact prompts: https://github.com/theonlypal/gpt-5.4-shrt-prerequisite-reassignment-runner

The analysis, 14-class classifier contract, statistical decision, and randomized schedule are frozen before the first primary request. All four cells contain 200 fresh requests. No earlier study is pooled into this dataset.

## Verify the published record without API calls

```sh
git clone https://github.com/theonlypal/gpt-5.4-shrt-prerequisite-reassignment-runner.git
git clone https://github.com/theonlypal/gpt-5.4-shrt-prerequisite-reassignment-analysis.git
python3 -m venv .venv
.venv/bin/pip install -r gpt-5.4-shrt-prerequisite-reassignment-analysis/requirements.txt
.venv/bin/python gpt-5.4-shrt-prerequisite-reassignment-analysis/independent_verifier.py
.venv/bin/python gpt-5.4-shrt-prerequisite-reassignment-analysis/analysis.py
```

The independent verifier reads raw request/response bytes, checks exact model IDs and Unicode, reconstructs every classification, checks the attempt ledger and hashes, and compares files with frozen Git objects. The analysis reports every class and uses the preregistered decision rule in `analysis_plan.json`. `results/RESULTS.json` and `results/RESULTS.md` are generated only after execution.
