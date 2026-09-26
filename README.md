# Prerequisite Reassignment Crossover

An 800-request confirmatory experiment on `gpt-5.4-2026-03-05`: which of two Arabic strings licenses continuation when the system rule reassigns the prerequisite?

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
