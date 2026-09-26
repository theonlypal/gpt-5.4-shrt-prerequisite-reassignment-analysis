# Prerequisite Reassignment Crossover in GPT-5.4

SUPPORTED

V0 versus non-V0 continuation tracks prerequisite reassignment between the two named Arabic strings under this frozen GPT-5.4 construction.

Requested model: `gpt-5.4-2026-03-05`. Returned: [gpt-5.4-2026-03-05].

Runner: `https://github.com/theonlypal/gpt-5.4-shrt-prerequisite-reassignment-runner`

Frozen runner commit: `2f08f1ce1f7acdd713d8761e1ff92e041c2aaced`

Frozen runner tag: `v1.0.0-frozen`

Analysis: `https://github.com/theonlypal/gpt-5.4-shrt-prerequisite-reassignment-analysis`

Preregistered analysis commit: `98e57fa53bfcc6879a213b5a44224d0ef47388f2`

Schedule SHA-256: `402cba470369118896cf3b90a4b89f0e3b450bf33b315c52b376d706266a592e`

| Cell | Named prerequisite | Input | Definitive / scheduled | Recorded |
| --- | --- | --- | ---: | ---: |
| E | شَرْط | شَرْط | 200/200 | 200 |
| F | شَرْط | شَمْس | 200/200 | 200 |
| G | شَمْس | شَرْط | 200/200 | 200 |
| H | شَمْس | شَمْس | 200/200 | 200 |

| Class | E | F | G | H |
| --- | ---: | ---: | ---: | ---: |
| T1 | 183 | 0 | 0 | 55 |
| T0 | 0 | 0 | 0 | 0 |
| HE | 16 | 0 | 0 | 144 |
| SHART | 1 | 0 | 1 | 0 |
| SHAMS | 0 | 0 | 0 | 1 |
| MX | 0 | 0 | 0 | 0 |
| OT | 0 | 0 | 1 | 0 |
| V0 | 0 | 200 | 198 | 0 |
| V1 | 0 | 0 | 0 | 0 |
| NV | 0 | 0 | 0 | 0 |
| REFUSAL | 0 | 0 | 0 | 0 |
| SAFETY | 0 | 0 | 0 | 0 |
| TOOL | 0 | 0 | 0 | 0 |
| ERROR | 0 | 0 | 0 | 0 |

All-class proportions and Wilson 95% intervals: `cell_summary.csv`.

P(V0|mismatch): 398/400 recorded (400 scheduled); proportion 0.995.
P(V0|match): 0/400 recorded (400 scheduled); proportion 0.

| Estimate | Value | 95% interval |
| --- | ---: | --- |
| ΔC = pF − pE | 1 | [0.9509921865, 1] |
| ΔS = pG − pH | 0.99 | [0.9338230998, 0.9976591648] |
| I_REASSIGN = pF + pG − pE − pH | 1.99 | [1.861645517, 1.997963014] |
| Firth named prerequisite × user input | -22.35638265 | [-28.95234038, -18.33478235] |

Contrast intervals: signed sum of two Bonferroni-Wilson 97.5% cell intervals; nominal 95% per contrast. Interaction interval: signed sum of Bonferroni-Wilson 98.75% cell intervals; nominal simultaneous 95%.

Firth predicted sign: negative. Penalized likelihood-ratio statistic (df=1): 1066.631495; two-sided p: 5.907999718e-234.

| Two-sided Fisher comparison | Raw p | Holm p (two tests) |
| --- | ---: | ---: |
| E_vs_F | 1.94264345e-119 | 3.885286899e-119 |
| H_vs_G | 3.943760467e-115 | 3.943760467e-115 |

Integrity: PASS.

classification disagreements: 0
hash failures: 0
protocol deviations: 0
model substitutions: 0
schedule deviations: 0

This result concerns the two named strings and the frozen prompts; semantic generalization and an internal gating mechanism are not identified.
