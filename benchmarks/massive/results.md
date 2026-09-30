# MASSIVE benchmark (Zet 0.1)

Model: `multilingual` · error budget 5% · seed 0 · MASSIVE 1.1 `dev`, up to 1000 parallel utterances per language · 2026-09-30

Every number is measured on the held-out 30% of the sample. Labels are MASSIVE's human labels.
The benchmark demonstrates the method; Zet's guarantee for your task comes from your own calibration data.

## scenario18 (18 options; 1000 sampled per language)

| language | held out | coverage | avg set size | automated | error among sure | upper bound (95%) |
|---|---|---|---|---|---|---|
| en-US | 301 | 94.7% | 6.98 | 0.0% | - | - |
| sv-SE | 301 | 96.7% | 9.96 | 0.0% | - | - |
| all | 602 | 95.7% | 8.47 | 0.0% | - | - |

Laya-only means using every top answer from the same checkpoint. Zet keeps that answer when sure and sends unsure answers to review.

| language | Laya-only wrong | Zet wrong automatically | sent to review | Laya wrong in review |
|---|---:|---:|---:|---:|
| en-US | 91/301 | 0 (no automatic answers) | 301 | 91/301 |
| sv-SE | 137/301 | 0 (no automatic answers) | 301 | 137/301 |
| all | 228/602 | 0 (no automatic answers) | 602 | 228/602 |

The model errors in review are mistakes a correct human review could catch; they are not measured human corrections.

## scenario6 (6 options; 582 sampled per language)

| language | held out | coverage | avg set size | automated | error among sure | upper bound (95%) |
|---|---|---|---|---|---|---|
| en-US | 175 | 94.3% | 1.23 | 83.4% | 2.1% | 5.2% |
| sv-SE | 175 | 97.7% | 2.11 | 46.3% | 1.2% | 5.7% |
| all | 350 | 96.0% | 1.67 | 64.9% | 1.8% | 4.0% |

Laya-only means using every top answer from the same checkpoint. Zet keeps that answer when sure and sends unsure answers to review.

| language | Laya-only wrong | Zet wrong automatically | sent to review | Laya wrong in review |
|---|---:|---:|---:|---:|
| en-US | 15/175 | 3/146 | 29 | 12/29 |
| sv-SE | 30/175 | 1/81 | 94 | 29/94 |
| all | 45/350 | 4/227 | 123 | 41/123 |

The model errors in review are mistakes a correct human review could catch; they are not measured human corrections.

Data: MASSIVE (FitzGerald et al., 2022), CC BY 4.0, derived from SLURP (Bastianelli et al., 2020), CC BY 4.0.
