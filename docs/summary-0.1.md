# Zet 0.1: summary

What was built, what deviates from the brief and why, and the measured numbers. Details and
reasoning for every decision: [decisions.md](decisions.md). Laya facts: [laya-notes.md](laya-notes.md).

## What was built

All seven steps of the brief:

1. **Laya study** (`docs/laya-notes.md`): input and output formats, both ONNX layouts and routing, with file and line references.
2. **Backends.** `LayaOnnxBackend` runs without PyTorch, on single-graph or split weights. `LayaHttpBackend` talks to `laya-serve`. `FakeBackend` is for tests. All three pass one shared contract test.
3. **Tasks** saved on disk: examples, predictions and corrections as append-only logs, calibration versions, and a question hash that refuses stale calibration.
4. **Calibration.**
   - Split-conformal prediction sets.
   - A Learn-then-Test threshold for `sure`, with exact Clopper–Pearson bounds.
   - Fitted per language group, with pooled fallback and label-conditional sets for choice questions.
   - Everything measured on a 30% held-out split.
5. **Audits and drift.** Sure answers are sampled for human spot checks, and drift warnings come from audit verdicts.
6. **Report and CLI.** `task.report()` and `zet init / add-examples / calibrate / predict / correct / confirm / report`.
7. **Fixtures, tests, docs.** 125 English and Swedish support emails, 123 tests, the MASSIVE benchmark script, README and CI.

## Deviations from the brief, and why

- **Laya's package can't be a dependency.** It requires PyTorch, so the torch-free parts are copied with license headers (Q1).
- **Laya's language detector didn't know Swedish**, and sent some Swedish, and all short Danish, to the English checkpoint. Zet's copy adds sv/no/da, plus a Norwegian/Danish family rule.
- **The threshold scan needed a rule the brief didn't state.** Without skipping thresholds too thin to ever pass (59 sure answers at a 5% budget), nothing was ever automated. The skip depends on counts only, so the guarantee stands.
- **English weights:** no full-precision English ONNX export is hosted anymore, so `model="auto"` answers English with the multilingual checkpoint (F1, open).
- **Recalibration** writes a new version instead of overwriting version 1.
- **An uncalibrated task still predicts**, with every answer `unsure`.

## Measured numbers

**Platform** (Windows 11, Smart App Control on, Python 3.14, CPU):
- torch-free install, and PyTorch never imported;
- 28 ms per (email, question).

**Equivalence with Laya's PyTorch path** (multilingual checkpoint, 12 emails × 4 questions):
- identical token rows and identical top answers in 48 of 48 cases;
- max probability difference 9.0e-4.

**MASSIVE 1.1, dev partition** (human labels; 1,000 parallel utterances per language for the
18-scenario task and all 582 available for the 6-scenario task; error budget 5%):

| question | language | coverage | avg set size | automated | error among sure | upper bound (95%) |
|---|---|---|---|---|---|---|
| 6 scenarios | English | 94.3% | 1.23 | 83.4% | 3 / 146 = 2.1% | 5.2% |
| 6 scenarios | Swedish | 97.7% | 2.11 | 46.3% | 1 / 81 = 1.2% | 5.7% |
| 18 scenarios | English | 94.7% | 6.98 | 0.0% | - | - |
| 18 scenarios | Swedish | 96.7% | 9.96 | 0.0% | - | - |

What these show:
- **Where the model is good enough** (6 options), Zet automates 65% overall: 4 errors among 227 sure answers.
- **Where it isn't** (18 options), Zet automates nothing. Its sets still contain the truth 95–97% of the time.
- **The language gap:** for six options, English automation is 83% and Swedish is 46%; Swedish's average set is larger (2.11 vs 1.23).
- **Caveat:** the held-out upper bounds are 5.2% for English and 5.7% for Swedish, so neither language's held-out data alone confirms the 5% budget. The combined upper bound is 4.0%. The 6-scenario `dev` subset is exhausted at 582 utterances per language.

**Direct Laya-only comparison, same checkpoint and held-out split:**

| question | Laya wrong if all answers were automatic | Zet wrong automatically | sent to review | Laya errors in review |
|---|---:|---:|---:|---:|
| 6 scenarios | 45 / 350 | 4 / 227 | 123 | 41 / 123 |
| 18 scenarios | 228 / 602 | no automatic answers | 602 | 228 / 602 |

On the 6-scenario task, Zet flagged 41 of Laya's 45 errors while asking for review of 82 correct
answers too. It did not change Laya's predictions. The benchmark did not measure actual human
review, so "errors in review" means mistakes a correct reviewer *could* fix, not observed final
pipeline accuracy. The 18-scenario task avoids automatic errors by automating nothing.

**Laya's own weak spots** (from the Windows smoke run; PyTorch Laya gives the same top answers):
- "No rush" and "Inte bråttom" came out *critical* (0.67 and 0.80);
- "I don't want a refund" came out refund = true (0.73; 0.95 in Swedish).

**Support-email fixture run on real weights** (`pytest --run-model -s -k fixture_run`): 125
synthetic labeled emails, 39 held out, four questions, 5% budget. Zet marked 0% of answers sure
for every question. This fixture is for correctness testing, not published performance; at this
size, it provides no automation benefit.

**CI:** the GitHub Actions matrix (Windows, Ubuntu, macOS; Python 3.10–3.14) passed at
`03d3905`. The current comparison changes pass the full local offline suite (133 passed,
3 skipped); the new public commit must run through CI as well.

## Not yet measured

- **English-checkpoint equivalence** (needs English weights, F1).
- **The lingua language detector**, which has never run.
