# Zet

**The decision model that learns your task and knows when to ask a human.**

Zet answers typed questions about text: pick an option (`choice`), pick a level (`score`), or
yes/no (`noul`). It runs existing open decision models, starting with
[Laya](https://github.com/NandhaKishorM/laya), without PyTorch. It then marks every answer
**sure** or **unsure** using a rule calibrated on labeled examples.

[Website](https://yoosseph.github.io/Zet/) · [Simple benchmark](benchmark.md) ·
[Source code](https://github.com/Yoosseph/Zet)

- **Knows when it doesn't know** (0.1). Zet tests a rule against labeled examples to limit errors
  among `sure` answers; the rest go to a human.
- **Learns your task** (0.2, planned). Improves from labeled examples, a teacher LLM and your
  corrections.

## Does Zet actually help?

On a test of **350 human-labeled voice-assistant requests**, Laya alone got **45 wrong**. With
Zet, **41 of those mistakes went to review**, **4 wrong answers still went through**, and **227
answers were accepted automatically**. Zet also asked for review of 82 answers Laya had right.
The test did not include actual human corrections. [Read the simple benchmark, its limits, and
sources](benchmark.md).

> Status: 0.1, in development. Not on PyPI yet.

## Install

```
git clone https://github.com/Yoosseph/Zet && cd Zet
pip install -e .
```

Needs Python 3.10+ on Windows, Linux or macOS. PyTorch is never needed: Zet runs Laya's ONNX
weights with ONNX Runtime (about 1.3 GB, downloaded on first use).

## The interface

On Windows, double-click **`Zet.bat`** in the repository folder. The first run sets everything up; after that Zet opens in your browser. On any system, `zet ui` does the same.

Everything Zet does is in the interface:
- create tasks and write their questions;
- upload labeled examples;
- calibrate, and read the results per language;
- try messages;
- work through the review queue with the keyboard;
- set spot checks and switch models;
- check the model and run the benchmark.

It runs only on your computer (127.0.0.1), and tasks are saved in `C:\Users\<you>\zet` (set `ZET_ROOT` to change that).

## Quickstart (Python)

Run from the repo folder; it uses the example support emails that ship with the tests.

```python
import yaml, zet

questions = yaml.safe_load(open("tests/fixtures/support/questions.yaml", encoding="utf-8"))
task = zet.Task.create("support", questions, backend="laya-onnx")
task.add_examples("tests/fixtures/support/emails.csv")   # labeled examples: body + one column per question
task.calibrate(error_budget=0.05)                         # at most 5% errors among sure answers

r = task.predict({"body": "Kan ni skicka om fakturan? Inte bråttom."})
a = r["urgency"]
print(a.answer, a.status, a.options)   # e.g. "2 unsure ['0', '1', '2']": ask a human
task.correct(r.id, urgency="not urgent")
task.report()
```

The same from the command line:

```
zet init support --questions tests/fixtures/support/questions.yaml
zet add-examples support tests/fixtures/support/emails.csv
zet calibrate support --error-budget 0.05
zet predict support "Kan ni skicka om fakturan? Inte bråttom."
zet report support
```

Every command has `--help` with an example. More in [docs/HOW-TO-RUN.md](docs/HOW-TO-RUN.md).

## What an answer contains

| field | meaning |
|---|---|
| `answer` | the most probable option (score questions: the level index, `"0"` is the first level) |
| `status` | `sure` or `unsure` |
| `options` | the prediction set: the options that together contain the truth at the calibrated rate |
| `probs` | the model's probability for every option |
| `audit` | `True` for the share of sure answers sampled for a human spot check |

## What "sure" guarantees, and what it doesn't

**Where the guarantee comes from.** Your calibration data. Benchmarks in this repository only
demonstrate the method; they say nothing about your task.

It is made of two separate guarantees:

- **Prediction sets** (conformal prediction). `options` contains the true answer at least
  1 − α of the time.
- **Sure answers** (selective prediction, Learn-then-Test). With 95% confidence, the error rate
  among `sure` answers is at most your `error_budget`.

Both are calibrated **per language**, because models behave very differently across languages.
A language with too few examples falls back to the pooled calibration, and the result says so.
Every number Zet reports is measured on a held-out 30% of your examples, never on the data it
was fitted on.

**What it assumes.** New data resembles the calibration data. When your traffic changes, the
guarantee can silently stop holding. Zet catches this in two ways:

- **Audits.** A share of sure answers (default 2%) is flagged for a human check. Confident
  mistakes never show up as `unsure`, so they must be sampled.
- **Drift warnings.** When audited sure answers go wrong more often than the budget allows,
  `predict()` and `report()` say to recalibrate.

**What it doesn't do.** It doesn't make the model smarter. Out of the box, Laya is confidently
wrong on some easy cases. In our Windows test it rated *"No rush"* and *"Inte bråttom"* as
**critical** (0.67 and 0.80), and read *"I don't want a refund"* as a refund request.
Calibration's job is to keep answers like these from being marked `sure`. Fixing the model
itself is 0.2's job.

**How much data.** Below 100 labeled examples per question, every result carries a warning that
the guarantee is weak. To pass a 5% budget, at least 59 answers must be sure in calibration.
With little data, expect mostly `unsure`: that's the honest answer.

## Results so far

The six-topic result above is Zet's strongest measured benefit so far. On the harder 18-topic
version of the same [MASSIVE benchmark](benchmarks/massive/results.md), Zet sent all 602 held-back
requests to review. On a small synthetic support-email test, it accepted no answers automatically
at a 5% error budget. See [benchmark.md](benchmark.md) for the plain-language comparison and
[docs/summary-0.1.md](docs/summary-0.1.md) for the technical details.

## Roadmap

- **0.1** (now): backends, tasks, calibration with an error budget, sure/unsure answers,
  per-language calibration, audits, drift warnings, report, CLI, and local review UI.
- **0.1.x**: synthetic multilingual email benchmark, human-verified per language
  ("verify a language").
- **0.2**: teacher LLM labeling, training from corrections, a regression gate, and rollback.
- **0.3**: Jev-compatible HTTP server, review UI improvements, and an automation-over-time chart.

## Acknowledgements

Zet is built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Convai
Innovations, and runs the ONNX conversion `soyelmismo/laya-multilingual-onnx` (Apache-2.0).
Benchmarks use [MASSIVE](https://github.com/alexa/massive) (CC BY 4.0), derived from SLURP
(CC BY 4.0). Zet is not endorsed by any of them. See `THIRD_PARTY_NOTICES.md`.

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
[docs/PRIVACY.md](docs/PRIVACY.md) before sharing code or examples. Changes are listed in
[CHANGELOG.md](CHANGELOG.md).
