# Zet design decisions

The decision log for Zet. Each entry says what was decided and, where it isn't obvious, why. IDs (Q1, F1, …) refer to the design review rounds. Laya facts behind these decisions are in [laya-notes.md](laya-notes.md).

Process rule: questions are raised only when they block the current step. Everything else goes under [Open items](#open-items).

Commit rule: all of 0.1 fits in at most 20 commits, about two per build step. Steps 1–2 took 9 (8 plus the how-to-run guide); steps 3–7 get the remaining 11 at most. Each commit is one working unit that passes the tests, not every small edit.

---

## Scope and framing

- **Where the guarantee comes from.** Zet's guarantee comes from each user's own calibration data. Benchmarks only demonstrate the method. The README says this plainly.
- **Milestones.**
  - **0.1:** the library as briefed, with numbers from MASSIVE plus the small fixture set.
  - **0.1.x:** the synthetic email benchmark and the language review tool.
  - **0.2 and 0.3:** as in the brief.
- **Multilingual numbers** for 0.1 come from MASSIVE, which is human-labeled.

## 0.1 decisions

### Laya integration

- **Q1: copy, don't depend.** Laya's torch-free logic is copied into `zet/_laya/` with both headers. An upstream PR proposes splitting `laya/common.py` into a torch-free module. The copies are deleted once that PR ships in a Laya release.
- **Q2: split ONNX format** (`encoder.onnx` + `head.onnx`). *Revisited by F1, which is open.*
  - The pre-converted Hub weights are 8-bit. The equivalence test therefore uses a full-precision export, or states an explicit tolerance.
  - The multilingual checkpoint is exported by the maintainer with `laya-ts/scripts/export_onnx.py` in WSL.
  - `embed()` runs the encoder on the state alone, mean-pooled over the attention mask. This is documented in laya-notes.md.
- **Q3: calibrate on Laya's probabilities as shipped.** Zet uses the post-temperature probabilities from each checkpoint's `rl_agent_config.json` unchanged, and does not refit temperatures.

### Calibration

- **Q4: selective threshold by Learn-then-Test.** Zet scans a fixed grid of about 50 thresholds from strictest to loosest and stops at the first one that fails. The docs state the guarantee as: *with 95% confidence, the error rate among `sure` answers is at most the budget.*
- **Q6: group by the resolved language** in both calibration and prediction. The `language` column in the data is used for stratifying the split and for a "detector disagreed" count.
- **Q7: fallback order.** Label-conditional calibration applies to conformal sets only, never to the selective threshold. Fallback goes (language, label) → language → pooled, and the level actually used is recorded in `calibration_group`.
- **Q8: conformal sets are never empty.** The top option is always included.
- **Q9: calibration is stored per version.** It lives at `zet/<task>/versions/<n>/calibration.json`, next to that version's question hash. `task.yaml` holds `current_version`.
- **Q10: drift uses a count-based window.**
  - The window is the last 200 audits per group.
  - Below about 60 audits the report says "insufficient audits", not "no drift".
  - `audit_rate` can be set per group.
- **Q17: language resolution order.**
  1. An explicit `language` field in the state.
  2. The optional extra `zet[langid]` (lingua), limited to the task's declared languages.
  3. The built-in detector.
  4. `unknown`, which uses pooled calibration.
  - The method used is recorded in `calibration.json`. `predict()` always uses the recorded method and raises only if that method is unavailable. Switching methods means recalibrating explicitly.
  - The built-in detector is Laya's `lang.py`, extended with sv/no/da, so Zet's groups line up with Laya's checkpoint routing.

### Platform and CI

- **Python and OS matrix.** CI runs 3.10 through 3.14 on Windows, Ubuntu and macOS. The maintainer's local Python is 3.14.
- **Windows is checked early.** An end-to-end ONNX run on Windows is an early milestone of step 2, not a final check: `python -m zet.smoke`. CI runs the same smoke module on a tiny model on every OS and Python version.
- **Checked on the maintainer's machine** (Windows 11, Smart App Control on, Python 3.14): `onnxruntime` 1.30.0, `numpy` 2.4.1 and `tokenizers` 0.22.2 all import.

### Step 2 results (Windows 11, Python 3.14, 2026-09-29)

- **End-to-end run:** `python -m zet.smoke` on the multilingual fp32 weights. Torch was not imported; 28 ms per (email, question) on CPU; 71 unit tests pass.
- **Equivalence with Laya's PyTorch path (multilingual):**
  - token rows identical in all 48 (case, question) pairs;
  - top answer identical in all 48;
  - max probability difference 9.0e-4.
  The brief's criterion ("same top label") holds. The probability tolerance is set to 1e-3, with the reason in the test. English is pending weights (F1).
- **Laya itself fails on negation and explicit non-urgency, in both English and Swedish, confidently.** "No rush" and "Inte bråttom" both came out *critical* (0.67 and 0.80), and "I don't want a refund" came out refund = true (0.73; 0.95 in Swedish). These are model errors, not Zet's: PyTorch Laya gives the same top answers. Calibration and audits must catch them, and the report should show it.
- **Smart App Control did not block PyTorch 2.14** on the maintainer's machine (it installed and ran for the reference). Zet's core stays torch-free regardless. The reference script should run in a separate venv, so Zet's test environment keeps proving the core needs no torch.

### Data and numbers

- **Q5: fixtures and benchmarks are separate.** About 120 fixtures exist for correctness tests only. Benchmarks produce the reported numbers.
- **Q16 and Q23: MASSIVE.**
  - **Source:** the `dev` partition, never `test`, because Laya's routing was chosen on `test`. The data is downloaded when the benchmark runs, not committed. It is CC BY 4.0, with attribution to both MASSIVE and SLURP.
  - **Sample:** about 300 dev utterances per language for 0.1, with a fixed seed. A full run on GPU is optional later. The output records the sample size.
  - **Questions:** an 18-option `scenario` question, plus a second, smaller question of 5–6 scenarios, so the results aren't only about Laya's weakness with many options.
  - **Languages:** en-US and sv-SE are shown as the parallel language-gap demo.
  - **How it runs:** a separate script, `benchmarks/massive/run.py`. Its output JSON is committed with the model revision and Zet version, and the README tables are generated from it.
  - **Checkpoints:** the multilingual checkpoint runs on every language, and the English checkpoint on en-US as a comparison row.
- **Q12: publishing rule.** Error numbers are published only for languages with at least 100 human-checked labels. All other languages are marked "unverified", and the label-error upper bound is printed next to every error figure.

### Build log: steps 3–7 (2026-09-29)

- **Step 3, tasks.** Built as briefed, plus:
  - an uncalibrated task still predicts, with every answer `unsure` and a warning;
  - a `language` field in a state selects the calibration group but is removed before the state reaches the model, so the model input is identical with or without it;
  - recalibrating writes a new version (`versions/2/`, …) instead of overwriting version 1.
- **Step 4, calibration.** Built as briefed (Q4, Q6–Q9).
  - **Found by the property tests: the threshold scan needed a rule the brief didn't state.** It skips thresholds with fewer sure answers than could ever pass (59 at a 5% budget, from Clopper–Pearson with zero errors). Without that rule, the scan's first test holds a handful of answers, always fails, and stops, so nothing is ever automated. The skip depends only on counts, never on labels, so the guarantee stands.
  - **Measured:** on a simulated over-confident model with 1,000 calibration examples, 29 of 40 splits automated some answers, and none exceeded the budget. Starting the scan later made things worse (26, 10 and 5 of 40), because over-confident models make more mistakes further down.
  - **Consequence for users:** on small data, whether anything gets automated depends partly on the split.
- **Step 5, audits and drift.** Audit sampling is uniform and reproducible from the prediction id. Drift counts only audited sure answers from the current calibration version, so recalibrating starts a fresh window.
- **Step 6, report and CLI.** Built as briefed. Examples the detector can't identify ("unknown") don't count as language disagreements.
- **Step 7, fixtures, benchmark, docs.**
  - 125 support emails, 25 in Swedish, LLM-written. They are for correctness tests and a demo run, not for published numbers (Q12).
  - The MASSIVE script is tested offline; its real run needs the maintainer's machine.
- **Routing while English weights are missing (F1).** `model="auto"` answers English with the multilingual checkpoint until an English export is given: `LayaOnnxBackend(english=<folder>)`.

### MASSIVE results (Windows, 2026-09-30)

Initial run: multilingual checkpoint, 300 parallel dev utterances per language, 5% budget.

- **6 scenarios:** automated 84.3% (English, 0/75 errors among sure answers) and 79.8% (Swedish, 3/71).
- **18 scenarios:** nothing automated. Average set size 7.4 (English) and 11.6 (Swedish); coverage 96.6% and 98.9%.
- **Conclusion:** the method behaves as designed on both a good and a weak setting.
- **Held-out samples were small (89 per language).** The Swedish held-out upper bound was 10.6%,
  so the initial held-out data alone could not confirm the 5% budget.

Expanded run: requested 1,000 parallel `dev` utterances per language. All 582 available
six-scenario utterances per language were used; the 18-scenario task used 1,000. Full tables:
`benchmarks/massive/results.md`.

- **Six scenarios:** on 350 held-out requests, Laya-only had 45 top-answer errors. Zet flagged
  41 of them among 123 reviewed requests, leaving 4 errors among 227 automatic answers.
- **Eighteen scenarios:** on 602 held-out requests, Laya-only had 228 errors. Zet reviewed all 602
  and automated nothing.
- **Why report both:** raw top-answer errors on the *same held-out split* show how many model
  mistakes the review gate flags and how much review work that costs. The benchmark has no human
  review outcomes, so flagged errors are only mistakes a correct reviewer could fix. Zet does not
  improve Laya's labels.
- **Limits:** the six-scenario held-out upper bounds are 5.2% for English and 5.7% for Swedish
  (4.0% pooled), so neither language is independently confirmed below 5% by the held-out set.

### Interface (2026-09-30)

Built ahead of 0.3 at the maintainer's request: every Zet function, with no terminal needed.

- **Start it:** `Zet.bat` (double-click; installs on first run), `zet ui`, or `python -m zet.ui`.
- **Server:** a Python standard-library HTTP server on 127.0.0.1 with a JSON API, plus one self-contained page (no external libraries or fonts, so it works offline).
- **Protection from other websites:** POST requests need an `X-Zet: 1` header. Browsers won't send a custom header cross-site without a CORS preflight, which the server never grants.
- **Slow work** (model loading or download, calibration, the benchmark) runs as background jobs with progress.
- **Layout:** five tabs per task: Try, Review, Examples, Calibration, Settings. Plus "Model check and benchmark", and a task builder.
- **Review:** keyboard-driven. Number keys answer the next unanswered question, A accepts the model's answers, Enter saves.
- **Tested:** over real HTTP (`tests/test_ui_api.py`), and by driving the page in headless Chromium through the full flow with zero browser errors. The browser test isn't in CI; run it by hand.

**Library changes made for it:**
- Tasks now default to `$ZET_ROOT` or `~/zet`, no longer `./zet`. Run from the repository, `./zet` *is* the package source folder, so tasks landed inside the code.
- Switching a task's model makes its calibration stale, like changing questions does. `Task.set_backend`.
- `Task.review_queue()` and `Task.verdicts()` added.
- `calibrate(progress=...)` reports progress.

**After first use on Windows (2026-09-30):**
- **Delete task** is recoverable: the folder is renamed to `.deleted-<name>-<time>`, and deleting requires typing the task's name. Prompted by a task created with a message pasted into a question field; questions can't be edited afterwards.
- **The task builder asks for confirmation** when a question doesn't end with "?", because the model is asked exactly that text about every message.
- **Uncalibrated results** say "no option is ruled out" instead of pointing at faded options that don't exist.
- **Examples can be removed and restored**, one at a time, with Undo. `examples.jsonl` stays append-only: removing appends `{"remove": id}`, restoring appends `{"restore": id}`, and `Task.examples()` applies them. The full history survives for 0.2 training data. Ids are numbered over everything ever added, so a removal never leads to a reused id. A calibration keeps what it measured until you recalibrate.

### Open items after 0.1

- **F1:** host English and multilingual exports on the maintainer's Hugging Face account; then the English equivalence test runs.
- **Fixture run:** `pytest --run-model -s -k fixture_run` passed on real multilingual weights.
  With 125 synthetic support emails (39 held out), none of the four questions automated any
  answers at a 5% budget. The larger MASSIVE run is complete; its aggregate results are
  committed without the source dataset.
- **Not yet tested for real:** the lingua language detector (`zet[langid]`). The code path exists but has never run.
- **Upstream to Laya** (unchanged): the torch-free `common.py` split, sv/no/da stopwords, and the attribution question.

## 0.1.x decisions (recorded, not built)

- **Synthetic email benchmark.**
  - **Generation:** labels are sampled from a documented prior table in `benchmarks/README.md` (Q13). Two generators from different providers write the emails, with style controls (Q14).
  - **Labeling (Q11c):** a model from a provider that didn't generate the email labels it blind. When the labeler and generator disagree, a third model adjudicates. Disagreement rates are reported per slice, and off-by-one on `score` questions is counted separately.
  - **Generators and license (Q15):** open-weight Apache-2.0/MIT generators are preferred. The benchmark is released under CC BY 4.0 with a dataset card.
  - **Languages:** many languages are generated, and each stays "unverified" until it passes review. The maintainer reviews Swedish.
- **Teacher separation.** The 0.2 teacher LLM must be a different model from the ones used to build the benchmark.
- **Q18: the tool draws the review sample.** A fixed-seed random sample is drawn from the frozen benchmark version. Items can't be skipped or reordered. "Can't judge" counts against the language. The review file records the benchmark hash, the seed and the sample ids.
- **Q19: review is blind.** For each email the reviewer judges naturalness, then assigns labels without seeing the benchmark's. The tool resolves disagreements afterwards.
- **Q20: what "verified" means.** A merged review of at least 100 examples (larger reviews are allowed) whose label-error upper bound is at most 5%.
  - A language above the threshold is marked "reviewed, labels too noisy" and is regenerated.
  - Verification is tied to a benchmark version; regenerating a language resets it.
- **Q21: one reviewer is enough.** The file records the reviewer's GitHub handle. When a second reviewer exists, their agreement is reported, and a large gap resets the language to unverified.
- **Q22: the review tool.**
  - **Command:** `zet bench review --lang <code>` uses terminal prompts only, can be resumed, and adds no dependencies.
  - **Output:** `benchmarks/reviews/<lang>/<handle>-<hash>.jsonl`. CI validates every review file.
  - **Contributors:** CONTRIBUTING.md documents the workflow as "Verify a language".
  - **Shared logic:** the review core is reused by 0.2's correction queue and 0.3's web UI.

---

## Open items

### Adopted provisionally in step 2 (not confirmed)

The maintainer said "proceed" without answering these, so step 2 was built with the recommended option. Each is isolated enough to change cheaply.

- **F1: weights and format. OPEN, needs a decision.** The provisional default, `distinctinteractive/laya-onnx`, went offline on 2026-09-29 (401 for everyone). This is the risk of depending on a stranger's copy of the weights.
  - **Current state:** multilingual comes from `soyelmismo/laya-multilingual-onnx` `model-fp32.onnx`, pinned to commit 0966c4f. English has no hosted full-precision export in the project configuration. `model="english"` needs a local export; `model="auto"` uses multilingual for English until an English export is supplied.
  - **Recommended:** host both checkpoints on the maintainer's own Hugging Face account (e.g. `Yoosseph/laya-onnx`), exported with Laya's `export_onnx.py`. PyTorch now installs on the maintainer's Windows machine, so this doesn't need WSL.
  - *Where to change it:* `zet/backends/weights.py` (`SOURCES`, `NO_SOURCE`).
- **F3: header for copied files.** Copied files carry `Copyright the Laya authors (Convai Innovations). Licensed under the Apache License, Version 2.0.` plus the "Modified from" line.
  - *Where to change it:* the headers in `zet/_laya/*.py` and THIRD_PARTY_NOTICES.md.
- **Q24: which checkpoint answers.** The original decision was to route `en` to English and other
  languages to multilingual. The current ONNX backend falls back to multilingual for English
  because no English export is configured; `english=<folder>` restores that route. Pinning one
  checkpoint is also supported. The HTTP backend sends an explicit model.
  - *Where to change it:* `zet/language.py` (`route_checkpoint`).

### Non-blocking

- **Upstream PRs to Laya:** a torch-free split of `common.py`; sv/no/da stopwords in `lang.py`; an issue asking for the preferred attribution wording.
- **lingua needs Python ≥ 3.12**, and each wheel is about 170 MB. `zet[langid]` can therefore be installed only on 3.12+. The docs must say this, and CI must test the extra only there.
- **HTTP backend and rounding.** Laya rounds probabilities to 4 decimals. `LayaHttpBackend` renormalises them, so the 1e-6 sum-to-1 contract still holds. What rounding affects is agreement with the ONNX backend, about 5e-5 per option.
- **How a state is sent to the model: settled, no change.** A dict state like `{"body": ...}` reaches the model as JSON. Measured on Windows against plain text of the same email: urgency 0.76 vs 0.79 critical, refund 0.90 vs 0.91 true — no meaningful difference, so states are sent as given.
- **Tokenizer parity test.** Zet reads `tokenizer.json` with `tokenizers`, and its token ids must match `transformers.AutoTokenizer` on every fixture. The test runs with `@pytest.mark.model`.
- **Option collisions.** Laya reports options whose token spans became identical after truncation. Zet surfaces these as a task warning.
- **`act_probability`** is kept in `predictions.jsonl`, unused in 0.1, as a possible 0.2 feature.
- **The MASSIVE 18-option question** falls in Laya's `choice:11+` temperature bucket, whose shipped value is clamped. Expect wide sets and low automation, and say so next to the numbers.
- **0.3 server.** Laya already serves `POST /v1/systemone`. Zet's server has to add value (status, options, audit), not duplicate it.
- **Swedish fixtures (step 7).** The maintainer won't hand-write Swedish, yet about 20% of roughly 120 fixtures should be Swedish, including negation and "Inte bråttom" cases. A source is still to be decided; one option is LLM-generated fixtures reviewed by the maintainer.
