# How to run Zet

Commands are for **Windows PowerShell**. On Linux or macOS, use the line marked `# Linux/macOS` instead.

No PyTorch is needed for any of this, except section 5, which runs Laya itself in WSL.

---

## The easy way: the interface

Double-click **`Zet.bat`** in the Zet folder. The first time, it installs Zet (a few minutes); after that it opens Zet in your browser. Keep the black window open while you use Zet; close it to stop.

Everything below can be done in the interface too: creating tasks, examples, calibration, trying messages, review, spot checks, switching models, the model check and the benchmark. The rest of this guide is for doing the same from a terminal or from Python.

Tasks are saved in `C:\Users\<you>\zet`, whether you use the interface or the terminal.

---

## 1. Install (once)

```powershell
cd Zet
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -e ".[test]"
```

If PowerShell refuses to run `Activate.ps1`, run this once, then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Every new terminal: `cd Zet` and activate again (the `Activate.ps1` line).

---

## 2. Run the tests

```powershell
python -m pytest
```

These run offline, with no download, in about 10 seconds. The expected result is all tests passing and 3 skipped. The skipped ones need real weights (sections 5 and 9).

---

## 3. Check the real model works (smoke test)

```powershell
python -m zet.smoke                                  # multilingual model, ~1.3 GB download first time
python -m zet.smoke --model C:\path\to\english-export  # English: only from a local export (no hosted copy yet)
```

What you should see:

- answers for 3 English and 3 Swedish emails;
- `torch imported: False` at the end;
- a report file: `zet-smoke-multilingual.json` or `zet-smoke-english.json`.

A line that starts with `FAIL` means something is wrong. Downloads are cached, so the second run is fast.

---

## 4. Use it from Python

```python
from zet import LayaOnnxBackend

questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": ["billing", "technical", "sales", "other"]},
    "urgency": {"type": "score", "instructions": "How urgent is it?",
                "criteria": ["not urgent", "soon", "critical"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
}

backend = LayaOnnxBackend()      # "auto": multilingual for English too until English weights are hosted
result = backend.predict_proba([{"body": "Kan ni skicka om fakturan? Inte bråttom."}], questions)
print(result[0])
# {'department': {'billing': ..., 'technical': ..., ...}, 'urgency': {'0': ..., '1': ..., '2': ...},
#  'refund': {'false': ..., 'true': ...}}
```

- **Pick one model instead of `auto`:** `LayaOnnxBackend(model="multilingual")`. For English,
  export the checkpoint locally and pass its folder as `model=`, or pass `english=<folder>` to
  `LayaOnnxBackend(model="auto")`.
- **Use a folder of ONNX files on disk:** `LayaOnnxBackend(model=r"C:\models\laya")`.
- **See the language and which model answered:** `backend.predict(...)` returns the same probabilities, plus `.meta` with those details.

---

## 5. Compare with Laya's PyTorch version

This checks that Zet gives the same answers as the original Laya. It needs PyTorch, so use a **separate** venv; Zet's own venv must stay torch-free.

```powershell
python -m venv .venv-laya
.venv-laya\Scripts\Activate.ps1
pip install laya
python scripts/make_laya_reference.py --checkpoint multilingual --revision main
deactivate
```

This writes `tests\fixtures\equivalence\reference-multilingual.json` and records which Laya commit `main` was.

**Back in Zet's venv:**

```powershell
.venv\Scripts\Activate.ps1
python -m pytest --run-model -s
```

The result should be all tests passing, with a line like `max |p_zet - p_laya| = 9.00e-04`. The English check is skipped until there are English weights.

---

## 6. Use a running Laya server instead (optional)

**In WSL:**

```bash
pip install "laya[serve]"
laya-serve                      # listens on http://localhost:8000
```

**In Python on Windows:**

```python
from zet import LayaHttpBackend
backend = LayaHttpBackend("http://localhost:8000")
```

It works exactly like section 4.

---

## 7. Rebuild the tiny test models (rarely needed)

Only needed if you change `scripts/make_tiny_model.py`.

```powershell
pip install -e ".[dev]"
python scripts/make_tiny_model.py
```

---

## 8. Your own task, from the command line

Put your questions in a YAML file (copy `tests\fixtures\support\questions.yaml` as a start) and your labeled examples in a CSV: a `body` column, one column per question with the true answer, and optionally a `language` column.

```powershell
zet init support --questions tests\fixtures\support\questions.yaml
zet add-examples support tests\fixtures\support\emails.csv
zet calibrate support --error-budget 0.05
zet predict support "Kan ni skicka om fakturan? Inte bråttom."
zet correct support <id-from-predict> urgency="not urgent"
zet report support
```

Everything is saved under `zet\support\`, so `predict` works in a new terminal without recalibrating. Answers marked `AUDIT` are the random spot checks: confirm them with `zet confirm support <id>`, or correct them. That's how Zet notices drift.

---

## 9. Measured numbers on the example emails (real model)

```powershell
python -m pytest --run-model -s -k fixture_run
```

This calibrates the real Laya model on the 125 example emails and prints the report. With this little data, expect a low automation rate: that's the honest result.

---

## 10. MASSIVE benchmark (the published numbers)

```powershell
python benchmarks\massive\run.py --per-language 1000
```

This downloads MASSIVE (about 40 MB, once), samples up to 1,000 English and 1,000 Swedish
utterances per question, and writes `benchmarks\massive\results.md`. The six-scenario `dev`
subset has only 582 per language. The run takes a few minutes on CPU. Add more languages with
`--languages en-US sv-SE de-DE`. Commit the results:

```powershell
git add benchmarks\massive\results.md benchmarks\massive\results.json
git commit -m "MASSIVE results"
git push
```

---

## Problems

| You see | Do this |
|---|---|
| `No module named zet` | Activate the venv (section 1) and run from the `Zet` folder. |
| `RuntimeWarning: ... not pinned to a revision` | Normal for now. Send the smoke report JSON to get the weights pinned. |
| Download is slow or fails | Run it again; finished parts are kept. |
| `torch imported: True` | A bug: the core must never load PyTorch. Report it with the smoke JSON. |
| `has no attention_mask input` | Your split ONNX export is outdated; re-export with Laya's current `laya-ts/scripts/export_onnx.py`. |
