"""Zet's torch-free path against Laya's PyTorch path (step 2 "done when").

Needs real weights, so it runs only with --run-model, and needs a reference recorded by
scripts/make_laya_reference.py (run in WSL). Checks, per case and question:
1. token rows are identical (tokenizer and sequence-building parity);
2. the top option is identical;
3. every probability is within ZET_EQUIV_TOL (default 1e-3). Measured on Windows, 2026-09-29:
   9.0e-4 for soyelmismo's fp32 multilingual export against Laya `main`, with identical token
   rows and identical top answers in all 48 (case, question) pairs. Laya rounds to 4 decimals
   (5e-5), so the rest is export-revision or ONNX Runtime graph-optimisation drift, which moves
   no answer. Quantized weights need a looser, explicitly stated tolerance.

Weights come from the default Hub source; set ZET_MODEL_DIR_ENGLISH / ZET_MODEL_DIR_MULTILINGUAL
to test a local export instead (e.g. your own split export).
"""
from __future__ import annotations

import json
import os

import pytest

from conftest import FIXTURES
from zet._laya.questions import option_keys, to_internal
from zet.backends import LayaOnnxBackend

TOL = float(os.environ.get("ZET_EQUIV_TOL", "1e-3"))


def _reference(checkpoint):
    path = FIXTURES / "equivalence" / f"reference-{checkpoint}.json"
    if not path.is_file():
        pytest.skip(f"no {path.name}; record it with scripts/make_laya_reference.py")
    return json.loads(path.read_text(encoding="utf-8"))


def _laya_probs(answer, qdef):
    keys = option_keys(qdef)
    if qdef["type"] == "noul":
        return {"false": 1.0 - answer["noul"], "true": answer["noul"]}
    return dict(zip(keys, answer["probabilities"].values()))


@pytest.mark.model
@pytest.mark.parametrize("checkpoint", ["english", "multilingual"])
def test_matches_laya_pytorch(checkpoint):
    ref = _reference(checkpoint)
    model = os.environ.get(f"ZET_MODEL_DIR_{checkpoint.upper()}", checkpoint)
    backend = LayaOnnxBackend(model=model)
    ck = backend._checkpoint(model)
    questions = ref["questions"]
    internal = {q: to_internal(d) for q, d in questions.items()}

    worst, mismatched_top = 0.0, []
    for case in ref["cases"]:
        rows = ck.rows_for_state(case["state"], list(questions), internal)
        for qid, row in zip(questions, rows):
            assert row["ids"] == case["rows"][qid]["ids"], f"token row differs: {case['state']} / {qid}"
            assert row["markers"] == case["rows"][qid]["markers"]
        zet = backend.predict_proba([case["state"]], questions)[0]
        for qid, qdef in questions.items():
            laya = _laya_probs(case["answers"][qid], qdef)
            if max(zet[qid], key=zet[qid].get) != max(laya, key=laya.get):
                mismatched_top.append((case["state"], qid, zet[qid], laya))
            worst = max(worst, max(abs(zet[qid][k] - laya[k]) for k in laya))
    print(f"{checkpoint}: max |p_zet - p_laya| = {worst:.2e} over {len(ref['cases'])} cases")
    assert not mismatched_top, mismatched_top
    assert worst <= TOL, f"max probability difference {worst:.2e} > {TOL:.0e}"
