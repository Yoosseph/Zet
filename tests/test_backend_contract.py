"""One contract, every backend: shape, all options present, probabilities sum to 1 within 1e-6.

The HTTP backend runs against a local stub that answers exactly as laya-serve does: probabilities
rounded to 4 decimals, noul as p(true) only (docs/laya-notes.md, "Output format").
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from conftest import QUESTIONS, STATES, TINY
from zet.backends import Backend, FakeBackend, LayaHttpBackend, LayaOnnxBackend
from zet._laya.questions import option_keys


class _LayaStub(BaseHTTPRequestHandler):
    """Answers POST /v1/systemone in Laya's wire format, using FakeBackend's numbers."""
    fake = FakeBackend(seed=7)
    requests: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])).decode("utf-8"))
        type(self).requests.append(body)
        probs = self.fake.predict_proba([body["state"]], body["questions"])[0]
        answers = {}
        for qid, qdef in body["questions"].items():
            p = probs[qid]
            if qdef["type"] == "noul":
                answers[qid] = {"noul": round(p["true"], 4), "confidence": 0.5,
                                "action": {"act_probability": 0.9}}
            else:
                key = "choice" if qdef["type"] == "choice" else "score"
                answers[qid] = {key: max(p, key=p.get),
                                "probabilities": {k: round(v, 4) for k, v in p.items()},
                                "action": {"act_probability": 0.9}}
        out = json.dumps({"model": body.get("model"), "answers": answers,
                          "routing": {"model": body.get("model")}}).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def laya_stub():
    server = HTTPServer(("127.0.0.1", 0), _LayaStub)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:%d" % server.server_address[1]
    server.shutdown()


@pytest.fixture(params=["fake", "onnx-single", "onnx-split", "http"])
def backend(request, laya_stub):
    if request.param == "fake":
        return FakeBackend(seed=1)
    if request.param == "onnx-single":
        return LayaOnnxBackend(model=str(TINY / "single"))
    if request.param == "onnx-split":
        return LayaOnnxBackend(model=str(TINY / "split"))
    return LayaHttpBackend(laya_stub, model="multilingual")


def test_satisfies_protocol(backend):
    assert isinstance(backend, Backend)
    assert isinstance(backend.name, str) and backend.name


def test_shape_options_and_sums(backend):
    out = backend.predict_proba(STATES, QUESTIONS)
    assert len(out) == len(STATES)
    for probs in out:
        assert set(probs) == set(QUESTIONS)
        for qid, qdef in QUESTIONS.items():
            p = probs[qid]
            assert list(p) == option_keys(qdef), qid       # every option, in model order
            assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in p.values())
            assert abs(sum(p.values()) - 1.0) <= 1e-6, (qid, sum(p.values()))


def test_option_key_conventions(backend):
    probs = backend.predict_proba(STATES[:1], QUESTIONS)[0]
    assert set(probs["refund_requested"]) == {"true", "false"}
    assert list(probs["urgency"]) == ["0", "1", "2"]
    assert list(probs["department"]) == ["billing", "technical", "sales", "other"]


def test_deterministic_and_batch_independent(backend):
    together = backend.predict_proba(STATES, QUESTIONS)
    again = backend.predict_proba(STATES, QUESTIONS)
    one_by_one = [backend.predict_proba([s], QUESTIONS)[0] for s in STATES]
    for a, b, c in zip(together, again, one_by_one):
        for qid in QUESTIONS:
            for opt in a[qid]:
                assert a[qid][opt] == pytest.approx(b[qid][opt], abs=1e-12)
                assert a[qid][opt] == pytest.approx(c[qid][opt], abs=1e-6)


def test_detect_language_returns_a_code(backend):
    assert backend.detect_language("Could you resend last month's invoice? No rush.") == "en"
    assert backend.detect_language("Jag vill inte ha pengarna tillbaka, bara en fungerande app.") == "sv"
    assert isinstance(backend.detect_language("ok"), str)


def test_embed_is_none_or_a_matrix(backend):
    emb = backend.embed(["first text", "second text"])
    assert emb is None or (isinstance(emb, np.ndarray) and emb.ndim == 2 and emb.shape[0] == 2)


def test_rejects_invalid_questions(backend):
    with pytest.raises(ValueError, match="unknown type"):
        backend.predict_proba(STATES[:1], {"q": {"type": "rank", "instructions": "x"}})
    with pytest.raises(ValueError, match="not unique as strings"):
        backend.predict_proba(STATES[:1], {"q": {"type": "choice", "instructions": "x",
                                                 "criteria": [1, "1"]}})


def test_single_and_split_formats_agree():
    single = LayaOnnxBackend(model=str(TINY / "single")).predict_proba(STATES, QUESTIONS)
    split = LayaOnnxBackend(model=str(TINY / "split")).predict_proba(STATES, QUESTIONS)
    for a, b in zip(single, split):
        for qid in QUESTIONS:
            assert a[qid] == pytest.approx(b[qid], abs=1e-9)


def test_http_sends_explicit_checkpoint_and_rebuilds_noul(laya_stub):
    _LayaStub.requests.clear()
    b = LayaHttpBackend(laya_stub)  # model="auto": Zet routes, the server doesn't
    preds = b.predict(STATES[:2], QUESTIONS)
    assert [r["model"] for r in _LayaStub.requests] == ["english", "multilingual"]
    assert [p.meta["checkpoint"] for p in preds] == ["english", "multilingual"]
    ref = FakeBackend(seed=7).predict_proba(STATES[:1], QUESTIONS)[0]["refund_requested"]
    got = preds[0].probs["refund_requested"]
    assert got["true"] == pytest.approx(ref["true"], abs=1e-4)
    assert got["false"] == pytest.approx(1 - got["true"], abs=1e-12)
