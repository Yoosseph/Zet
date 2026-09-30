"""LayaHttpBackend: calls a running `laya-serve` (POST /v1/systemone, docs/http-api.md in Laya).

Lets Windows users run Laya's PyTorch server in WSL and Zet natively. Standard library only.

Laya rounds every probability to 4 decimals and reports noul as p(true) only, so this backend
rebuilds {"false", "true"} for noul and renormalises each distribution to sum to exactly 1. Its
probabilities can differ from the ONNX backend's by rounding (about 5e-5 per option).

Checkpoint choice matches LayaOnnxBackend: `model="auto"` routes each state by Zet's built-in
language and sends that checkpoint name explicitly, so the server's own router never decides.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from ..language import builtin_language, route_checkpoint
from .base import BaseBackend, Prediction, State, normalise, prepare_questions


class LayaHttpBackend(BaseBackend):
    name = "laya-http"

    def __init__(self, url: str = "http://localhost:8000", *, model: str = "auto",
                 api_key: Optional[str] = None, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("LAYA_API_KEY")
        self.timeout = float(timeout)

    def checkpoint_for(self, state: State) -> str:
        if self.model != "auto":
            return self.model
        return route_checkpoint(builtin_language(state))

    def _post(self, body: Dict) -> Dict:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.url + "/v1/systemone", data=data, method="POST")
        req.add_header("content-type", "application/json")
        if self.api_key:
            req.add_header("authorization", "Bearer " + self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"Laya server returned HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"cannot reach a Laya server at {self.url}: {e.reason}. Start one "
                                  "with `laya-serve` (pip install laya[serve]).") from e

    @staticmethod
    def _to_probs(answer: Dict, qtype: str, keys: List[str]) -> Dict[str, float]:
        if qtype == "noul":
            p = float(answer["noul"])
            return normalise({"false": 1.0 - p, "true": p}, keys)
        values = list(answer["probabilities"].values())
        if len(values) != len(keys):
            raise ValueError(f"server returned {len(values)} probabilities for {len(keys)} options")
        # Laya's `probabilities` follow option order; JSON may re-key labels (True -> "true").
        return normalise(dict(zip(keys, values)), keys)

    def predict(self, states: List[State], questions: Dict[str, Dict]) -> List[Prediction]:
        _, keys = prepare_questions(questions)
        out = []
        for st in states:
            ck = self.checkpoint_for(st)
            resp = self._post({"state": st, "questions": questions, "model": ck})
            answers = resp.get("answers") or {}
            probs = {}
            for qid, qdef in questions.items():
                if qid not in answers:
                    raise ValueError(f"server response has no answer for question {qid!r}")
                probs[qid] = self._to_probs(answers[qid], qdef["type"], keys[qid])
            meta = {"checkpoint": ck, "language": builtin_language(st), "routing": resp.get("routing"),
                    "act_probability": {qid: (answers[qid].get("action") or {}).get("act_probability")
                                        for qid in questions}}
            out.append(Prediction(probs=probs, meta=meta))
        return out
