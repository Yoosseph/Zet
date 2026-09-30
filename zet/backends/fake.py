"""FakeBackend: deterministic, configurable probabilities. Used by unit tests and simulations.

Three ways to set the probabilities, checked in this order:
1. `fn(state, question_key, question_def, option_keys) -> {option: probability}` for full control
   (calibration simulations use this to plant a known error rate).
2. `fixed={question_key: {option: probability}}`, the same answer for every state.
3. Otherwise a pseudo-random distribution derived from a hash of (seed, state, question), so the
   same inputs always give the same output on every platform.

The language comes from a state's `language` field when present, else the built-in detector.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Dict, List, Optional

import numpy as np

from .base import BaseBackend, Prediction, State, normalise, prepare_questions

ProbFn = Callable[[State, str, Dict, List[str]], Dict[str, float]]


def _stable_seed(*parts) -> int:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


class FakeBackend(BaseBackend):
    name = "fake"

    def __init__(self, fn: Optional[ProbFn] = None, *, fixed: Optional[Dict[str, Dict[str, float]]] = None,
                 seed: int = 0, concentration: float = 0.5, embed_dim: int = 16):
        self.fn = fn
        self.fixed = fixed or {}
        self.seed = seed
        self.concentration = float(concentration)
        self.embed_dim = int(embed_dim)
        self.calls = 0  # number of predict() calls, for tests

    def _random(self, state: State, qid: str, keys: List[str]) -> Dict[str, float]:
        rng = np.random.default_rng(_stable_seed(self.seed, state, qid))
        v = rng.dirichlet([self.concentration] * len(keys))
        return dict(zip(keys, v))

    def predict(self, states: List[State], questions: Dict[str, Dict]) -> List[Prediction]:
        _, keys = prepare_questions(questions)
        self.calls += 1
        out = []
        for st in states:
            probs = {}
            for qid, qdef in questions.items():
                if self.fn is not None:
                    raw = self.fn(st, qid, qdef, keys[qid])
                elif qid in self.fixed:
                    raw = self.fixed[qid]
                else:
                    raw = self._random(st, qid, keys[qid])
                probs[qid] = normalise(raw, keys[qid])
            out.append(Prediction(probs=probs, meta={"checkpoint": "fake",
                                                     "language": self.detect_language(st)}))
        return out

    def detect_language(self, text) -> str:
        if isinstance(text, dict) and isinstance(text.get("language"), str) and text["language"]:
            return text["language"]
        return super().detect_language(text)

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        vecs = [np.random.default_rng(_stable_seed(self.seed, "embed", t)).standard_normal(self.embed_dim)
                for t in texts]
        return np.array(vecs, dtype=np.float32).reshape(len(texts), self.embed_dim)
