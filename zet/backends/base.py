"""The backend layer: (states, questions) -> per-option probabilities.

Everything above this layer is backend-agnostic. A backend is anything that satisfies `Backend`.
The three built-in backends also share `BaseBackend`, which adds `predict`: the same
probabilities plus per-state metadata (which checkpoint answered, the resolved language, Laya's
act probability) that tasks store in `predictions.jsonl`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import numpy as np

from .._laya.questions import option_keys, validate_questions

State = Union[str, Dict[str, Any], List[Any]]
Probs = Dict[str, Dict[str, float]]  # {question_key: {option: probability}}


@runtime_checkable
class Backend(Protocol):
    name: str

    def predict_proba(self, states: List[State], questions: Dict[str, Dict]) -> List[Probs]:
        """For each state: {question_key: {option: probability}}. Probabilities per question sum
        to 1. noul questions use options {"true", "false"}; score questions use level indices
        as strings ("0".."k-1"); choice questions use their labels as strings."""
        ...

    def detect_language(self, text: str) -> str:
        """A language code, or "unknown"."""
        ...

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """(n, d) float32 embeddings, or None when the backend has none (allowed in 0.1)."""
        ...


@dataclass
class Prediction:
    probs: Probs
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseBackend:
    name = "base"

    def predict(self, states: List[State], questions: Dict[str, Dict]) -> List[Prediction]:
        raise NotImplementedError

    def predict_proba(self, states: List[State], questions: Dict[str, Dict]) -> List[Probs]:
        return [p.probs for p in self.predict(states, questions)]

    def detect_language(self, text: str) -> str:
        from ..language import builtin_language

        return builtin_language(text)

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        return None


def normalise(probs: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    """Exactly `keys`, non-negative, summing to 1 (float64). Raises on an unusable distribution."""
    missing = [k for k in keys if k not in probs]
    extra = [k for k in probs if k not in keys]
    if missing or extra:
        raise ValueError("options do not match the question: missing %s, unexpected %s" % (missing, extra))
    v = np.array([float(probs[k]) for k in keys], dtype=np.float64)
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError("probabilities must be finite and non-negative, got %r" % (probs,))
    total = v.sum()
    if total <= 0:
        raise ValueError("probabilities sum to zero: %r" % (probs,))
    v = v / total
    return {k: float(x) for k, x in zip(keys, v)}


def prepare_questions(questions: Dict[str, Dict]):
    """Validate questions; return (internal forms, option keys per question)."""
    internal = validate_questions(questions)
    keys = {qid: option_keys(qdef) for qid, qdef in questions.items()}
    return internal, keys
