"""Backends turn (states, questions) into per-option probabilities."""
from __future__ import annotations

from .base import Backend, BaseBackend, Prediction, Probs, State
from .fake import FakeBackend
from .laya_http import LayaHttpBackend
from .laya_onnx import LayaOnnxBackend

_REGISTRY = {"laya-onnx": LayaOnnxBackend, "laya-http": LayaHttpBackend, "fake": FakeBackend}


def get_backend(name: str, **kwargs) -> BaseBackend:
    """Build a backend by its registered name ("laya-onnx", "laya-http", "fake")."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; use one of {sorted(_REGISTRY)}") from None
    return cls(**kwargs)


__all__ = ["Backend", "BaseBackend", "FakeBackend", "LayaHttpBackend", "LayaOnnxBackend",
           "Prediction", "Probs", "State", "get_backend"]
