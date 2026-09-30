"""Built-in language detection (Q17, step 3 of the resolution order) and checkpoint routing (Q24).

The detector is Laya's own `lang.py`, extended with Swedish, Norwegian and Danish, so the groups Zet
calibrates on line up with how Laya decides what its English checkpoint can read.
"""
from __future__ import annotations

from typing import Any

from ._laya.lang import analyse

UNKNOWN = "unknown"
ENGLISH = "english"
MULTILINGUAL = "multilingual"


def builtin_language(state: Any) -> str:
    """A language code ("en", "sv", "da-no", ...), "und-<script>" for an unidentified non-Latin
    script, or "unknown" when nothing identifies the language (short or code-like text)."""
    det = analyse(state)
    lang = det.get("language")
    if lang:
        return str(lang)
    script = det.get("script")
    if script and script not in ("latin", "unknown"):
        return "und-%s" % script
    return UNKNOWN


def route_checkpoint(language: str) -> str:
    """English text goes to the English checkpoint; everything else, including "unknown", goes to
    the multilingual one. Laya's own notes rate English-checkpoint-on-foreign-text as the costlier
    mistake: it collapses while staying confident (laya/router.py:21-24)."""
    return ENGLISH if language.split("-")[0].lower() == "en" else MULTILINGUAL


def get_detector(method: str = "builtin", languages=None):
    """A function state -> language code, for a detection method recorded in a calibration.

    "builtin": Laya's detector with sv/no/da. "lingua": lingua-language-detector limited to
    `languages` (ISO 639-1 codes); install with `pip install zet[langid]` (Python 3.12+).
    """
    if method == "builtin":
        return builtin_language
    if method == "lingua":
        try:
            from lingua import IsoCode639_1, LanguageDetectorBuilder
        except ImportError as e:
            raise RuntimeError("this calibration used the 'lingua' language detector; install it with "
                               "`pip install zet[langid]` (Python 3.12+) or recalibrate with "
                               "language_detector='builtin'") from e
        if not languages:
            raise ValueError("the lingua detector needs the task's languages, e.g. ['en', 'sv']")
        codes = [getattr(IsoCode639_1, c.split("-")[0].upper()) for c in languages]
        detector = LanguageDetectorBuilder.from_iso_codes_639_1(*codes).build()

        def detect(state) -> str:
            from ._laya.lang import state_text

            lang = detector.detect_language_of(state_text(state) if not isinstance(state, str) else state)
            return lang.iso_code_639_1.name.lower() if lang is not None else UNKNOWN

        return detect
    raise ValueError(f"unknown language detector {method!r}; use 'builtin' or 'lingua'")
