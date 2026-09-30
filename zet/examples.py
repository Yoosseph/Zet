"""Labeled examples: loading from CSV/JSONL and normalising labels to option keys.

CSV format (UTF-8, Excel's BOM is fine):
- `body`: the text; required.
- `state.<field>`: extra state fields, e.g. `state.subject`.
- one column per question key, holding the true label (empty = not labeled for that question);
- `language` (optional): the example's language; otherwise it is detected.

Labels are normalised to Zet's option keys:
- choice: the label, matched exactly, then case-insensitively;
- score: the level index ("0", "1", ...) or the level's text;
- noul: true/false, yes/no, 1/0, ja/nej.
"""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from ._laya.questions import option_keys

_TRUE = {"true", "yes", "y", "1", "ja", "j", "t"}
_FALSE = {"false", "no", "n", "0", "nej", "f"}


def normalise_label(qid: str, qdef: Dict, value: Any) -> Optional[str]:
    """The option key for `value`, None for an empty cell. Raises ValueError naming the question."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    keys = option_keys(qdef)
    t = qdef["type"]
    if t == "noul":
        low = text.lower()
        if low in _TRUE:
            return "true"
        if low in _FALSE:
            return "false"
        raise ValueError(f"question {qid!r}: {text!r} is not a yes/no label (use true/false)")
    if t == "score":
        if text in keys:
            return text
        levels = [str(c).strip().lower() for c in qdef["criteria"]]
        if text.lower() in levels:
            return str(levels.index(text.lower()))
        raise ValueError(f"question {qid!r}: {text!r} is neither a level index 0..{len(keys) - 1} "
                         f"nor one of the level texts {qdef['criteria']}")
    if text in keys:
        return text
    by_lower = {k.lower(): k for k in keys}
    if text.lower() in by_lower:
        return by_lower[text.lower()]
    raise ValueError(f"question {qid!r}: {text!r} is not one of the options {keys}")


def _example(state: Dict[str, Any], labels: Dict[str, Any], questions: Dict[str, Dict],
             language: Optional[str]) -> Dict[str, Any]:
    norm = {}
    for qid, v in labels.items():
        if qid not in questions:
            raise ValueError(f"label for unknown question {qid!r}; questions are {sorted(questions)}")
        key = normalise_label(qid, questions[qid], v)
        if key is not None:
            norm[qid] = key
    ex = {"state": state, "labels": norm}
    if language:
        ex["language"] = str(language).strip()
    return ex


def read_csv(path: Union[str, Path], questions: Dict[str, Dict]) -> List[Dict[str, Any]]:
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "body" not in cols:
            raise ValueError(f"{path}: no 'body' column (columns: {cols})")
        qcols = [c for c in cols if c in questions]
        if not qcols:
            raise ValueError(f"{path}: no column matches a question key {sorted(questions)}; "
                             "label columns must be named exactly like the questions")
        unknown = [c for c in cols if c not in questions and c not in ("body", "language")
                   and not c.startswith("state.")]
        if unknown:
            warnings.warn(f"{path}: ignoring columns {unknown} (not body, language, state.* or a "
                          "question key)", stacklevel=3)
        out = []
        for line, row in enumerate(reader, start=2):
            body = (row.get("body") or "").strip()
            if not body:
                continue
            state = {"body": body}
            for c in cols:
                if c.startswith("state.") and row.get(c) not in (None, ""):
                    state[c[len("state."):]] = row[c]
            try:
                out.append(_example(state, {q: row.get(q) for q in qcols}, questions, row.get("language")))
            except ValueError as e:
                raise ValueError(f"{path}, line {line}: {e}") from None
        return out


def read_jsonl(path: Union[str, Path], questions: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """One JSON object per line: {"state": {...} or "text", "labels": {...}, "language": "sv"?}."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            obj = json.loads(raw)
            try:
                out.append(from_dict(obj, questions))
            except ValueError as e:
                raise ValueError(f"{path}, line {line}: {e}") from None
    return out


def from_dict(obj: Dict[str, Any], questions: Dict[str, Dict]) -> Dict[str, Any]:
    state = obj.get("state", obj.get("body"))
    if state is None:
        raise ValueError("example has no 'state' (or 'body')")
    if isinstance(state, str):
        state = {"body": state}
    return _example(state, obj.get("labels", {}), questions, obj.get("language"))


def load(source: Union[str, Path, Iterable[Dict[str, Any]]], questions: Dict[str, Dict]) -> List[Dict[str, Any]]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix.lower() == ".csv":
            return read_csv(p, questions)
        if p.suffix.lower() in (".jsonl", ".json"):
            return read_jsonl(p, questions)
        raise ValueError(f"{p}: use a .csv or .jsonl file")
    return [from_dict(o, questions) for o in source]
