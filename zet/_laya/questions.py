# Copyright the Laya authors (Convai Innovations). Licensed under the Apache License, Version 2.0.
# Originally part of Laya, https://github.com/NandhaKishorM/laya
# (laya/agent.py `Agent._check_question` and `Agent._to_internal` @ 9d95567).
# Modified from Laya (Apache-2.0), https://github.com/NandhaKishorM/laya, by the Zet authors.
#
# Changes from the original:
# - Module-level functions instead of static methods on the torch Agent.
# - Choice labels must also be unique after `str()`, because Zet keys every option by string.
# - `option_keys` added: the option keys Zet reports for each question type.
"""Question validation, normalisation and option keys."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .sequence import QTYPES, resolve_noul_labels


def check_question(qid: str, qdef: Any) -> None:
    """Reject a question that cannot be answered, naming it and what to fix."""
    if not isinstance(qdef, dict):
        raise ValueError("question %r: definition must be a dict, got %s" % (qid, type(qdef).__name__))
    t = qdef.get("type")
    if t not in QTYPES:
        raise ValueError("question %r: unknown type %r; use one of %s" % (qid, t, sorted(QTYPES)))
    if "instructions" not in qdef:
        raise ValueError("question %r: no 'instructions'; add the text the model should answer" % (qid,))
    crit = qdef.get("criteria")
    if t == "choice":
        if not isinstance(crit, (dict, list)):
            raise ValueError("question %r: a choice question takes 'criteria' as a dict of "
                             "label -> description, or a list of labels" % (qid,))
        if not crit:
            raise ValueError("question %r: a choice question needs at least one criterion" % (qid,))
        labels = list(crit if isinstance(crit, list) else crit.keys())
        for i, label in enumerate(labels):
            if isinstance(label, (list, dict, set, bytearray)):
                raise ValueError(
                    "question %r: choice label %d is a %s; a label is rendered as option text "
                    "and used as the answer key, so it must be a scalar (a string, number or "
                    "bool), got %r" % (qid, i, type(label).__name__, label))
            if label is None:
                raise ValueError("question %r: choice label %d is null; a label must be a string, "
                                 "number or bool" % (qid, i))
        if isinstance(crit, list):
            keys: Dict[Any, int] = {}
            for i, label in enumerate(crit):
                try:
                    first = keys[label]
                except TypeError as exc:
                    raise ValueError(
                        "question %r: choice label %d (%r) cannot be an answer key because it is "
                        "unhashable" % (qid, i, label)) from exc
                except KeyError:
                    keys[label] = i
                else:
                    raise ValueError(
                        "question %r: choice label %d (%r) repeats label %d; the labels are the "
                        "answer keys, so every option needs its own (1, 1.0 and True are one "
                        "key)" % (qid, i, label, first))
        as_str = [str(label) for label in labels]
        if len(set(as_str)) != len(as_str):
            raise ValueError("question %r: choice labels %r are not unique as strings; Zet keys "
                             "every option by its string form" % (qid, labels))
    elif t == "score":
        if not isinstance(crit, list):
            raise ValueError("question %r: a score question takes 'criteria' as a list of level "
                             "descriptions, index 0 first" % (qid,))
        if not crit:
            raise ValueError("question %r: a score question needs at least one level" % (qid,))
        if None in crit:
            raise ValueError("question %r: score level %d is null; give every level a description, "
                             "index 0 first" % (qid, crit.index(None)))
    elif crit is not None and not isinstance(crit, dict):
        raise ValueError("question %r: a noul question takes 'criteria' as a dict with optional "
                         "'true'/'false' descriptions, or omits it" % (qid,))
    elif isinstance(crit, dict):
        keys_lower = {str(k).lower() for k in crit}
        if not keys_lower <= {"true", "false"}:
            raise ValueError(
                "question %r: a noul question takes 'criteria' keyed only 'true'/'false' (either "
                "or both, and omitted is fine), got %s. To word the answer differently, keep "
                "'criteria' keyed 'true'/'false' and set 'labels' instead." % (qid, sorted(keys_lower)))
    if "labels" in qdef:
        if t != "noul":
            raise ValueError("question %r: 'labels' is only supported for noul questions" % (qid,))
        try:
            resolve_noul_labels(qdef["labels"])
        except ValueError as e:
            raise ValueError("question %r: %s" % (qid, e)) from e


def to_internal(qdef: Dict) -> Dict:
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    q = {"t": t, "ins": ins, "crit": crit}
    if "labels" in qdef:
        q["labels"] = qdef["labels"]
    return q


def option_keys(qdef: Dict) -> List[str]:
    """Zet's option keys, in the model's option order.

    choice: the labels as strings. score: level indices "0".."k-1". noul: ["false", "true"].
    """
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice":
        return [str(k) for k in (crit if isinstance(crit, list) else crit.keys())]
    if t == "score":
        return [str(i) for i in range(len(crit))]
    return ["false", "true"]


def validate_questions(questions: Dict[str, Dict]) -> Dict[str, Dict]:
    """Check every question and return their internal forms, keyed by question id."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty dict of question id -> definition")
    for qid, qdef in questions.items():
        check_question(qid, qdef)
    return {qid: to_internal(qdef) for qid, qdef in questions.items()}
