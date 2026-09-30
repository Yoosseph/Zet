# Copyright the Laya authors (Convai Innovations). Licensed under the Apache License, Version 2.0.
# Originally part of Laya, https://github.com/NandhaKishorM/laya (laya/common.py `collate_items` @ 9d95567).
# Modified from Laya (Apache-2.0), https://github.com/NandhaKishorM/laya, by the Zet authors.
#
# Changes from the original: numpy arrays instead of torch tensors; takes a flat list of rows;
# training-only fields (targets, labels, meta) dropped.
"""Pad token rows into the arrays the ONNX graphs take."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def collate_rows(rows: List[Dict], pad_id: int) -> Dict[str, np.ndarray]:
    """rows: dicts with `ids`, `markers`, `qtype`. Returns int64/bool arrays."""
    if not rows:
        raise ValueError("collate_rows: no rows")
    n, length = len(rows), max(len(r["ids"]) for r in rows)
    kmax = max(len(r["markers"]) for r in rows)
    ids = np.full((n, length), pad_id, dtype=np.int64)
    att = np.zeros((n, length), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    for i, r in enumerate(rows):
        ids[i, : len(r["ids"])] = r["ids"]
        att[i, : len(r["ids"])] = 1
        k = len(r["markers"])
        mpos[i, :k] = r["markers"]
        mmask[i, :k] = True
    qtype = np.array([r["qtype"] for r in rows], dtype=np.int64)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos,
            "marker_mask": mmask, "qtype": qtype}
