"""LayaOnnxBackend: Laya's weights under ONNX Runtime, no PyTorch.

Pipeline per call (docs/laya-notes.md, "The answer"):
validate questions -> one token row per (state, question) -> pad -> ONNX Runtime -> softmax of
logits / temperature -> per-option probabilities.

Two weight formats, detected from the files present:
- single graph: `model.onnx` or `model-fp32.onnx` (input_ids, attention_mask, marker_pos, marker_mask, qtype) -> logits
- split: `encoder.onnx` (input_ids, attention_mask) -> last_hidden_state, then `head.onnx`
  (hidden_states, marker_pos, marker_mask, qtype, attention_mask) -> logits. Split weights also
  give `embed()`.

Checkpoint choice (decision Q24, provisional): `model="auto"` routes each state by its built-in
language: English to the English checkpoint, everything else to the multilingual one. Each
checkpoint loads lazily on first use. `model="english"`, `"multilingual"` or a local directory
pins one checkpoint.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .._laya.collate import collate_rows
from .._laya.sequence import (
    QTYPES, build_sequence, clamp_temperature, collapsed_options, encode_state, render_options,
    temp_bucket,
)
from .._laya.tokenizer import LayaTokenizer
from ..language import builtin_language, route_checkpoint
from . import weights as _weights
from .base import BaseBackend, Prediction, State, prepare_questions


class _Checkpoint:
    """One loaded checkpoint: config, tokenizer and ONNX sessions."""

    def __init__(self, name: str, model_dir: Path, providers: Optional[Sequence[str]] = None,
                 intra_op_threads: Optional[int] = None):
        import onnxruntime as ort

        self.name = name
        self.dir = Path(model_dir)
        cfg_path = self.dir / "rl_agent_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"{self.dir} has no rl_agent_config.json; not a Laya checkpoint")
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.max_len = int(self.cfg.get("max_len", 512))
        self.head_max_len = int(self.cfg.get("head_max_len", 192))
        self.temperature = [clamp_temperature(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])]
        if len(self.temperature) != 3:
            raise ValueError(f"{cfg_path}: temperature must be a list of 3 floats")
        self.temperature_by_options = {k: clamp_temperature(v)
                                       for k, v in self.cfg.get("temperature_by_options", {}).items()}
        self.tok = LayaTokenizer(self.dir)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            so.intra_op_num_threads = int(intra_op_threads)
        prov = list(providers) if providers else ["CPUExecutionProvider"]

        def session(fname: str):
            return ort.InferenceSession(str(self.dir / fname), sess_options=so, providers=prov)

        # A folder may hold both a quantized model.onnx and a full-precision model-fp32.onnx
        # (soyelmismo's layout); the full-precision one is the one Zet is verified against.
        single = next((f for f in ("model-fp32.onnx", "model.onnx") if (self.dir / f).is_file()), None)
        if single is not None:
            self.format = "single"
            self.onnx_file = single
            self.graph = session(single)
            self.encoder = self.head = None
            self._qtype_rank = self._rank(self.graph, "qtype")
        elif (self.dir / "encoder.onnx").is_file() and (self.dir / "head.onnx").is_file():
            self.format = "split"
            self.onnx_file = "encoder.onnx + head.onnx"
            self.graph = None
            self.encoder = session("encoder.onnx")
            self.head = session("head.onnx")
            head_inputs = {i.name for i in self.head.get_inputs()}
            if "attention_mask" not in head_inputs:
                # Without it, padding leaks into batch mates through the head transformer
                # (laya-ts/scripts/export_onnx.py:113-119). Refuse rather than return wrong answers.
                raise ValueError(f"{self.dir / 'head.onnx'} has no attention_mask input; re-export it "
                                 "with the current laya-ts/scripts/export_onnx.py")
            self._qtype_rank = self._rank(self.head, "qtype")
        else:
            raise FileNotFoundError(f"{self.dir}: expected model.onnx (or model-fp32.onnx), "
                                    "or encoder.onnx + head.onnx")
        self.revision = _weights.resolved_revision(self.dir)

    @staticmethod
    def _rank(sess, name: str) -> int:
        for i in sess.get_inputs():
            if i.name == name:
                return len(i.shape) if i.shape is not None else 1
        raise ValueError(f"ONNX graph has no {name!r} input")

    def rows_for_state(self, state: State, qids: List[str], internal: Dict[str, Dict]) -> List[Dict]:
        """One token row per question; the state is tokenized once (onnx_agent.py:527-561)."""
        truncate_left = isinstance(state, list)
        state_ids = encode_state(self.tok, state)
        rows = []
        for qid in qids:
            q = internal[qid]
            seq, markers, stats = build_sequence(
                self.tok, state, q, self.max_len, self.head_max_len,
                truncate_left=truncate_left, state_ids=state_ids, return_stats=True)
            n_opts = len(render_options(q))
            if len(markers) != n_opts:
                raise ValueError(
                    "question %r: its %d options and question text need %d tokens, more than "
                    "max_len=%d allows once head_max_len=%d is spent on them; use shorter options "
                    "or fewer of them" % (qid, n_opts, len(seq), self.max_len, self.head_max_len))
            rows.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "options": stats})
        return rows

    def run(self, rows: List[Dict]) -> tuple[np.ndarray, np.ndarray]:
        b = collate_rows(rows, self.tok.pad_token_id)
        qtype = b["qtype"].reshape(-1, 1) if self._qtype_rank == 2 else b["qtype"]
        if self.format == "single":
            logits, act = self.graph.run(["logits", "act_logits"], {
                "input_ids": b["input_ids"], "attention_mask": b["attention_mask"],
                "marker_pos": b["marker_pos"], "marker_mask": b["marker_mask"], "qtype": qtype})
        else:
            (hidden,) = self.encoder.run(None, {"input_ids": b["input_ids"],
                                                "attention_mask": b["attention_mask"]})
            logits, act = self.head.run(["logits", "act_logits"], {
                "hidden_states": hidden.astype(np.float32), "marker_pos": b["marker_pos"],
                "marker_mask": b["marker_mask"], "qtype": qtype,
                "attention_mask": b["attention_mask"]})
        return np.asarray(logits, dtype=np.float64), np.asarray(act, dtype=np.float64)

    def probs(self, logits_row: np.ndarray, qtype: int, k: int) -> np.ndarray:
        """softmax(logits / T) with Laya's bucketed temperature (onnx_agent.py:567-578)."""
        t = self.temperature_by_options.get(temp_bucket(qtype, k), self.temperature[qtype])
        z = logits_row[:k] / t
        p = np.exp(z - z.max())
        return p / p.sum()

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        if self.encoder is None:
            return None
        rows = []
        for text in texts:
            ids = encode_state(self.tok, text)[: self.max_len - 2]
            rows.append({"ids": [self.tok.cls_token_id] + ids + [self.tok.sep_token_id],
                         "markers": [0], "qtype": 0})
        b = collate_rows(rows, self.tok.pad_token_id)
        (hidden,) = self.encoder.run(None, {"input_ids": b["input_ids"],
                                            "attention_mask": b["attention_mask"]})
        mask = b["attention_mask"][:, :, None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        return pooled.astype(np.float32)


class LayaOnnxBackend(BaseBackend):
    name = "laya-onnx"

    def __init__(self, model: str = "auto", *, english: Optional[str] = None, revision: Optional[str] = None,
                 cache_dir: Optional[str] = None, providers: Optional[Sequence[str]] = None,
                 batch_rows: int = 16, intra_op_threads: Optional[int] = None):
        """
        model: "auto" (route by language), "english", "multilingual", or a local directory.
        english: with model="auto", a folder holding an English export. Without one (and while no
            English weights are hosted), "auto" answers English with the multilingual checkpoint too.
        revision: Hub revision to download (ignored for a local directory or "auto").
        batch_rows: token rows per ONNX run; each (state, question) pair is one row.
        """
        self.model = model
        self.english = english
        self.revision = revision
        self.cache_dir = cache_dir
        self.providers = providers
        self.batch_rows = max(1, int(batch_rows))
        self.intra_op_threads = intra_op_threads
        self._loaded: Dict[str, _Checkpoint] = {}
        self._lock = threading.Lock()

    # -- checkpoints -------------------------------------------------------------------------
    def checkpoint_for(self, state: State) -> str:
        if self.model != "auto":
            return self.model
        ck = route_checkpoint(builtin_language(state))
        if ck == "english" and self.english is None and "english" not in _weights.SOURCES:
            return "multilingual"  # no English weights available: the multilingual model reads English too
        return ck

    def _checkpoint(self, name: str) -> _Checkpoint:
        with self._lock:
            ck = self._loaded.get(name)
            if ck is None:
                rev = self.revision if self.model != "auto" else None
                target = self.english if (name == "english" and self.english) else name
                model_dir = _weights.resolve(target, revision=rev, cache_dir=self.cache_dir)
                ck = _Checkpoint(name, model_dir, self.providers, self.intra_op_threads)
                self._loaded[name] = ck
            return ck

    # -- Backend -----------------------------------------------------------------------------
    def predict(self, states: List[State], questions: Dict[str, Dict]) -> List[Prediction]:
        internal, keys = prepare_questions(questions)
        qids = list(questions)
        by_ck: Dict[str, List[int]] = {}
        for i, st in enumerate(states):
            by_ck.setdefault(self.checkpoint_for(st), []).append(i)

        out: List[Optional[Prediction]] = [None] * len(states)
        for ck_name, idxs in by_ck.items():
            ck = self._checkpoint(ck_name)
            # (state index, question index, row); rows sorted by length so a batch pads little.
            flat = []
            per_state_rows = {}
            for i in idxs:
                rows = ck.rows_for_state(states[i], qids, internal)
                per_state_rows[i] = rows
                flat.extend((i, j, r) for j, r in enumerate(rows))
            flat.sort(key=lambda x: len(x[2]["ids"]))
            logits_of: Dict[tuple, np.ndarray] = {}
            act_of: Dict[tuple, float] = {}
            for start in range(0, len(flat), self.batch_rows):
                chunk = flat[start:start + self.batch_rows]
                logits, act = ck.run([r for _, _, r in chunk])
                act_p = np.exp(act - act.max(axis=-1, keepdims=True))
                act_p = act_p / act_p.sum(axis=-1, keepdims=True)
                for n, (i, j, _) in enumerate(chunk):
                    logits_of[(i, j)] = logits[n]
                    act_of[(i, j)] = float(act_p[n, 0])
            for i in idxs:
                probs, act_meta = {}, {}
                for j, qid in enumerate(qids):
                    k = len(per_state_rows[i][j]["markers"])
                    p = ck.probs(logits_of[(i, j)], QTYPES[internal[qid]["t"]], k)
                    probs[qid] = {key: float(v) for key, v in zip(keys[qid], p)}
                    act_meta[qid] = act_of[(i, j)]
                meta = {"checkpoint": ck.name, "format": ck.format, "revision": ck.revision,
                        "language": builtin_language(states[i]), "act_probability": act_meta}
                collapsed = collapsed_options(qids, per_state_rows[i])
                if collapsed:
                    meta["collapsed_options"] = collapsed
                out[i] = Prediction(probs=probs, meta=meta)
        return out  # type: ignore[return-value]

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Mean-pooled encoder output of the state alone; None with single-graph weights."""
        if not texts:
            return None
        groups: Dict[str, List[int]] = {}
        for i, t in enumerate(texts):
            groups.setdefault(self.checkpoint_for(t), []).append(i)
        vectors: Dict[int, np.ndarray] = {}
        for ck_name, idxs in groups.items():
            emb = self._checkpoint(ck_name).embed([texts[i] for i in idxs])
            if emb is None:
                return None
            vectors.update(zip(idxs, emb))
        dims = {v.shape[0] for v in vectors.values()}
        if len(dims) > 1:
            # English and multilingual encoders have different widths; mixing them is meaningless.
            raise ValueError("embed() got texts routed to checkpoints with different widths; "
                             "pin one checkpoint with model='english' or 'multilingual'")
        return np.stack([vectors[i] for i in range(len(texts))])
