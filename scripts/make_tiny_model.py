"""Build tiny Laya-shaped ONNX models for the unit tests (tests/fixtures/tiny_model/).

They have exactly the inputs and outputs of Laya's real graphs, so the ONNX backend's code path
runs in CI with no download. The math is a toy: each option's logit is a fixed score of the
option's first token. Both formats compute the same numbers, so tests can compare them.

Needs `onnx` (dev only). The generated files are committed; rerun only to change them:
    python scripts/make_tiny_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tiny_model"
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
# Option-leading words get distinct scores so answers are predictable; everything else scores 0.
SCORED = {"billing": 2.0, "technical": 1.0, "sales": 0.5, "other": -0.5,
          "false": -0.3, "true": 0.3, "level": 0.0, "yes": 0.4, "no": -0.4}
WORDS = sorted(set(SCORED) | {
    "choice", "score", "noul", "question", "which", "department", "how", "urgent", "is", "it",
    "does", "the", "customer", "ask", "for", "a", "refund", "invoice", "please", "resend", "rush",
    "not", "soon", "critical", "statement", "holds", "hold", "body", "kan", "ni", "skicka", "om",
    "faktura", "inte", "bråttom", "jag", "vill", "ha", "pengarna", "tillbaka", "app", "crashes",
    "login", "0", "1", "2", ":", ",", ".", "?", "{", "}", "\"", "'", "!"})
OPSET, IR = 17, 8


def build_tokenizer() -> dict:
    vocab = {t: i for i, t in enumerate(SPECIAL + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    for t in SPECIAL:
        tok.add_special_tokens([t])
    OUT.mkdir(parents=True, exist_ok=True)
    return vocab, tok


def embedding(vocab: dict) -> np.ndarray:
    emb = np.zeros((len(vocab), 1), dtype=np.float32)
    for w, s in SCORED.items():
        emb[vocab[w], 0] = s
    return emb


def encoder_nodes(emb: np.ndarray):
    init = [numpy_helper.from_array(emb, "embedding")]
    nodes = [helper.make_node("Gather", ["embedding", "input_ids"], ["last_hidden_state"], axis=0)]
    return init, nodes


def head_nodes(hidden: str):
    """logits[n, k] = hidden[n, marker_pos + 1, 0], -1e4 where marker_mask is False."""
    init = [numpy_helper.from_array(np.array([1], dtype=np.int64), "one"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2"),
            numpy_helper.from_array(np.array(-1e4, dtype=np.float32), "neg")]
    nodes = [
        helper.make_node("Squeeze", [hidden, "axis2"], ["h2"]),                     # [n, S]
        helper.make_node("Add", ["marker_pos", "one"], ["next_pos"]),                # [n, k]
        helper.make_node("GatherElements", ["h2", "next_pos"], ["raw"], axis=1),     # [n, k]
        helper.make_node("Where", ["marker_mask", "raw", "neg"], ["logits"]),
        helper.make_node("ReduceMax", ["logits"], ["top"], axes=[1], keepdims=1),    # [n, 1]
        helper.make_node("Neg", ["top"], ["ntop"]),
        helper.make_node("Concat", ["top", "ntop"], ["act_logits"], axis=1),         # [n, 2]
    ]
    return init, nodes


def io(name, elem, shape):
    return helper.make_tensor_value_info(name, elem, shape)


def save(graph, path: Path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def main():
    vocab, tok = build_tokenizer()
    emb = embedding(vocab)
    cfg = {"max_len": 96, "head_max_len": 48, "temperature": [1.0, 1.5, 1.0],
           "temperature_by_options": {"choice:3-5": 2.0}}
    tcfg = {"cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]", "pad_token": "[PAD]"}
    common_in = [io("input_ids", TensorProto.INT64, ["n", "s"]),
                 io("attention_mask", TensorProto.INT64, ["n", "s"])]
    head_in = [io("marker_pos", TensorProto.INT64, ["n", "k"]),
               io("marker_mask", TensorProto.BOOL, ["n", "k"])]
    outs = [io("logits", TensorProto.FLOAT, ["n", "k"]), io("act_logits", TensorProto.FLOAT, ["n", 2])]

    for fmt in ("single", "split"):
        d = OUT / fmt
        d.mkdir(parents=True, exist_ok=True)
        tok.save(str(d / "tokenizer.json"))
        (d / "tokenizer_config.json").write_text(json.dumps(tcfg, indent=1), encoding="utf-8")
        (d / "rl_agent_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    # Single graph: qtype is [n], as in laya/onnx_agent.py.
    ei, en = encoder_nodes(emb)
    hi, hn = head_nodes("last_hidden_state")
    save(helper.make_graph(en + hn, "laya_tiny_single",
                           common_in + head_in + [io("qtype", TensorProto.INT64, ["n"])], outs, ei + hi),
         OUT / "single" / "model.onnx")

    # Split: encoder, then head with hidden_states and qtype [n, 1], as in laya-ts/scripts/export_onnx.py.
    save(helper.make_graph(en, "laya_tiny_encoder", common_in,
                           [io("last_hidden_state", TensorProto.FLOAT, ["n", "s", 1])], ei),
         OUT / "split" / "encoder.onnx")
    hi2, hn2 = head_nodes("hidden_states")
    save(helper.make_graph(hn2, "laya_tiny_head",
                           [io("hidden_states", TensorProto.FLOAT, ["n", "s", 1])] + head_in
                           + [io("qtype", TensorProto.INT64, ["n", 1]),
                              io("attention_mask", TensorProto.INT64, ["n", "s"])], outs, hi2),
         OUT / "split" / "head.onnx")
    # An older split export whose head lacks attention_mask; the backend must refuse it.
    old = OUT / "split_old_head"
    old.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "rl_agent_config.json", "encoder.onnx"):
        (old / name).write_bytes((OUT / "split" / name).read_bytes())
    hi3, hn3 = head_nodes("hidden_states")
    save(helper.make_graph(hn3, "laya_tiny_head_old",
                           [io("hidden_states", TensorProto.FLOAT, ["n", "s", 1])] + head_in
                           + [io("qtype", TensorProto.INT64, ["n", 1])], outs, hi3),
         old / "head.onnx")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
