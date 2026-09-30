"""LayaOnnxBackend internals, with hand-computed expectations on the tiny model.

The tiny model's logit for an option is the fixed score of the option's first token
(scripts/make_tiny_model.py): billing 2.0, technical 1.0, sales 0.5, other -0.5,
false -0.3, true 0.3, level 0.0. Its config sets temperature [1.0, 1.5, 1.0] and
temperature_by_options {"choice:3-5": 2.0}.
"""
from __future__ import annotations

import math
import shutil

import pytest

from conftest import QUESTIONS, TINY
from zet._laya.sequence import build_sequence, clamp_temperature, render_options, temp_bucket
from zet._laya.tokenizer import LayaTokenizer
from zet.backends import LayaOnnxBackend
from zet.backends import weights


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [v / sum(e) for v in e]


def test_render_options_matches_laya():
    assert render_options({"t": "choice", "crit": {"billing": None, "tech": "technical faults"}}) == \
        ["billing", "tech: technical faults"]
    assert render_options({"t": "score", "crit": ["calm", "urgent"]}) == ["level 0: calm", "level 1: urgent"]
    assert render_options({"t": "noul", "crit": None}) == [
        "false: no, the statement does not hold", "true: yes, the statement holds"]


def test_build_sequence_layout():
    tok = LayaTokenizer(TINY / "single")
    q = {"t": "noul", "ins": "is it urgent", "crit": None}
    ids, markers = build_sequence(tok, "please resend", q, max_len=96, head_max_len=48)
    words = tok._tok.decode(ids, skip_special_tokens=False).split()
    # [CLS] noul question : is it urgent [SEP] [MASK] false : ... [MASK] true : ... [SEP] please resend [SEP]
    assert words[:7] == ["[CLS]", "noul", "question", ":", "is", "it", "urgent"]
    assert [ids[m] for m in markers] == [tok.mask_token_id] * 2
    assert tok._tok.id_to_token(ids[markers[0] + 1]) == "false"
    assert tok._tok.id_to_token(ids[markers[1] + 1]) == "true"
    assert words[-3:] == ["please", "resend", "[SEP]"]


def test_state_truncation_keeps_markers_and_direction():
    tok = LayaTokenizer(TINY / "single")
    q = {"t": "noul", "ins": "x", "crit": None}
    long_state = " ".join(["invoice"] * 200)
    ids, markers = build_sequence(tok, long_state, q, max_len=40, head_max_len=48)
    assert len(ids) == 40 and len(markers) == 2
    turns = ["billing"] * 60 + ["sales"]
    ids_left, _ = build_sequence(tok, " ".join(turns), q, max_len=40, truncate_left=True)
    assert tok._tok.id_to_token(ids_left[-2]) == "sales"  # left truncation keeps the latest turn


def test_temperature_buckets_and_clamp():
    assert temp_bucket(0, 4) == "choice:3-5"
    assert temp_bucket(2, 2) == "noul:2"
    assert temp_bucket(0, 18) == "choice:11+"
    assert clamp_temperature(0.1006) == 0.5
    assert clamp_temperature(float("nan")) == 1.0
    assert clamp_temperature("x") == 1.0


@pytest.mark.parametrize("fmt", ["single", "split"])
def test_probabilities_hand_computed(fmt):
    b = LayaOnnxBackend(model=str(TINY / fmt))
    p = b.predict_proba([{"body": "please resend the invoice"}], QUESTIONS)[0]
    dept = softmax([2.0 / 2.0, 1.0 / 2.0, 0.5 / 2.0, -0.5 / 2.0])      # bucket choice:3-5 -> T=2.0
    assert list(p["department"].values()) == pytest.approx(dept, abs=1e-6)
    assert list(p["urgency"].values()) == pytest.approx([1 / 3] * 3, abs=1e-6)  # all "level ..."
    assert p["refund_requested"]["true"] == pytest.approx(softmax([-0.3, 0.3])[1], abs=1e-6)


def test_metadata():
    b = LayaOnnxBackend(model=str(TINY / "single"))
    pred = b.predict([{"body": "Jag vill inte ha pengarna tillbaka, bara en fungerande app."}],
                     QUESTIONS)[0]
    assert pred.meta["language"] == "sv"
    assert pred.meta["format"] == "single"
    assert set(pred.meta["act_probability"]) == set(QUESTIONS)
    assert all(0.0 <= v <= 1.0 for v in pred.meta["act_probability"].values())


def test_embed_mean_pools_the_state_alone():
    b = LayaOnnxBackend(model=str(TINY / "split"))
    emb = b.embed(["billing", "billing technical"])
    # [CLS] billing [SEP] -> mean(0, 2, 0); [CLS] billing technical [SEP] -> mean(0, 2, 1, 0)
    assert emb[:, 0] == pytest.approx([2 / 3, 3 / 4], abs=1e-6)
    assert LayaOnnxBackend(model=str(TINY / "single")).embed(["x"]) is None


def test_too_many_options_for_max_len_is_a_clear_error():
    b = LayaOnnxBackend(model=str(TINY / "single"))  # max_len 96
    q = {"big": {"type": "choice", "instructions": "x",
                 "criteria": {"opt%d" % i: "billing " * 30 for i in range(30)}}}  # 30 x 4 tokens > 96
    with pytest.raises(ValueError, match="max_len"):
        b.predict_proba(["hi"], q)


def test_split_head_without_attention_mask_is_refused():
    with pytest.raises(ValueError, match="attention_mask"):
        LayaOnnxBackend(model=str(TINY / "split_old_head")).predict_proba(["hi"], QUESTIONS)


def test_missing_files_are_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="rl_agent_config.json"):
        LayaOnnxBackend(model=str(tmp_path)).predict_proba(["hi"], QUESTIONS)


def test_sha256_pins_are_enforced(tmp_path):
    d = tmp_path / "m"
    shutil.copytree(TINY / "single", d)
    good = weights.sha256_file(d / "model.onnx")
    assert weights.resolve(str(d), sha256={"model.onnx": good}) == d
    with pytest.raises(ValueError, match="does not match"):
        weights.resolve(str(d), sha256={"model.onnx": "0" * 64})


def test_revision_is_read_from_hub_snapshot_path(tmp_path):
    p = tmp_path / "models--x--y" / "snapshots" / "abc123" / "english"
    p.mkdir(parents=True)
    assert weights.resolved_revision(p) == "abc123"
    assert weights.resolved_revision(tmp_path) is None


def test_unknown_model_name():
    with pytest.raises(ValueError, match="unknown model"):
        weights.resolve("klingon")


def test_full_precision_file_is_preferred(tmp_path):
    d = tmp_path / "both"
    shutil.copytree(TINY / "single", d)
    shutil.copy(d / "model.onnx", d / "model-fp32.onnx")
    b = LayaOnnxBackend(model=str(d))
    assert b._checkpoint(str(d)).onnx_file == "model-fp32.onnx"


def test_english_without_hosted_weights_explains_what_to_do():
    with pytest.raises(FileNotFoundError, match="export_onnx.py"):
        weights.resolve("english")


def test_download_failure_suggests_a_local_folder(monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise OSError("repository gone")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    with pytest.raises(RuntimeError, match="local export"):
        weights.resolve("multilingual")
