"""Built-in language detection, checkpoint routing, and FakeBackend's configurability."""
from __future__ import annotations

import pytest

from conftest import QUESTIONS
from zet.backends import FakeBackend, LayaOnnxBackend, get_backend
from zet.language import builtin_language, route_checkpoint


@pytest.mark.parametrize("text,want", [
    ("Could you resend last month's invoice? No rush.", "en"),
    ("Kan ni skicka om förra månadens faktura? Inte bråttom.", "sv"),
    # Laya's own detector calls this English (is_english=True, undecided); it has no å/ä/ö.
    ("Jag vill inte ha pengarna tillbaka, bara en fungerande app.", "sv"),
    ("Hej, appen kraschar när jag loggar in", "sv"),
    ("Hvad koster det, og hvordan betaler jeg?", "da"),
    ("Eg vil ikkje ha pengane tilbake, berre ein app.", "no"),
    # Nothing separates Danish from Norwegian here; before the family rule this was English.
    ("Jeg vil ikke have pengene tilbage, bare en app der virker.", "da-no"),
    ("Ich möchte mein Geld nicht zurück, nur eine funktionierende App.", "de"),
    ("Je ne veux pas de remboursement, juste une application qui marche.", "fr"),
    ("Мне нужен возврат денег за заказ", "und-cyrillic"),
    ("Tack!", "unknown"),
])
def test_builtin_language(text, want):
    assert builtin_language(text) == want


def test_dict_states_are_read_by_field():
    assert builtin_language({"body": "Kan ni skicka om förra månadens faktura? Inte bråttom."}) == "sv"


@pytest.mark.parametrize("lang,ck", [("en", "english"), ("en-US", "english"), ("sv", "multilingual"),
                                     ("da-no", "multilingual"), ("unknown", "multilingual"),
                                     ("und-cyrillic", "multilingual")])
def test_route_checkpoint(lang, ck):
    assert route_checkpoint(lang) == ck


def test_onnx_auto_routes_by_language(tmp_path):
    en = "Could you resend last month's invoice? No rush."
    # No hosted English weights: auto answers English with the multilingual checkpoint.
    assert LayaOnnxBackend(model="auto").checkpoint_for(en) == "multilingual"
    b = LayaOnnxBackend(model="auto", english=str(tmp_path))
    assert b.checkpoint_for(en) == "english"
    assert b.checkpoint_for("Tack!") == "multilingual"
    assert LayaOnnxBackend(model="english").checkpoint_for("Tack!") == "english"


def test_fake_fixed_and_fn():
    b = FakeBackend(fixed={"refund_requested": {"true": 3, "false": 1}})
    p = b.predict_proba(["x"], {"refund_requested": QUESTIONS["refund_requested"]})[0]
    assert p["refund_requested"] == {"false": 0.25, "true": 0.75}  # normalised, model order

    def fn(state, qid, qdef, keys):
        return {k: (1.0 if k == keys[0] else 0.0) for k in keys}

    p = FakeBackend(fn).predict_proba(["x"], QUESTIONS)[0]
    assert p["department"]["billing"] == 1.0 and p["urgency"]["0"] == 1.0


def test_fake_rejects_bad_distributions():
    with pytest.raises(ValueError, match="missing"):
        FakeBackend(fixed={"refund_requested": {"true": 1.0}}).predict_proba(
            ["x"], {"refund_requested": QUESTIONS["refund_requested"]})
    with pytest.raises(ValueError, match="non-negative"):
        FakeBackend(fixed={"refund_requested": {"true": -1.0, "false": 2.0}}).predict_proba(
            ["x"], {"refund_requested": QUESTIONS["refund_requested"]})


def test_fake_language_field_wins_and_seed_matters():
    b = FakeBackend(seed=3)
    assert b.detect_language({"body": "Could you resend the invoice?", "language": "sv"}) == "sv"
    a = FakeBackend(seed=1).predict_proba(["same"], QUESTIONS)[0]
    c = FakeBackend(seed=2).predict_proba(["same"], QUESTIONS)[0]
    assert a != c


def test_registry():
    assert get_backend("fake").name == "fake"
    assert get_backend("laya-onnx", model="english").name == "laya-onnx"
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("jev")
