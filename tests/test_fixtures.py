"""Step 7: the support-email fixtures, and the full flow on them.

The unit test runs everywhere on FakeBackend. The model test (--run-model) runs the real Laya
multilingual checkpoint over the fixtures and prints the report: the measured numbers of the
0.1 fixture run. With 125 examples (25 Swedish), expect a low automation rate and wide bounds:
that is the honest result on this little data, not a failure.
"""
from __future__ import annotations

import csv
import os

import pytest
import yaml

from conftest import FIXTURES
from zet import FakeBackend, LayaOnnxBackend, Task

SUPPORT = FIXTURES / "support"


def load_questions():
    return yaml.safe_load((SUPPORT / "questions.yaml").read_text(encoding="utf-8"))


def rows():
    with open(SUPPORT / "emails.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_fixture_set_meets_the_brief():
    r = rows()
    assert len(r) >= 100
    sv = sum(x["language"] == "sv" for x in r)
    assert 0.15 <= sv / len(r) <= 0.25
    bodies = " ".join(x["body"] for x in r)
    for phrase in ("don't want a refund", "No rush", "Inte bråttom", "vill inte ha pengarna tillbaka"):
        assert phrase in bodies, phrase
    assert {x["department"] for x in r} == {"billing", "technical", "sales", "other"}
    assert {x["refund_requested"] for x in r} == {"yes", "no"}


def test_full_flow_on_fixtures(tmp_path):
    t = Task.create("support", load_questions(), backend=FakeBackend(seed=3), root=tmp_path)
    assert t.add_examples(SUPPORT / "emails.csv") == len(rows())
    cal = t.calibrate(error_budget=0.1)
    for qid, q in cal["questions"].items():
        assert q["n_labeled"] == len(rows()) and q["n_test"] > 0
        assert not any("weak guarantee" in w for w in q["warnings"])  # 125 >= 100
    r = t.predict({"body": "Kan ni skicka om fakturan? Inte bråttom.", "language": "sv"})
    assert r.language == "sv" and set(r) == set(load_questions())
    rep = t.report(print_it=False)
    assert set(rep["questions"]) == set(load_questions())


@pytest.mark.model
def test_fixture_run_on_laya(tmp_path):
    model = os.environ.get("ZET_MODEL_DIR_MULTILINGUAL", "multilingual")
    t = Task.create("support", load_questions(), backend=LayaOnnxBackend(model=model), root=tmp_path)
    t.add_examples(SUPPORT / "emails.csv")
    t.calibrate(error_budget=0.05)
    rep = t.report()
    assert rep["calibrated"]
