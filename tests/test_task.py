"""Step 3: create, reload, examples, predict, correct/confirm, and the question-hash guard."""
from __future__ import annotations

import json

import pytest
import yaml

from conftest import QUESTIONS
from zet import FakeBackend, StaleCalibrationError, Task, ZetError
from zet.task import question_hash


@pytest.fixture
def task(tmp_path):
    return Task.create("support", QUESTIONS, backend=FakeBackend(seed=1), root=tmp_path)


def test_create_writes_the_layout(task, tmp_path):
    d = tmp_path / "support"
    for f in ("task.yaml", "examples.jsonl", "predictions.jsonl", "corrections.jsonl"):
        assert (d / f).is_file()
    cfg = yaml.safe_load((d / "task.yaml").read_text(encoding="utf-8"))
    assert cfg["question_hash"] == question_hash(QUESTIONS)
    assert cfg["backend"]["name"] == "fake" and cfg["current_version"] is None


def test_create_refuses_to_clobber_and_bad_names(task, tmp_path):
    with pytest.raises(ZetError, match="already exists"):
        Task.create("support", QUESTIONS, backend=FakeBackend(), root=tmp_path)
    with pytest.raises(ZetError, match="not a usable task name"):
        Task.create("a/b", QUESTIONS, backend=FakeBackend(), root=tmp_path)
    with pytest.raises(ValueError, match="unknown type"):
        Task.create("bad", {"q": {"type": "rank", "instructions": "x"}}, root=tmp_path)


def test_reload_from_disk(task, tmp_path):
    again = Task.load("support", root=tmp_path, backend=FakeBackend(seed=1))
    assert again.questions == QUESTIONS and again.name == "support"
    with pytest.raises(ZetError, match="no task"):
        Task.load("nope", root=tmp_path)


def test_add_examples_from_csv_normalises_labels(task, tmp_path):
    csv = tmp_path / "labeled.csv"
    csv.write_text(
        "body,state.subject,department,urgency,refund_requested,language\n"
        "Resend my invoice please,Invoice,Billing,not urgent,no,en\n"
        "Appen kraschar,Krasch,technical,2,ja,sv\n"
        "Price for 20 seats?,,sales,,,\n", encoding="utf-8-sig")
    assert task.add_examples(csv) == 3
    ex = task.examples()
    assert ex[0]["state"] == {"body": "Resend my invoice please", "subject": "Invoice"}
    assert ex[0]["labels"] == {"department": "billing", "urgency": "0", "refund_requested": "false"}
    assert ex[1]["labels"] == {"department": "technical", "urgency": "2", "refund_requested": "true"}
    assert ex[1]["language"] == "sv"
    assert ex[2]["labels"] == {"department": "sales"} and "language" not in ex[2]
    assert [e["id"] for e in ex] == ["e000001", "e000002", "e000003"]


def test_bad_labels_name_the_line_and_question(task, tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("body,department\nhello,finance\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2.*department.*finance"):
        task.add_examples(csv)
    no_q = tmp_path / "noq.csv"
    no_q.write_text("body,dept\nhello,billing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no column matches a question key"):
        task.add_examples(no_q)


def test_predict_uncalibrated_is_all_unsure_and_logged(task):
    r = task.predict({"body": "Could you resend last month's invoice? No rush."})
    assert r.calibrated is False and "not calibrated" in r.warnings[0]
    assert set(r) == set(QUESTIONS)
    a = r["department"]
    assert a.status == "unsure" and a.answer == max(a.probs, key=a.probs.get)
    assert abs(sum(a.probs.values()) - 1) < 1e-9 and r.language == "en"
    rec = task.prediction(r.id)
    assert rec["answers"]["department"]["answer"] == a.answer
    many = task.predict(["one", {"body": "two", "language": "sv"}])
    assert [m.language for m in many][1] == "sv"
    assert len(task.predictions()) == 3


def test_language_field_is_not_sent_to_the_model(tmp_path):
    seen = []

    def fn(state, qid, qdef, keys):
        seen.append(state)
        return {k: 1.0 for k in keys}

    t = Task.create("t", QUESTIONS, backend=FakeBackend(fn), root=tmp_path)
    t.predict({"body": "hej", "language": "sv"})
    assert all("language" not in s for s in seen)


def test_correct_and_confirm(task):
    r = task.predict("The app crashes on login")
    task.correct(r.id, department="Technical", urgency="critical")
    task.confirm(r.id, "refund_requested")
    c = task.corrections()
    assert c[0]["kind"] == "correction"
    assert c[0]["labels"] == {"department": "technical", "urgency": "2"}
    assert c[1]["kind"] == "confirm" and c[1]["labels"] == {"refund_requested": r["refund_requested"].answer}
    with pytest.raises(ZetError, match="no prediction"):
        task.correct("p000", department="sales")
    with pytest.raises(ZetError, match="unknown question"):
        task.correct(r.id, dept="sales")


def test_question_change_invalidates_calibration(task):
    vdir = task.path / "versions" / "1"
    vdir.mkdir(parents=True)
    (vdir / "calibration.json").write_text(json.dumps({"question_hash": question_hash(QUESTIONS),
                                                       "backend": task.config["backend"]}))
    task.config["current_version"] = 1
    task._save_config()
    assert task.calibration() is not None
    cfg = yaml.safe_load((task.path / "task.yaml").read_text(encoding="utf-8"))
    cfg["questions"]["urgency"]["criteria"] = ["calm", "urgent"]
    (task.path / "task.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    edited = Task.load("support", root=task.path.parent, backend=FakeBackend())
    with pytest.raises(StaleCalibrationError, match="recalibrate|calibrate"):
        edited.predict("hello")


def test_default_root_is_not_the_current_folder(tmp_path, monkeypatch):
    from zet import default_root, list_tasks
    monkeypatch.setenv("ZET_ROOT", str(tmp_path / "home"))
    assert default_root() == tmp_path / "home"
    Task.create("a", QUESTIONS, backend=FakeBackend())
    Task.create("b", QUESTIONS, backend=FakeBackend())
    assert list_tasks() == ["a", "b"] and (tmp_path / "home" / "a" / "task.yaml").is_file()
    monkeypatch.delenv("ZET_ROOT")
    assert default_root().name == "zet" and default_root().parent == type(tmp_path).home()


def test_changing_the_model_invalidates_calibration(tmp_path):
    from test_calibration_property import _task
    t = _task(tmp_path)
    t.calibrate(error_budget=0.1)
    t.set_backend("laya-onnx", {"model": "multilingual"})
    with pytest.raises(StaleCalibrationError, match="model"):
        t.calibration()


def test_review_queue(tmp_path):
    from test_calibration_property import _task
    t = _task(tmp_path)
    assert t.review_queue() == []                       # uncalibrated: nothing to review yet
    t.calibrate(error_budget=0.1)
    t.set_audit_rate(1.0)
    sure = t.predict({"body": "s", "sim": [0.999, 0.0005, 0.0003, 0.0002], "language": "en"})
    t.set_audit_rate(0.0)
    unsure = t.predict({"body": "u", "sim": [0.45, 0.45, 0.05, 0.05], "language": "en"})
    quiet = t.predict({"body": "q", "sim": [0.999, 0.0005, 0.0003, 0.0002], "language": "en"})
    q = t.review_queue()
    assert [r["id"] for r in q] == [sure.id, unsure.id]  # audits first; sure-and-unaudited skipped
    assert q[0]["is_audit"] and q[1]["pending"] == ["dept"]
    t.correct(unsure.id, dept="b")
    t.confirm(sure.id)
    assert t.review_queue() == [] and quiet.id not in [r["id"] for r in q]


def test_delete_is_recoverable(tmp_path):
    from zet import list_tasks
    t = Task.create("gone", QUESTIONS, backend=FakeBackend(), root=tmp_path)
    moved = t.delete()
    assert list_tasks(tmp_path) == [] and moved.name.startswith(".deleted-gone-")
    assert (moved / "task.yaml").is_file()            # rename it back to restore
    moved.rename(tmp_path / "gone")
    assert list_tasks(tmp_path) == ["gone"]


def test_remove_and_restore_examples_keeps_history_and_ids(task):
    task.add_examples([{"state": "a", "labels": {"department": "billing"}},
                       {"state": "b", "labels": {"department": "sales"}}])
    assert task.remove_examples(["e000001", "nope"]) == 1
    assert [e["id"] for e in task.examples()] == ["e000002"]
    task.add_examples([{"state": "c", "labels": {}}])
    assert [e["id"] for e in task.examples()] == ["e000002", "e000003"]   # no id reused
    assert task.restore_examples(["e000001"]) == 1
    assert sorted(e["id"] for e in task.examples()) == ["e000001", "e000002", "e000003"]
    raw = (task.path / "examples.jsonl").read_text(encoding="utf-8")
    assert '"remove": "e000001"' in raw and '"restore": "e000001"' in raw   # append-only history


def test_removed_examples_are_left_out_of_calibration(tmp_path):
    from test_calibration_property import _task
    t = _task(tmp_path, n=300)
    t.remove_examples([e["id"] for e in t.examples()[:100]])
    cal = t.calibrate(error_budget=0.1)
    assert cal["split"]["n_examples"] == 200
