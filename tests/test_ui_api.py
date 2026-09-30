"""The interface's HTTP API, end to end over real HTTP, on the fake backend and the tiny ONNX model."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import FIXTURES, TINY
from zet import FakeBackend
from zet.ui.server import App, serve

SUPPORT = FIXTURES / "support"


@pytest.fixture
def server(tmp_path):
    app = App(tmp_path, backend_factory=lambda name: FakeBackend(seed=4))
    srv = serve(app=app, open_browser=False, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.url.rstrip("/")
    srv.shutdown()


def call(base, method, path, body=None, header=True):
    req = urllib.request.Request(base + path, method=method,
                                 data=json.dumps(body or {}).encode() if method == "POST" else None)
    if method == "POST":
        req.add_header("Content-Type", "application/json")
        if header:
            req.add_header("X-Zet", "1")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait(base, job):
    for _ in range(300):
        if job["status"] != "running":
            return job
        time.sleep(0.05)
        _, job = call(base, "GET", "/api/jobs/" + job["id"])
    raise TimeoutError(job)


def questions():
    import yaml
    return yaml.safe_load((SUPPORT / "questions.yaml").read_text(encoding="utf-8"))


def test_page_and_guards(server):
    st, page = call(server, "GET", "/")
    assert st == 200 and b"<title>Zet</title>" in page
    assert call(server, "POST", "/api/tasks", {"name": "x"}, header=False)[0] == 403   # no X-Zet header
    assert call(server, "GET", "/api/nope")[0] == 404
    assert call(server, "GET", "/api/tasks/missing")[0] == 404


def test_full_flow(server):
    st, d = call(server, "POST", "/api/tasks", {"name": "support", "questions": questions(), "backend": "laya-onnx"})
    assert st == 200 and d["name"] == "support" and d["version"] is None
    st, err = call(server, "POST", "/api/tasks", {"name": "support", "questions": questions()})
    assert st == 400 and "already exists" in err["error"]

    csv_text = (SUPPORT / "emails.csv").read_text(encoding="utf-8")
    st, r = call(server, "POST", "/api/tasks/support/examples", {"csv": csv_text, "filename": "emails.csv"})
    assert st == 200 and r["added"] == 125
    st, bad = call(server, "POST", "/api/tasks/support/examples", {"csv": "body,department\nhi,finance\n", "filename": "mine.csv"})
    assert st == 400 and "mine.csv" in bad["error"] and "finance" in bad["error"]
    st, one = call(server, "POST", "/api/tasks/support/examples",
                   {"rows": [{"state": {"body": "Hej"}, "labels": {"department": "other"}, "language": "sv"}]})
    assert one["total"] == 126
    st, ex = call(server, "GET", "/api/tasks/support/examples?limit=5")
    assert ex["total"] == 126 and len(ex["rows"]) == 5 and ex["rows"][0]["state"]["body"] == "Hej"
    assert ex["label_counts"]["department"]["billing"] > 0 and ex["languages"]["sv"] == 26

    st, tmpl = call(server, "GET", "/api/tasks/support/template.csv")
    assert st == 200 and tmpl.decode().startswith("body,department,urgency,churn_risk,refund_requested,language")

    job = wait(server, call(server, "POST", "/api/tasks/support/calibrate", {"error_budget": 0.1})[1])
    assert job["status"] == "done" and job["result"]["version"] == 1
    st, rep = call(server, "GET", "/api/tasks/support/report")
    assert rep["calibrated"] and set(rep["questions"]) == set(questions())

    call(server, "POST", "/api/tasks/support/audit-rate", {"rate": 1.0})
    job = wait(server, call(server, "POST", "/api/tasks/support/predict",
                            {"texts": ["Could you resend the invoice?", "Appen kraschar"], "language": ""})[1])
    assert job["status"] == "done" and len(job["result"]) == 2
    res = job["result"][0]
    assert set(res["answers"]) == set(questions()) and res["state"]["body"] == "Could you resend the invoice?"

    st, rv = call(server, "GET", "/api/tasks/support/review")
    assert rv["calibrated"] and rv["queue"]
    item = rv["queue"][0]
    labels = {q: item["answers"][q]["answer"] for q in item["pending"]}
    first = item["pending"][0]
    other = next(k for k in item["answers"][first]["probs"] if k != labels[first])
    labels[first] = other
    st, v = call(server, "POST", "/api/tasks/support/verdict", {"id": item["id"], "labels": labels})
    assert st == 200 and first in v["corrected"]
    st, rv2 = call(server, "GET", "/api/tasks/support/review")
    assert item["id"] not in [r["id"] for r in rv2["queue"]]
    st, hist = call(server, "GET", "/api/tasks/support/history")
    assert hist["total"] == 2 and any(r["verdicts"] for r in hist["rows"])

    st, d = call(server, "POST", "/api/tasks/support/audit-rate", {"rate": 0.1, "group": "sv"})
    assert d["audit_rate_by_group"] == {"sv": 0.1}
    st, d = call(server, "POST", "/api/tasks/support/audit-rate", {"rate": 0.1, "group": "sv", "remove": True})
    assert d["audit_rate_by_group"] == {}

    st, d = call(server, "POST", "/api/tasks/support/backend", {"name": "laya-onnx", "options": {"model": "multilingual"}})
    assert d["stale"] and "model" in d["stale"]                    # changing the model needs recalibration
    st, state = call(server, "GET", "/api/state")
    assert [t["name"] for t in state["tasks"]] == ["support"] and state["tasks"][0]["stale"]


def test_model_check_runs_on_a_local_onnx_folder(server):
    job = wait(server, call(server, "POST", "/api/tools/check-model", {"model": str(TINY / "split")})[1])
    assert job["status"] == "done", job
    r = job["result"]
    assert r["torch_loaded"] is False and len(r["results"]) == 6 and r["ms_per_decision"] >= 0


def test_errors_come_back_as_messages(server):
    call(server, "POST", "/api/tasks", {"name": "t", "questions": questions()})
    job = wait(server, call(server, "POST", "/api/tasks/t/calibrate", {})[1])
    assert job["status"] == "error" and "no labeled examples" in job["error"]
    st, e = call(server, "POST", "/api/tasks/t/predict", {"texts": ["  "]})
    assert st == 400 and "write a message" in e["error"]


def test_delete_needs_the_name_typed(server):
    call(server, "POST", "/api/tasks", {"name": "old", "questions": questions()})
    st, e = call(server, "POST", "/api/tasks/old/delete", {"confirm": "wrong"})
    assert st == 400 and "type the task's name" in e["error"]
    st, r = call(server, "POST", "/api/tasks/old/delete", {"confirm": "old"})
    assert st == 200 and ".deleted-old-" in r["moved_to"]
    assert call(server, "GET", "/api/state")[1]["tasks"] == []


def test_remove_and_undo_examples(server):
    call(server, "POST", "/api/tasks", {"name": "t", "questions": questions()})
    call(server, "POST", "/api/tasks/t/examples", {"rows": [{"state": {"body": "oops"}, "labels": {"department": "sales"}}]})
    ex = call(server, "GET", "/api/tasks/t/examples")[1]["rows"][0]
    st, r = call(server, "POST", "/api/tasks/t/examples/remove", {"ids": [ex["id"]]})
    assert r == {"changed": 1, "total": 0}
    assert call(server, "GET", "/api/tasks/t")[1]["examples"] == 0      # counts ignore removal markers
    st, r = call(server, "POST", "/api/tasks/t/examples/remove", {"ids": [ex["id"]], "restore": True})
    assert r["total"] == 1
