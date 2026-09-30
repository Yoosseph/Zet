"""Step 6: the CLI end to end, on the fake backend (no model download)."""
from __future__ import annotations

import json
import re

import yaml
from typer.testing import CliRunner

from conftest import QUESTIONS
from zet.cli import app

runner = CliRunner()


def run(*args):
    r = runner.invoke(app, list(args))
    assert r.exit_code == 0, r.output
    return r.output


def test_every_command_has_help_with_an_example():
    for cmd in ("init", "add-examples", "calibrate", "predict", "correct", "confirm", "report"):
        out = run(cmd, "--help")
        assert "Example:" in out and "zet " + cmd in out


def test_end_to_end(tmp_path):
    root = str(tmp_path / "zet")
    qfile = tmp_path / "questions.yaml"
    qfile.write_text(yaml.safe_dump(QUESTIONS, allow_unicode=True), encoding="utf-8")
    csv = tmp_path / "labeled.csv"
    rows = ["body,department,urgency,refund_requested,language"]
    for i in range(150):
        rows.append(f"Email number {i},{['billing', 'technical', 'sales', 'other'][i % 4]},{i % 3},"
                    f"{'yes' if i % 5 == 0 else 'no'},{'sv' if i % 5 == 0 else 'en'}")
    csv.write_text("\n".join(rows), encoding="utf-8")

    assert "created task 'support'" in run("init", "support", "-q", str(qfile), "--backend", "fake", "--root", root)
    assert "added 150 examples" in run("add-examples", "support", str(csv), "--root", root)
    out = run("calibrate", "support", "--error-budget", "0.1", "--root", root)
    assert "held-out" in out and "department (choice)" in out
    out = run("predict", "support", "Kan ni skicka om fakturan? Inte bråttom.", "--language", "sv", "--root", root)
    pid = re.search(r"^(p[0-9a-f]{12})", out, re.M).group(1)
    assert "[sv]" in out and "bråttom" not in out  # the summary shows answers, not the text
    assert "recorded correction" in run("correct", "support", pid, "department=sales", "--root", root)
    assert "confirmed" in run("confirm", "support", pid, "--root", root)
    data = json.loads(run("report", "support", "--json", "--root", root))
    assert data["calibrated"] and set(data["questions"]) == set(QUESTIONS)
    line = run("predict", "support", "hello", "--json", "--root", root).strip()
    assert json.loads(line)["answers"]["department"]["status"] in ("sure", "unsure")


def test_errors_are_one_line_messages(tmp_path):
    r = runner.invoke(app, ["report", "missing", "--root", str(tmp_path)])
    assert r.exit_code == 1 and "no task 'missing'" in r.output
