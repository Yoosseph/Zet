"""Step 5: audit sampling and drift warnings."""
from __future__ import annotations

import warnings

import pytest

from test_calibration_property import QS, _task
from zet import DriftWarning


def confident(label="a"):
    probs = {"a": 0.001, "b": 0.001, "c": 0.001, "d": 0.001}
    probs[label] = 0.997
    return [probs[k] for k in "abcd"]


def test_audit_rate_samples_only_sure_answers(tmp_path):
    t = _task(tmp_path)
    t.calibrate(error_budget=0.1)
    t.set_audit_rate(0.3)
    rs = t.predict([{"body": f"x{i}", "sim": confident(), "language": "en"} for i in range(400)])
    audited = [r for r in rs if r["dept"].audit]
    assert all(r["dept"].sure for r in audited)
    assert 0.2 < len(audited) / 400 < 0.4       # ~30%
    unsure = t.predict({"body": "u", "sim": [0.45, 0.45, 0.05, 0.05], "language": "en"})
    assert unsure["dept"].audit is False
    t.set_audit_rate(0.0, group="en")
    assert not any(r["dept"].audit for r in t.predict([{"body": "y", "sim": confident(), "language": "en"}] * 50))
    with pytest.raises(Exception, match="between 0 and 1"):
        t.set_audit_rate(2)


def _run_audits(t, truth_matches: bool, n=150):
    t.set_audit_rate(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DriftWarning)
        rs = t.predict([{"body": f"z{i}", "sim": confident("a"), "language": "en"} for i in range(n)])
    for i, r in enumerate(rs):
        if r["dept"].audit:
            if truth_matches or i % 3:        # shifted: a third of confident answers are wrong
                t.confirm(r.id)
            else:
                t.correct(r.id, dept="b")


def test_shifted_distribution_triggers_drift(tmp_path):
    t = _task(tmp_path)
    t.calibrate(error_budget=0.05)
    _run_audits(t, truth_matches=False)
    d = t.drift()["dept"]["en"]
    assert d["status"] == "drift" and d["audits"] == 150 and d["lower"] > 0.05
    with pytest.warns(DriftWarning, match="Recalibrate"):
        r = t.predict({"body": "again", "sim": confident(), "language": "en"})
    assert any("drift" in w for w in r.warnings)


def test_unshifted_distribution_does_not(tmp_path):
    t = _task(tmp_path)
    t.calibrate(error_budget=0.05)
    _run_audits(t, truth_matches=True)
    assert t.drift()["dept"]["en"]["status"] == "ok"
    with warnings.catch_warnings():
        warnings.simplefilter("error", DriftWarning)
        t.predict({"body": "fine", "sim": confident(), "language": "en"})


def test_few_audits_are_reported_as_insufficient(tmp_path):
    t = _task(tmp_path)
    t.calibrate(error_budget=0.05)
    _run_audits(t, truth_matches=False, n=30)
    assert t.drift()["dept"]["en"]["status"] == "insufficient audits"


def test_recalibration_starts_a_fresh_drift_window(tmp_path):
    t = _task(tmp_path)
    t.calibrate(error_budget=0.05)
    _run_audits(t, truth_matches=False)
    t.calibrate(error_budget=0.05)
    assert t.drift() == {}
