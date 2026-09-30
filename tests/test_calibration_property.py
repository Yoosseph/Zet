"""Step 4 property tests on simulated data (FakeBackend / direct records).

The simulated model reports probabilities p; the true label is drawn either from p (a calibrated
model) or from a flatter distribution (an over-confident model, like Laya out of the box).
"""
from __future__ import annotations

import numpy as np
import pytest

from zet import FakeBackend, Task
from zet.calibration import evaluate, fit_question

K = ["a", "b", "c", "d"]


def simulate(n, seed, overconfident=False, groups=("en",), rare=None):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        logits = rng.normal(0, 3.5, len(K))
        p = np.exp(logits - logits.max()); p /= p.sum()
        truth_p = p
        if overconfident:  # reality is flatter than what the model reports
            t = np.exp(logits / 1.4 - (logits / 1.4).max()); truth_p = t / t.sum()
        y = K[rng.choice(len(K), p=truth_p)]
        if rare is not None and rng.random() < 0.05:
            y = rare  # a rare class the model systematically underrates
        g = groups[i % len(groups)]
        out.append(({k: float(v) for k, v in zip(K, p)}, y, g))
    return out


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_conformal_coverage_across_seeds(alpha):
    covs = []
    for seed in range(20):
        data = simulate(2500, seed, overconfident=True)
        params = fit_question(data[:500], "choice", alpha, budget=0.05)
        covs.append(evaluate(params, data[500:])["overall"]["coverage"])
    assert np.mean(covs) >= 1 - alpha - 0.01, covs
    assert min(covs) >= 1 - alpha - 0.04, covs


def test_selective_threshold_keeps_sure_error_under_budget():
    budget, over, automated = 0.05, 0, 0
    for seed in range(20):
        data = simulate(5000, 100 + seed, overconfident=True)
        params = fit_question(data[:1000], "choice", 0.05, budget)
        m = evaluate(params, data[1000:])["overall"]
        if m["n_sure"]:
            automated += 1
            over += m["sure_error"] > budget + 0.015
    # An over-confident model can still automate its easiest answers in most splits (measured:
    # 29 of 40 with 1,000 calibration examples). A split whose first testable threshold (59 sure
    # answers at a 5% budget) holds an error stops the scan: the price of the guarantee.
    assert automated >= 11, automated
    assert over <= 1  # the guarantee holds with 95% confidence, so a rare miss is allowed


def test_overconfident_model_is_not_trusted_at_face_value():
    data = simulate(3000, 7, overconfident=True)
    params = fit_question(data[:1000], "choice", 0.05, 0.05)
    naive_sure = [(p, y) for p, y, _ in data[1000:] if max(p.values()) >= 0.9]
    naive_err = np.mean([max(p, key=p.get) != y for p, y in naive_sure])
    assert naive_err > 0.05          # trusting p >= 0.9 would blow the budget...
    assert params["pooled"]["threshold"] > 0.9   # ...so Zet demands more


def test_small_group_falls_back_to_pooled():
    data = simulate(600, 3, groups=("en",)) + [(p, y, "sv") for p, y, _ in simulate(5, 4)]
    params = fit_question(data, "choice", 0.05, 0.05)
    assert params["min_group_size"] == 19
    assert params["groups"]["en"]["threshold_source"] == "en"
    assert params["groups"]["en"]["cutoff_source"] == "en"
    assert params["groups"]["sv"]["threshold_source"] == "pooled"
    assert params["groups"]["sv"]["cutoff_source"] == "pooled"


def test_label_conditional_cutoffs_protect_a_rare_class():
    alpha = 0.1
    data = simulate(8000, 11, rare="d")
    params = fit_question(data[:3000], "choice", alpha, 0.05)
    assert "d" in params["groups"]["en"]["label_cutoffs"]
    rare_test = [r for r in data[3000:] if r[1] == "d"]
    cov = evaluate(params, rare_test)["overall"]["coverage"]
    assert cov >= 1 - alpha - 0.05, cov


def test_score_and_noul_questions():
    rng = np.random.default_rng(5)
    recs = []
    for _ in range(800):
        p = rng.dirichlet([1, 1, 1]); y = str(rng.choice(3, p=p))
        recs.append(({str(i): float(v) for i, v in enumerate(p)}, y, "en"))
    params = fit_question(recs[:300], "score", 0.1, 0.05)
    assert evaluate(params, recs[300:])["overall"]["coverage"] >= 0.85
    nrecs = []
    for _ in range(800):
        t = float(rng.beta(0.5, 0.5)); y = "true" if rng.random() < t else "false"
        nrecs.append(({"false": 1 - t, "true": t}, y, "en"))
    params = fit_question(nrecs[:300], "noul", 0.1, 0.05)
    assert evaluate(params, nrecs[300:])["overall"]["coverage"] >= 0.85


# -- end to end through Task ----------------------------------------------------------------------
QS = {"dept": {"type": "choice", "instructions": "Which department?", "criteria": K}}


def _task(tmp_path, n=1000, seed=0):
    def fn(state, qid, qdef, keys):
        return dict(zip(keys, state["sim"]))

    t = Task.create("sim", QS, backend=FakeBackend(fn), root=tmp_path)
    rows = []
    for i, (p, y, g) in enumerate(simulate(n, seed, groups=("en", "sv"))):
        rows.append({"state": {"body": f"email {i}", "sim": [p[k] for k in K]}, "labels": {"dept": y},
                     "language": g})
    t.add_examples(rows)
    return t


def test_calibrate_writes_a_version_and_measures_held_out(tmp_path):
    t = _task(tmp_path)
    cal = t.calibrate(error_budget=0.1, seed=1)
    assert cal["version"] == 1 and (t.path / "versions" / "1" / "calibration.json").is_file()
    q = cal["questions"]["dept"]
    assert q["n_labeled"] == 1000 and 280 <= q["n_test"] <= 320
    assert q["test"]["overall"]["n"] == q["n_test"]
    assert set(q["test"]["groups"]) == {"en", "sv"}
    assert t.calibrate(error_budget=0.1)["version"] == 2  # recalibration adds a version
    assert t.config["current_version"] == 2


def test_predict_after_calibration(tmp_path):
    t = _task(tmp_path)
    t.calibrate(error_budget=0.1)
    sure = t.predict({"body": "x", "sim": [0.999, 0.0005, 0.0003, 0.0002], "language": "en"})
    a = sure["dept"]
    assert a.status == "sure" and a.options == ["a"] and a.calibration_group == "en"
    unsure = t.predict({"body": "x", "sim": [0.45, 0.45, 0.05, 0.05], "language": "sv"})
    assert unsure["dept"].status == "unsure" and len(unsure["dept"].options) >= 2
    other = t.predict({"body": "x", "sim": [0.999, 0.0005, 0.0003, 0.0002], "language": "fi"})
    assert other["dept"].calibration_group == "pooled"
    assert any("no calibration data for language 'fi'" in w for w in other["dept"].warnings)


def test_small_data_carries_a_warning(tmp_path):
    t = _task(tmp_path, n=60)
    t.calibrate(error_budget=0.1)
    r = t.predict({"body": "x", "sim": [0.97, 0.01, 0.01, 0.01], "language": "en"})
    assert any("weak guarantee" in w for w in r["dept"].warnings)
