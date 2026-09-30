"""Every calibration formula against a hand-computed example (brief, step 4)."""
from __future__ import annotations

import pytest

from zet.calibration import (
    THRESHOLDS, conformal_cutoff, cp_lower, cp_upper, evaluate, min_group_size, prediction_set,
    select_threshold,
)


@pytest.mark.parametrize("n", [1, 30, 59, 100, 1000])
def test_cp_upper_with_zero_errors_has_a_closed_form(n):
    # P(X <= 0) = (1 - u)^n = 0.05  =>  u = 1 - 0.05^(1/n)
    assert cp_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-9)


def test_cp_upper_known_values():
    # scipy.stats.beta.ppf(0.95, k + 1, n - k), rounded to 3 decimals
    assert cp_upper(0, 59) == pytest.approx(0.050, abs=5e-4)
    assert cp_upper(1, 100) == pytest.approx(0.047, abs=5e-4)
    assert cp_upper(2, 100) == pytest.approx(0.062, abs=5e-4)
    assert cp_upper(2, 200) == pytest.approx(0.031, abs=5e-4)
    assert cp_upper(5, 5) == 1.0 and cp_upper(0, 0) == 1.0


def test_cp_lower():
    # all errors: P(X >= n) = l^n = 0.05  =>  l = 0.05^(1/n)
    assert cp_lower(10, 10) == pytest.approx(0.05 ** (1 / 10), abs=1e-9)
    assert cp_lower(0, 50) == 0.0
    # lower bound sits below the observed rate, upper above
    assert cp_lower(12, 200) < 12 / 200 < cp_upper(12, 200)


def test_min_group_size():
    assert min_group_size(0.05) == 19      # ceil(1/0.05) - 1
    assert min_group_size(0.1) == 9
    assert min_group_size(0.2) == 4


def test_conformal_cutoff_hand_example():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert conformal_cutoff(scores, 0.2) == 0.9          # k = ceil(11 * 0.8) = 9
    assert conformal_cutoff(scores, 0.5) == 0.6          # k = ceil(11 * 0.5) = 6
    assert conformal_cutoff(scores, 0.05) == 1.0         # k = 11 > n: every option
    assert conformal_cutoff([0.3] * 19, 0.05) == 0.3     # n = 19: k = ceil(19.0) = 19, no float slip
    assert conformal_cutoff([], 0.1) == 1.0


def test_prediction_sets():
    p = {"a": 0.7, "b": 0.2, "c": 0.1}
    assert prediction_set(p, 0.85, "choice") == ["a", "b"]           # 1-p: .3, .8, .9
    assert prediction_set(p, 0.1, "choice") == ["a"]                 # empty -> top option kept
    assert prediction_set(p, 1.0, "choice") == ["a", "b", "c"]
    assert prediction_set(p, 0.1, "choice", {"b": 0.85}) == ["a", "b"]  # label-conditional
    s = {"0": 0.5, "1": 0.05, "2": 0.45}
    assert prediction_set(s, 0.6, "score") == ["0", "1", "2"]       # 0 and 2 -> contiguous range


def test_threshold_grid_is_fixed_and_descending():
    assert len(THRESHOLDS) == 50 and THRESHOLDS[0] == pytest.approx(0.9999)
    assert all(a > b for a, b in zip(THRESHOLDS, THRESHOLDS[1:])) and THRESHOLDS[-1] == 0.0


def test_select_threshold_hand_example():
    # 100 confident correct answers at 0.99, then 100 at 0.6 of which 30 are wrong.
    top = [0.99] * 100 + [0.6] * 100
    correct = [True] * 100 + [True] * 70 + [False] * 30
    t, log = select_threshold(top, correct, [True] * 200, budget=0.05)
    # at t <= 0.99: 100 sure, 0 errors, upper = 1 - 0.05^(1/100) = 0.0295 <= 0.05 -> pass
    # at t <= 0.6: 200 sure, 30 errors, upper ~ 0.19 > 0.05 -> stop
    assert log[0]["n_sure"] == 100 and log[0]["passed"]
    assert log[-1]["n_sure"] == 200 and not log[-1]["passed"]
    assert 0.6 < t <= 0.99


def test_min_testable():
    from zet.calibration import min_testable
    assert min_testable(0.05) == 59     # 1 - 0.05^(1/59) = 0.0495 <= 0.05; 58 gives 0.0503
    assert min_testable(0.1) == 29
    assert cp_upper(0, 59) <= 0.05 < cp_upper(0, 58)


def test_few_confident_answers_do_not_stop_the_scan():
    # 3 answers at 0.9999 cannot be tested (n < 59); the scan must go on to 0.95 with 100 clean.
    top = [0.99995] * 3 + [0.95] * 100
    t, log = select_threshold(top, [True] * 103, [True] * 103, budget=0.05)
    assert log[0]["n_sure"] == 103 and t is not None and t <= 0.95


def test_select_threshold_none_when_nothing_qualifies():
    t, log = select_threshold([0.9] * 30, [True] * 30, [True] * 30, budget=0.05)
    assert t is None and log == []                   # 30 clean answers can never prove 5%
    t, _ = select_threshold([0.99] * 100, [True] * 100, [False] * 100, budget=0.05)
    assert t is None                                 # nothing eligible (no one-option sets)


def test_held_out_comparison_counts_all_four_outcomes():
    entry = {"cutoff": 0.4, "label_cutoffs": {}, "threshold": 0.8}
    params = {"type": "choice", "confidence": 0.95, "pooled": entry, "groups": {}}
    records = [
        ({"yes": 0.9, "no": 0.1}, "yes", "en"),  # correct and sure
        ({"yes": 0.9, "no": 0.1}, "no", "en"),   # wrong and sure
        ({"yes": 0.6, "no": 0.4}, "yes", "en"),  # correct and reviewed
        ({"yes": 0.6, "no": 0.4}, "no", "en"),   # wrong and reviewed
    ]
    m = evaluate(params, records)["overall"]
    assert (m["n"], m["n_sure"], m["laya_errors"], m["sure_errors"],
            m["review_errors"]) == (4, 2, 2, 1, 1)
