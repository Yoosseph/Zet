"""Calibration math. Pure functions; no I/O, no backend.

Two guarantees, kept separate (brief section 4):

A. Prediction sets (split conformal). Nonconformity score s = 1 - p(true label). On n calibration
   scores the cutoff is the ceil((n+1)(1-alpha))-th smallest; the set is every option with
   1 - p <= cutoff. It contains the true answer with probability >= 1 - alpha.

B. Automation with an error budget (selective prediction, Learn-then-Test). An answer is `sure`
   when its set has one option and its top probability clears a threshold t. t is chosen by
   fixed-sequence testing down a fixed grid: from the strictest threshold, keep lowering t while
   the one-sided 95% Clopper-Pearson upper bound of the error among sure answers stays within the
   budget; stop at the first failure. With 95% confidence, the error rate among sure answers is at
   most the budget.

   Thresholds with too few sure answers to ever pass are skipped: below
   n_min = ceil(log(1 - confidence) / log(1 - budget)) answers (59 for a 5% budget), even zero
   errors cannot bring the bound under the budget, so testing there would stop the scan before any
   real test. Whether a threshold is skipped depends only on how many answers clear it -- the
   model's probabilities, never the labels -- so the scan stays valid conditionally on them.

Groups (Mondrian): cutoffs and thresholds are fitted per language group. A group with fewer than
ceil(1/alpha) - 1 examples falls back to the pooled calibration. For choice questions, set
cutoffs are also label-conditional: (group, label) -> group -> pooled.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CONFIDENCE = 0.95
# Fixed, data-independent threshold grid, strictest first: 1 - 10^x for x from -4 to 0 (50 steps).
THRESHOLDS: Tuple[float, ...] = tuple(float(1.0 - 10.0 ** x) for x in np.linspace(-4.0, 0.0, 50))


# -- binomial bounds ------------------------------------------------------------------------------
def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space (stable for large n)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lgn = math.lgamma(n + 1)
    terms = [lgn - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def _bisect(f, lo: float, hi: float, iters: int = 100) -> float:
    """Root of a decreasing function f on [lo, hi]."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp_upper(errors: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided Clopper-Pearson upper bound on an error rate: errors out of n."""
    if n <= 0 or errors >= n:
        return 1.0
    alpha = 1.0 - confidence
    return _bisect(lambda p: _binom_cdf(errors, n, p) - alpha, errors / n, 1.0)


def cp_lower(errors: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided Clopper-Pearson lower bound on an error rate: errors out of n."""
    if n <= 0 or errors <= 0:
        return 0.0
    return _bisect(lambda p: _binom_cdf(errors - 1, n, p) - confidence, 0.0, errors / n)


# -- conformal prediction sets --------------------------------------------------------------------
def min_group_size(alpha: float) -> int:
    """Fewer calibration examples than this cannot give a finite cutoff at level alpha."""
    return max(1, math.ceil(1.0 / alpha - 1e-9) - 1)


def conformal_cutoff(scores: Sequence[float], alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score; 1.0 (every option) when n is too small."""
    n = len(scores)
    if n == 0:
        return 1.0
    k = math.ceil((n + 1) * (1.0 - alpha) - 1e-9)
    if k > n:
        return 1.0
    return float(sorted(scores)[k - 1])


def prediction_set(probs: Dict[str, float], cutoff: float, qtype: str,
                   label_cutoffs: Optional[Dict[str, float]] = None) -> List[str]:
    """Options with 1 - p <= cutoff (per label when given), in option order.

    Never empty: the top option is always included (decision Q8; can only raise coverage).
    Score questions report the contiguous range of levels spanning the set.
    """
    keys = list(probs)
    top = max(keys, key=probs.get)
    chosen = []
    for k in keys:
        c = label_cutoffs.get(k, cutoff) if label_cutoffs else cutoff
        if 1.0 - probs[k] <= c + 1e-12 or k == top:
            chosen.append(k)
    if qtype == "score":
        idx = [int(k) for k in chosen]
        chosen = [str(i) for i in range(min(idx), max(idx) + 1)]
    return chosen


# -- selective threshold (Learn-then-Test, fixed sequence) ----------------------------------------
def min_testable(budget: float, confidence: float = CONFIDENCE) -> int:
    """Fewest sure answers that can pass: zero errors in n gives upper bound 1 - (1-conf)^(1/n)."""
    return max(1, math.ceil(math.log(1.0 - confidence) / math.log(1.0 - budget) - 1e-9))


def select_threshold(top_p: Sequence[float], correct: Sequence[bool], eligible: Sequence[bool],
                     budget: float, grid: Sequence[float] = THRESHOLDS,
                     confidence: float = CONFIDENCE) -> Tuple[Optional[float], List[Dict]]:
    """The lowest grid threshold passing fixed-sequence testing, or None; plus the test log.

    An answer is sure at threshold t when it is eligible (a one-option set) and top_p >= t.
    """
    top = np.asarray(top_p, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    el = np.asarray(eligible, dtype=bool)
    n_min = min_testable(budget, confidence)
    chosen, log, last_n = None, [], -1
    for t in grid:
        sure = el & (top >= t)
        n = int(sure.sum())
        if n < n_min or n == last_n:
            continue  # untestable here (depends on counts only), or nothing new to test
        last_n = n
        errors = int((sure & ~ok).sum())
        upper = cp_upper(errors, n, confidence)
        passed = upper <= budget
        log.append({"threshold": t, "n_sure": n, "errors": errors, "upper": upper, "passed": passed})
        if not passed:
            break
        chosen = t
    return chosen, log


# -- fitting one question -------------------------------------------------------------------------
POOLED = "pooled"
Record = Tuple[Dict[str, float], str, str]  # (probs, true option key, language group)


def _entry(scores: List[float], labels: List[str], alpha: float, min_n: int, qtype: str,
           source: str, fallback: Optional[Dict] = None) -> Dict:
    """Cutoffs for one group: its own when it has enough data, else the fallback's."""
    own = len(scores) >= min_n or fallback is None
    e = {"n": len(scores), "cutoff": conformal_cutoff(scores, alpha) if own else fallback["cutoff"],
         "cutoff_source": source if own else fallback["cutoff_source"], "label_cutoffs": {}}
    if qtype == "choice":
        by_label: Dict[str, List[float]] = {}
        for s, y in zip(scores, labels):
            by_label.setdefault(y, []).append(s)
        for y, ys in by_label.items():
            if len(ys) >= min_n:
                e["label_cutoffs"][y] = {"cutoff": conformal_cutoff(ys, alpha), "source": f"{source}/{y}"}
        if not own:  # labels this group cannot calibrate itself come from the fallback
            for y, lc in fallback["label_cutoffs"].items():
                e["label_cutoffs"].setdefault(y, lc)
    return e


def apply_entry(entry: Dict, probs: Dict[str, float], qtype: str) -> Tuple[List[str], bool]:
    """(prediction set, sure) for one answer under one group's calibration."""
    lc = {y: v["cutoff"] for y, v in entry["label_cutoffs"].items()} or None
    options = prediction_set(probs, entry["cutoff"], qtype, lc)
    t = entry.get("threshold")
    sure = len(options) == 1 and t is not None and max(probs.values()) >= t
    return options, sure


def fit_question(records: Sequence[Record], qtype: str, alpha: float, budget: float,
                 confidence: float = CONFIDENCE) -> Dict:
    """Per-group conformal cutoffs and selective thresholds for one question."""
    min_n = min_group_size(alpha)
    scores = [1.0 - p[y] for p, y, _ in records]
    labels = [y for _, y, _ in records]
    pooled = _entry(scores, labels, alpha, min_n, qtype, POOLED)
    groups: Dict[str, Dict] = {}
    for g in sorted({g for _, _, g in records}):
        idx = [i for i, r in enumerate(records) if r[2] == g]
        groups[g] = _entry([scores[i] for i in idx], [labels[i] for i in idx], alpha, min_n, qtype, g, pooled)

    def ltt(idx: List[int], entry_of) -> Tuple[Optional[float], List[Dict]]:
        top, correct, eligible = [], [], []
        for i in idx:
            p, y, g = records[i]
            opts = prediction_set(p, entry_of(g)["cutoff"], qtype,
                                  {k: v["cutoff"] for k, v in entry_of(g)["label_cutoffs"].items()} or None)
            best = max(p, key=p.get)
            top.append(p[best]); correct.append(best == y); eligible.append(len(opts) == 1)
        return select_threshold(top, correct, eligible, budget, confidence=confidence)

    everyone = list(range(len(records)))
    pooled["threshold"], pooled["ltt"] = ltt(everyone, lambda g: groups[g])
    pooled["threshold_source"] = POOLED
    for g, e in groups.items():
        if e["n"] >= min_n:
            e["threshold"], e["ltt"] = ltt([i for i in everyone if records[i][2] == g], lambda _g, e=e: e)
            e["threshold_source"] = g
        else:
            e["threshold"], e["threshold_source"], e["ltt"] = pooled["threshold"], POOLED, []
    return {"type": qtype, "alpha": alpha, "budget": budget, "confidence": confidence,
            "min_group_size": min_n, "n_cal": len(records), "pooled": pooled, "groups": groups}


def entry_for(params: Dict, group: str) -> Tuple[Dict, str]:
    """The calibration used for `group`, and the name of the group it came from."""
    e = params["groups"].get(group)
    if e is None:
        return params["pooled"], POOLED
    return e, e.get("threshold_source", group)


def evaluate(params: Dict, records: Sequence[Record]) -> Dict:
    """Held-out metrics, overall and per group: coverage, set size, automation, sure error."""
    def empty():
        return {"n": 0, "covered": 0, "set_size": 0, "sure": 0, "sure_errors": 0,
                "laya_errors": 0, "review_errors": 0}

    acc: Dict[str, Dict] = {"overall": empty()}
    for p, y, g in records:
        entry, _ = entry_for(params, g)
        options, sure = apply_entry(entry, p, params["type"])
        top = max(p, key=p.get)
        wrong = top != y
        for key in ("overall", g):
            a = acc.setdefault(key, empty())
            a["n"] += 1; a["covered"] += y in options; a["set_size"] += len(options)
            a["sure"] += sure; a["sure_errors"] += sure and wrong
            a["laya_errors"] += wrong; a["review_errors"] += not sure and wrong

    def summarise(a: Dict) -> Dict:
        n, s = a["n"], a["sure"]
        return {"n": n, "coverage": a["covered"] / n if n else None,
                "avg_set_size": a["set_size"] / n if n else None,
                "automation": s / n if n else None, "n_sure": s, "sure_errors": a["sure_errors"],
                "laya_errors": a["laya_errors"], "review_errors": a["review_errors"],
                "sure_error": a["sure_errors"] / s if s else None,
                "sure_error_upper": cp_upper(a["sure_errors"], s, params["confidence"]) if s else None}

    return {"overall": summarise(acc.pop("overall")), "groups": {g: summarise(a) for g, a in sorted(acc.items())}}
