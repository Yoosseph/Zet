"""task.report(): every number measured on the held-out split, per question and language group."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build(task) -> Dict[str, Any]:
    cal = task.calibration()
    out: Dict[str, Any] = {"task": task.name, "examples": len(task.examples()),
                           "predictions": len(task.predictions()), "calibrated": cal is not None}
    if cal is None:
        out["warnings"] = ["not calibrated: add labeled examples and run calibrate()"]
        return out
    drift = task.drift()
    audits = _audit_counts(task, cal["version"])
    out.update({"version": cal["version"], "error_budget": cal["error_budget"], "alpha": cal["alpha"],
                "confidence": cal["confidence"], "language_method": cal["language_method"],
                "language_disagreements": cal["language_disagreements"], "split": cal["split"],
                "questions": {}})
    for qid, q in cal["questions"].items():
        test = q.get("test") or {"overall": {}, "groups": {}}
        rows = []
        for g in sorted(set(q["groups"]) | set(test["groups"])):
            e = q["groups"].get(g, {})
            rows.append(_row(g, e.get("n", 0), test["groups"].get(g, {}), e.get("threshold"),
                             e.get("threshold_source"), audits.get((qid, g), (0, 0)),
                             drift.get(qid, {}).get(g)))
        overall = _row("all", q["n_cal"], test["overall"], q["pooled"].get("threshold"), "pooled",
                       tuple(map(sum, zip(*[audits.get((qid, g), (0, 0)) for g in q["groups"]]))) or (0, 0),
                       None)
        out["questions"][qid] = {"type": q["type"], "n_labeled": q["n_labeled"], "n_test": q["n_test"],
                                 "fallback_groups": q["fallback_groups"], "warnings": q["warnings"],
                                 "overall": overall, "groups": rows}
    return out


def _row(group, n_cal, m, threshold, source, audit, drift) -> Dict[str, Any]:
    return {"group": group, "n_cal": n_cal, "n_test": m.get("n", 0), "coverage": m.get("coverage"),
            "avg_set_size": m.get("avg_set_size"), "automation": m.get("automation"),
            "n_sure": m.get("n_sure", 0), "sure_errors": m.get("sure_errors", 0),
            "laya_errors": m.get("laya_errors", 0), "review_errors": m.get("review_errors", 0),
            "sure_error": m.get("sure_error"), "sure_error_upper": m.get("sure_error_upper"),
            "threshold": threshold, "calibration": source, "audits": audit[0], "audit_verdicts": audit[1],
            "drift": (drift or {}).get("status", "no audits")}


def _audit_counts(task, version: int) -> Dict[tuple, tuple]:
    verdict_ids = {c["id"] for c in task.corrections()}
    counts: Dict[tuple, List[int]] = {}
    for rec in task.predictions():
        if rec.get("version") != version:
            continue
        for qid, a in rec["answers"].items():
            if a.get("audit"):
                c = counts.setdefault((qid, a.get("calibration_group") or rec.get("language")), [0, 0])
                c[0] += 1
                c[1] += rec["id"] in verdict_ids
    return {k: tuple(v) for k, v in counts.items()}


# -- text -------------------------------------------------------------------------------------------
def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _num(x: Optional[float], fmt: str = "{:.2f}") -> str:
    return "-" if x is None else fmt.format(x)


def render(r: Dict[str, Any]) -> str:
    lines = [f"Task {r['task']}: {r['examples']} labeled examples, {r['predictions']} predictions"]
    if not r["calibrated"]:
        return "\n".join(lines + ["  " + w for w in r.get("warnings", [])])
    sp = r["split"]
    lines += [
        f"Calibration v{r['version']}: error budget {r['error_budget']:.0%} among sure answers "
        f"(with {r['confidence']:.0%} confidence); prediction sets at {1 - r['alpha']:.0%} coverage.",
        f"All numbers below are measured on {sp['n_test']} held-out examples "
        f"({sp['test_fraction']:.0%} split, seed {sp['seed']}), never on the calibration data.",
        f"Language: explicit field, else {r['language_method']['detector']} detector"
        + (f"; detector disagreed with the given language on {r['language_disagreements']} example(s)"
           if r["language_disagreements"] else ""),
    ]
    head = ["group", "cal", "test", "coverage", "set size", "automated", "sure err", "err <= (95%)",
            "audits", "drift"]
    for qid, q in r["questions"].items():
        lines += ["", f"{qid} ({q['type']}): {q['n_labeled']} labeled, {q['n_test']} held out"]
        table = [head]
        for row in q["groups"] + [q["overall"]]:
            table.append([row["group"] + ("*" if row["group"] in q["fallback_groups"] else ""),
                          str(row["n_cal"]), str(row["n_test"]), _pct(row["coverage"]),
                          _num(row["avg_set_size"]), _pct(row["automation"]),
                          _pct(row["sure_error"]) + (f" ({row['sure_errors']}/{row['n_sure']})" if row["n_sure"] else ""),
                          _pct(row["sure_error_upper"]),
                          f"{row['audit_verdicts']}/{row['audits']}" if row["audits"] else "0",
                          row["drift"] if row["group"] != "all" else ""])
        widths = [max(len(t[i]) for t in table) for i in range(len(head))]
        for t in table:
            lines.append("  " + "  ".join(c.ljust(w) for c, w in zip(t, widths)).rstrip())
        if q["fallback_groups"]:
            lines.append(f"  * too few examples, pooled calibration used: {', '.join(q['fallback_groups'])}")
        for w in q["warnings"]:
            lines.append(f"  ! {w}")
    lines += ["", "coverage: how often the prediction set held the true answer. automated: share marked sure.",
              "sure err: measured error among sure answers; err <= (95%): its upper bound from the held-out",
              "sample alone (loose when few answers are sure). The calibrated budget assumes",
              "new data resembles the labeled examples; it is not a per-answer correctness check.",
              "Audits and drift warnings can reveal later failures, but cannot catch every mistake."]
    return "\n".join(lines)
