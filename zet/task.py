"""Tasks: a named, saved decision setup.

On disk, `<root>/<name>/` holds:
- task.yaml          questions, backend, question hash, current version, audit settings
- examples.jsonl     labeled examples (append-only)
- predictions.jsonl  every prediction: id, state, language, answers, audit flag (append-only)
- corrections.jsonl  human verdicts on predictions: corrections and confirmations (append-only)
- versions/<n>/calibration.json   calibration for version n (0.1 only writes version 1)

The question hash guards validity: calibration is stored with the hash of the questions it was
measured on, and `predict` refuses a calibration whose hash no longer matches.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml

from . import __version__
from . import calibration as _cal
from . import examples as _examples
from .language import get_detector
from .backends import BaseBackend, get_backend
from .backends.base import prepare_questions
from .results import Answer, Result

State = Union[str, Dict[str, Any]]
LANGUAGE_FIELD = "language"


class ZetError(Exception):
    """A problem the user can fix; the message says how."""


class StaleCalibrationError(ZetError):
    pass


class DriftWarning(UserWarning):
    """Audited sure answers are wrong more often than the error budget allows: recalibrate."""


DRIFT_WINDOW = 200       # audits per (question, group) considered
DRIFT_MIN_AUDITS = 60    # below this, drift status is "insufficient audits"


def default_root() -> Path:
    """Where tasks live unless --root / root= says otherwise: $ZET_ROOT, else ~/zet.

    Not the current folder: run from the repository, ./zet would be the package's own source."""
    import os

    return Path(os.environ.get("ZET_ROOT") or Path.home() / "zet")


def list_tasks(root: Union[str, Path, None] = None) -> List[str]:
    """Names of the tasks under `root`, sorted."""
    root = Path(root) if root is not None else default_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "task.yaml").is_file() and not p.name.startswith("."))


def question_hash(questions: Dict[str, Dict]) -> str:
    blob = json.dumps(questions, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _as_state(state: State) -> Dict[str, Any]:
    if isinstance(state, str):
        return {"body": state}
    if isinstance(state, dict):
        return dict(state)
    raise TypeError(f"a state is a dict or a string, got {type(state).__name__}")


class Task:
    """A named decision setup: questions, backend, examples, calibration, logs."""

    def __init__(self, path: Path, config: Dict[str, Any], backend: Optional[BaseBackend] = None):
        self.path = Path(path)
        self.config = config
        self.name: str = config["name"]
        self.questions: Dict[str, Dict] = config["questions"]
        self._backend = backend

    # -- creation and loading -------------------------------------------------------------------
    @classmethod
    def create(cls, name: str, questions: Dict[str, Dict], backend: Union[str, BaseBackend] = "laya-onnx",
               backend_options: Optional[Dict[str, Any]] = None, root: Union[str, Path, None] = None,
               audit_rate: float = 0.02, overwrite: bool = False) -> "Task":
        """Create a task on disk. `backend` is a registered name or a backend instance."""
        if not name or any(c in name for c in '/\\:*?"<>|') or name in (".", ".."):
            raise ZetError(f"{name!r} is not a usable task name; use letters, digits, - and _")
        prepare_questions(questions)  # validates, with a message naming the bad question
        path = Path(root if root is not None else default_root()) / name
        if (path / "task.yaml").exists() and not overwrite:
            raise ZetError(f"task {name!r} already exists at {path}; load it with Task.load({name!r}) "
                           "or pass overwrite=True")
        instance = backend if isinstance(backend, BaseBackend) else None
        backend_name = backend.name if instance is not None else str(backend)
        config = {
            "name": name, "zet_version": __version__, "created": _now(),
            "questions": questions, "question_hash": question_hash(questions),
            "backend": {"name": backend_name, "options": dict(backend_options or {})},
            "current_version": None, "audit_rate": float(audit_rate), "audit_rate_by_group": {},
        }
        path.mkdir(parents=True, exist_ok=True)
        for fname in ("examples.jsonl", "predictions.jsonl", "corrections.jsonl"):
            (path / fname).touch()
        task = cls(path, config, instance)
        task._save_config()
        return task

    @classmethod
    def load(cls, name: str, root: Union[str, Path, None] = None, backend: Optional[BaseBackend] = None) -> "Task":
        path = Path(root if root is not None else default_root()) / name
        cfg_path = path / "task.yaml"
        if not cfg_path.is_file():
            raise ZetError(f"no task {name!r} at {path}; create it with Task.create or `zet init`")
        with open(cfg_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(path, config, backend)

    def _save_config(self) -> None:
        tmp = self.path / "task.yaml.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)
        tmp.replace(self.path / "task.yaml")

    @property
    def backend(self) -> BaseBackend:
        if self._backend is None:
            spec = self.config["backend"]
            self._backend = get_backend(spec["name"], **spec.get("options", {}))
        return self._backend

    # -- examples -------------------------------------------------------------------------------
    def add_examples(self, source: Union[str, Path, Iterable[Dict[str, Any]]]) -> int:
        """Add labeled examples from a .csv/.jsonl file or a list of dicts. Returns how many."""
        new = _examples.load(source, self.questions)
        # Number from every example ever added, removed ones included, so ids are never reused.
        start = sum(1 for r in _read_jsonl(self.path / "examples.jsonl") if "state" in r)
        for i, ex in enumerate(new):
            ex["id"] = "e%06d" % (start + i + 1)
            ex["added"] = _now()
        _append_jsonl(self.path / "examples.jsonl", new)
        return len(new)

    def examples(self) -> List[Dict[str, Any]]:
        """The current labeled examples: everything added, minus what was removed.

        examples.jsonl stays append-only: removing appends {"remove": id} and undoing appends
        {"restore": id}, so the file keeps the full history."""
        rows, removed = [], set()
        for r in _read_jsonl(self.path / "examples.jsonl"):
            if "remove" in r:
                removed.add(r["remove"])
            elif "restore" in r:
                removed.discard(r["restore"])
            else:
                rows.append(r)
        return [r for r in rows if r.get("id") not in removed]

    def remove_examples(self, ids: Iterable[str]) -> int:
        """Remove examples by id (undo with restore_examples). Recalibrate to use the change."""
        current = {e["id"] for e in self.examples()}
        ids = [i for i in ids if i in current]
        _append_jsonl(self.path / "examples.jsonl", [{"remove": i, "time": _now()} for i in ids])
        return len(ids)

    def restore_examples(self, ids: Iterable[str]) -> int:
        """Bring back removed examples."""
        ever = {r["id"] for r in _read_jsonl(self.path / "examples.jsonl") if "state" in r}
        current = {e["id"] for e in self.examples()}
        ids = [i for i in ids if i in ever and i not in current]
        _append_jsonl(self.path / "examples.jsonl", [{"restore": i, "time": _now()} for i in ids])
        return len(ids)

    # -- language -------------------------------------------------------------------------------
    def resolve_language(self, state: Dict[str, Any], explicit: Optional[str] = None,
                         method: Optional[Dict[str, Any]] = None) -> str:
        """Q17: an explicit language (field or example column) wins, else the detector the
        calibration was measured with (the built-in one when uncalibrated)."""
        lang = explicit or state.get(LANGUAGE_FIELD)
        if isinstance(lang, str) and lang.strip():
            return lang.strip()
        method = method or {"detector": "builtin"}
        return get_detector(method["detector"], method.get("languages"))(_model_state(state))

    # -- calibration ----------------------------------------------------------------------------
    def calibrate(self, error_budget: float = 0.05, alpha: Optional[float] = None,
                  test_fraction: float = 0.3, seed: int = 0, language_detector: str = "builtin",
                  languages: Optional[List[str]] = None, batch_size: int = 64,
                  progress=None) -> Dict[str, Any]:
        """Fit calibration on the labeled examples and measure it on a held-out split.

        error_budget: the most errors allowed among `sure` answers (with 95% confidence).
        alpha: prediction sets miss the truth at most this often; defaults to error_budget.
        test_fraction: share of examples held out for the reported numbers (stratified by
            language and label, fixed seed).
        language_detector: "builtin" (Laya's, extended) or "lingua" (pip install zet[langid],
            limited to `languages`). predict() will always use the same one.
        """
        if not 0 < error_budget < 1:
            raise ZetError("error_budget must be between 0 and 1, e.g. 0.05")
        alpha = float(alpha if alpha is not None else error_budget)
        exs = self.examples()
        if not exs:
            raise ZetError(f"task {self.name!r} has no labeled examples; add some with add_examples()")
        method = {"detector": language_detector, "languages": list(languages) if languages else None}
        get_detector(language_detector, method["languages"])  # fail now if it is unavailable

        states = [ex["state"] for ex in exs]
        probs: List[Dict[str, Dict[str, float]]] = []
        for i in range(0, len(states), batch_size):
            if progress:
                progress(i, len(states), f"model reading examples {i}/{len(states)}")
            probs.extend(self.backend.predict_proba([_model_state(s) for s in states[i:i + batch_size]],
                                                    self.questions))
        if progress:
            progress(len(states), len(states), "fitting calibration")
        langs = [self.resolve_language(ex["state"], ex.get("language"), method) for ex in exs]
        detected = [self.resolve_language(ex["state"], None, method) if ex.get("language") else None
                    for ex in exs]
        disagree = sum(1 for ex, d in zip(exs, detected)
                       if d is not None and d != "unknown" and not d.startswith("und-")
                       and d.split("-")[0].lower() != ex["language"].split("-")[0].lower())
        test_idx = _stratified_split(exs, langs, test_fraction, seed)

        questions_out = {}
        for qid, qdef in self.questions.items():
            idx = [i for i, ex in enumerate(exs) if qid in ex["labels"]]
            rec = lambda i: (probs[i][qid], exs[i]["labels"][qid], langs[i])  # noqa: E731
            cal_recs = [rec(i) for i in idx if i not in test_idx]
            test_recs = [rec(i) for i in idx if i in test_idx]
            params = _cal.fit_question(cal_recs, qdef["type"], alpha, error_budget)
            warn = []
            if len(idx) < 100:
                warn.append(f"weak guarantee: only {len(idx)} labeled examples (fewer than 100)")
            if not test_recs:
                warn.append("no held-out examples: nothing measured; add more labeled examples")
            fallback = sorted(g for g, e in params["groups"].items() if e["threshold_source"] != g)
            params.update({"n_labeled": len(idx), "n_test": len(test_recs),
                           "test": _cal.evaluate(params, test_recs) if test_recs else None,
                           "fallback_groups": fallback, "warnings": warn})
            questions_out[qid] = params

        version = int(self.config.get("current_version") or 0) + 1
        cal = {"version": version, "created": _now(), "zet_version": __version__,
               "question_hash": question_hash(self.questions), "backend": self.config["backend"],
               "error_budget": error_budget, "alpha": alpha, "confidence": _cal.CONFIDENCE,
               "language_method": method, "language_disagreements": disagree,
               "split": {"test_fraction": test_fraction, "seed": seed,
                         "n_examples": len(exs), "n_test": len(test_idx)},
               "questions": questions_out}
        vdir = self.path / "versions" / str(version)
        vdir.mkdir(parents=True, exist_ok=True)
        with open(vdir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(cal, f, ensure_ascii=False, indent=1)
        self.config["current_version"] = version
        self.config["error_budget"] = error_budget
        self._save_config()
        return cal

    def calibration(self) -> Optional[Dict[str, Any]]:
        """The current version's calibration, or None when the task is not calibrated."""
        v = self.config.get("current_version")
        if v is None:
            return None
        path = self.path / "versions" / str(v) / "calibration.json"
        if not path.is_file():
            raise ZetError(f"task.yaml points at version {v}, but {path} is missing; recalibrate")
        with open(path, encoding="utf-8") as f:
            cal = json.load(f)
        if cal.get("backend") != self.config["backend"]:
            raise StaleCalibrationError(
                f"the model of task {self.name!r} changed since version {v} was calibrated "
                f"({cal.get('backend')} -> {self.config['backend']}); its guarantee no longer applies. "
                "Run task.calibrate() (or `zet calibrate`) again.")
        if cal.get("question_hash") != question_hash(self.questions):
            raise StaleCalibrationError(
                f"the questions of task {self.name!r} changed since version {v} was calibrated "
                f"(hash {cal.get('question_hash')} -> {question_hash(self.questions)}); its guarantee no "
                "longer applies. Run task.calibrate() (or `zet calibrate`) again.")
        return cal

    # -- prediction -----------------------------------------------------------------------------
    def predict(self, states: Union[State, List[State]]) -> Union[Result, List[Result]]:
        """Answer every question for one state (returns a Result) or a list (returns a list)."""
        single = not isinstance(states, list)
        items = [_as_state(s) for s in ([states] if single else states)]
        if not items:
            return []
        cal = self.calibration()
        method = cal["language_method"] if cal else None
        languages = [self.resolve_language(s, None, method) for s in items]
        preds = self.backend.predict([_model_state(s) for s in items], self.questions)
        results, records = [], []
        for state, lang, pred in zip(items, languages, preds):
            rid = "p" + uuid.uuid4().hex[:12]
            answers = self._answers(rid, pred.probs, lang, cal)
            res = Result(id=rid, answers=answers, language=lang, calibrated=cal is not None,
                         version=self.config.get("current_version"))
            if cal is None:
                res.warnings.append("task is not calibrated: every answer is unsure. "
                                    "Add labeled examples and run calibrate().")
            results.append(res)
            records.append({"id": rid, "time": _now(), "version": res.version, "state": state,
                            "language": lang, "answers": {q: a.to_dict() for q, a in answers.items()},
                            "meta": pred.meta})
        _append_jsonl(self.path / "predictions.jsonl", records)
        if cal is not None:
            for msg in self._drift_warnings():
                warnings.warn(msg, DriftWarning, stacklevel=2)
                for r in results:
                    r.warnings.append(msg)
        return results[0] if single else results

    def _answers(self, rid: str, probs: Dict[str, Dict[str, float]], language: str,
                 cal: Optional[Dict[str, Any]]) -> Dict[str, Answer]:
        answers = {}
        for qid in self.questions:
            p = probs[qid]
            top = max(p, key=p.get)
            if cal is None:
                answers[qid] = Answer(question=qid, answer=top, status="unsure", options=list(p), probs=p)
                continue
            params = cal["questions"][qid]
            entry, source = _cal.entry_for(params, language)
            options, sure = _cal.apply_entry(entry, p, params["type"])
            warn = list(params.get("warnings", []))
            if language not in params["groups"]:
                warn.append(f"no calibration data for language {language!r}; pooled calibration used")
            elif source != language:
                warn.append(f"too few examples for language {language!r}; pooled calibration used")
            answers[qid] = Answer(question=qid, answer=top, status="sure" if sure else "unsure",
                                  options=options, probs=p, calibration_group=source, warnings=warn,
                                  audit=sure and _sampled(rid, qid, self.audit_rate(language)))
        return answers

    # -- audits and drift -----------------------------------------------------------------------
    def audit_rate(self, group: Optional[str] = None) -> float:
        return float(self.config.get("audit_rate_by_group", {}).get(group, self.config.get("audit_rate", 0.02)))

    def set_audit_rate(self, rate: float, group: Optional[str] = None) -> None:
        """Share of sure answers sampled for a human spot check (default 2%); per language if given."""
        if not 0 <= rate <= 1:
            raise ZetError("audit rate must be between 0 and 1, e.g. 0.02")
        if group is None:
            self.config["audit_rate"] = float(rate)
        else:
            self.config.setdefault("audit_rate_by_group", {})[group] = float(rate)
        self._save_config()

    def drift(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Per question and language group: error rate of audited sure answers over the last
        DRIFT_WINDOW verdicts, with its 95% Clopper-Pearson lower bound. Status is "drift" when
        that lower bound exceeds the error budget, "ok", or "insufficient audits"."""
        cal = self.calibration()
        if cal is None:
            return {}
        budget, version = cal["error_budget"], cal["version"]
        verdicts: Dict[str, Dict[str, str]] = {}
        for c in self.corrections():  # later verdicts override earlier ones
            verdicts.setdefault(c["id"], {}).update(c["labels"])
        audits: Dict[tuple, List[bool]] = {}
        for rec in self.predictions():
            if rec.get("version") != version or rec["id"] not in verdicts:
                continue
            for qid, a in rec["answers"].items():
                if a.get("audit") and a.get("status") == "sure" and qid in verdicts[rec["id"]]:
                    group = a.get("calibration_group") or rec.get("language")
                    audits.setdefault((qid, group), []).append(verdicts[rec["id"]][qid] != a["answer"])
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for (qid, group), errs in audits.items():
            window = errs[-DRIFT_WINDOW:]
            n, e = len(window), sum(window)
            lower = _cal.cp_lower(e, n)
            status = ("insufficient audits" if n < DRIFT_MIN_AUDITS
                      else "drift" if lower > budget else "ok")
            out.setdefault(qid, {})[group] = {"audits": n, "errors": e, "error_rate": e / n,
                                              "lower": lower, "budget": budget, "status": status}
        return out

    def _drift_warnings(self) -> List[str]:
        msgs = []
        for qid, groups in self.drift().items():
            for g, d in groups.items():
                if d["status"] == "drift":
                    msgs.append(f"drift: {qid!r} in group {g!r}: {d['errors']}/{d['audits']} audited sure "
                                f"answers were wrong (at least {d['lower']:.1%} with 95% confidence, budget "
                                f"{d['budget']:.0%}). Recalibrate with fresh labeled examples.")
        return msgs

    # -- report ---------------------------------------------------------------------------------
    def report(self, print_it: bool = True) -> Dict[str, Any]:
        """Per question and language group, all measured on held-out data: coverage, set size,
        automation rate, error among sure answers with its 95% upper bound, fallbacks, audits,
        drift. Prints a table and returns the same numbers as a dict."""
        from . import report as _report

        r = _report.build(self)
        if print_it:
            print(_report.render(r))
        return r

    # -- removal ------------------------------------------------------------------------------
    def delete(self) -> Path:
        """Remove the task from the task list. Recoverable: the folder is renamed to
        `.deleted-<name>-<time>` in the same place; rename it back to restore it."""
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self.path.with_name(f".deleted-{self.name}-{stamp}")
        self.path.rename(target)
        self.path = target
        return target

    # -- model --------------------------------------------------------------------------------
    def set_backend(self, backend: Union[str, BaseBackend], options: Optional[Dict[str, Any]] = None) -> None:
        """Switch the model. The current calibration stops applying until you recalibrate."""
        instance = backend if isinstance(backend, BaseBackend) else None
        self.config["backend"] = {"name": backend.name if instance else str(backend),
                                  "options": dict(options or {})}
        self._backend = instance
        self._save_config()

    # -- review ---------------------------------------------------------------------------------
    def verdicts(self) -> Dict[str, Dict[str, str]]:
        """Latest human label per prediction id and question (corrections and confirmations)."""
        out: Dict[str, Dict[str, str]] = {}
        for c in self.corrections():
            out.setdefault(c["id"], {}).update(c["labels"])
        return out

    def review_queue(self) -> List[Dict[str, Any]]:
        """Predictions of the current calibration that still need a person: an audit sample or
        an unsure answer without a verdict. Audits first (they feed drift), newest first."""
        v = self.config.get("current_version")
        if v is None:
            return []
        verdicts = self.verdicts()
        queue = []
        for rec in self.predictions():
            if rec.get("version") != v:
                continue
            done = verdicts.get(rec["id"], {})
            pending = [q for q, a in rec["answers"].items()
                       if q not in done and (a.get("audit") or a.get("status") != "sure")]
            if pending:
                audit = any(rec["answers"][q].get("audit") for q in pending)
                queue.append({**rec, "pending": pending, "is_audit": audit})
        queue.reverse()
        queue.sort(key=lambda r: not r["is_audit"])
        return queue

    # -- human verdicts -------------------------------------------------------------------------
    def prediction(self, prediction_id: str) -> Dict[str, Any]:
        for rec in reversed(_read_jsonl(self.path / "predictions.jsonl")):
            if rec["id"] == prediction_id:
                return rec
        raise ZetError(f"no prediction {prediction_id!r} in task {self.name!r}")

    def correct(self, prediction_id: str, **labels: Any) -> None:
        """Record the true answer(s) for a prediction, e.g. task.correct(r.id, department="sales")."""
        rec = self.prediction(prediction_id)
        if not labels:
            raise ZetError("give at least one label, e.g. correct(id, department='sales')")
        norm = {}
        for qid, v in labels.items():
            if qid not in self.questions:
                raise ZetError(f"unknown question {qid!r}; questions are {sorted(self.questions)}")
            norm[qid] = _examples.normalise_label(qid, self.questions[qid], v)
        _append_jsonl(self.path / "corrections.jsonl", [{
            "id": prediction_id, "time": _now(), "kind": "correction", "labels": norm,
            "predicted": {q: rec["answers"][q]["answer"] for q in norm}}])

    def confirm(self, prediction_id: str, *questions: str) -> None:
        """Record that the predicted answers were right (all questions, or the ones named)."""
        rec = self.prediction(prediction_id)
        qs = list(questions) or list(self.questions)
        labels = {q: rec["answers"][q]["answer"] for q in qs}
        _append_jsonl(self.path / "corrections.jsonl", [{
            "id": prediction_id, "time": _now(), "kind": "confirm", "labels": labels, "predicted": labels}])

    def corrections(self) -> List[Dict[str, Any]]:
        return _read_jsonl(self.path / "corrections.jsonl")

    def predictions(self) -> List[Dict[str, Any]]:
        return _read_jsonl(self.path / "predictions.jsonl")


def _model_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """What the model reads: the state without Zet's own `language` field, so calibration and
    prediction feed the model identical input whether or not a language was given."""
    return {k: v for k, v in state.items() if k != LANGUAGE_FIELD}


def _stratified_split(exs: List[Dict[str, Any]], langs: List[str], test_fraction: float, seed: int) -> set:
    """Indices of the held-out examples: per (language, first label) stratum, a fixed-seed shuffle."""
    import random

    strata: Dict[tuple, List[int]] = {}
    for i, (ex, lang) in enumerate(zip(exs, langs)):
        first = next(iter(sorted(ex["labels"].items())), (None, None))
        strata.setdefault((lang, first), []).append(i)
    rng = random.Random(seed)
    test = set()
    for key in sorted(strata, key=repr):
        idx = strata[key]
        rng.shuffle(idx)
        test.update(idx[: int(round(len(idx) * test_fraction))])
    return test


def _sampled(prediction_id: str, question: str, rate: float) -> bool:
    """Audit sampling: uniform and independent of the answer, reproducible from the ids."""
    if rate <= 0:
        return False
    u = int(hashlib.sha256(f"{prediction_id}:{question}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return u < rate
