"""The Zet interface: a local web server with a JSON API and one page.

Start it by double-clicking Zet.bat (Windows), or `zet ui`, or `python -m zet.ui`. It listens on
127.0.0.1 only. POST requests must carry the header `X-Zet: 1`: browsers refuse to send custom
headers cross-site without a CORS preflight, which this server never grants, so other websites
cannot drive it.

Everything slow (model download and loading, calibration, the benchmark) runs as a background job
the page polls, so the interface never freezes.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from .. import __version__
from ..task import StaleCalibrationError, Task, ZetError, default_root, list_tasks

STATIC = Path(__file__).parent / "static"
REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "benchmarks" / "massive" / "run.py"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# -- background jobs -------------------------------------------------------------------------------
class Jobs:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn: Callable[[Callable[..., None]], Any], task: Optional[str] = None) -> Dict:
        job = {"id": uuid.uuid4().hex[:10], "kind": kind, "task": task, "status": "running",
               "message": "Starting", "progress": None, "result": None, "error": None,
               "started": time.time(), "finished": None}

        def report(message: Optional[str] = None, done: Optional[int] = None, total: Optional[int] = None):
            if message:
                job["message"] = message
            if total:
                job["progress"] = [done or 0, total]

        def run():
            try:
                job["result"] = fn(report)
                job["status"] = "done"
            except Exception as e:  # shown to the person; details kept for debugging
                job["status"] = "error"
                job["error"] = str(e) or type(e).__name__
                job["trace"] = traceback.format_exc(limit=6)
            job["finished"] = time.time()

        with self._lock:
            self._jobs[job["id"]] = job
            if len(self._jobs) > 100:
                for old in sorted(self._jobs.values(), key=lambda j: j["started"])[:50]:
                    self._jobs.pop(old["id"], None)
        threading.Thread(target=run, daemon=True).start()
        return self.public(job)

    def get(self, job_id: str) -> Dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise ApiError("no such job (the interface was probably restarted)", 404)
        return self.public(job)

    @staticmethod
    def public(job: Dict) -> Dict:
        return {k: v for k, v in job.items() if k != "trace"}


# -- the application ---------------------------------------------------------------------------------
class App:
    def __init__(self, root: Optional[Path] = None, backend_factory=None):
        self.root = Path(root) if root else default_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = Jobs()
        self._tasks: Dict[str, Task] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._backend_factory = backend_factory  # tests inject FakeBackend here

    # tasks are cached so a loaded model stays in memory between requests
    def task(self, name: str) -> Task:
        t = self._tasks.get(name)
        if t is None:
            try:
                t = Task.load(name, root=self.root,
                              backend=self._backend_factory(name) if self._backend_factory else None)
            except ZetError as e:
                raise ApiError(str(e), 404)
            self._tasks[name] = t
        return t

    def lock(self, name: str) -> threading.Lock:
        return self._locks.setdefault(name, threading.Lock())

    def forget(self, name: str) -> None:
        self._tasks.pop(name, None)

    # -- summaries --------------------------------------------------------------------------------
    def state(self) -> Dict:
        return {"version": __version__, "root": str(self.root), "benchmark": BENCHMARK.is_file(),
                "tasks": [self.summary(n) for n in list_tasks(self.root)]}

    def summary(self, name: str) -> Dict:
        t = self.task(name)
        cal, stale = self._calibration(t)
        drift = False
        if cal:
            try:
                drift = any(d["status"] == "drift" for g in t.drift().values() for d in g.values())
            except ZetError:
                pass
        return {"name": name, "examples": len(t.examples()),
                "predictions": _count_lines(t.path / "predictions.jsonl"),
                "version": cal["version"] if cal else None, "stale": stale, "drift": drift,
                "review": len(t.review_queue()) if cal else 0}

    @staticmethod
    def _calibration(t: Task) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            return t.calibration(), None
        except StaleCalibrationError as e:
            return None, str(e)
        except ZetError as e:
            return None, str(e)

    def detail(self, name: str) -> Dict:
        t = self.task(name)
        cal, stale = self._calibration(t)
        versions = []
        vroot = t.path / "versions"
        if vroot.is_dir():
            for d in sorted(vroot.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0, reverse=True):
                f = d / "calibration.json"
                if f.is_file():
                    c = json.loads(f.read_text(encoding="utf-8"))
                    versions.append({"version": c["version"], "created": c["created"],
                                     "error_budget": c["error_budget"], "examples": c["split"]["n_examples"],
                                     "backend": c.get("backend")})
        langs = sorted({ex.get("language") for ex in t.examples() if ex.get("language")})
        return {**self.summary(name), "path": str(t.path), "questions": t.questions,
                "backend": t.config["backend"], "audit_rate": t.config.get("audit_rate", 0.02),
                "audit_rate_by_group": t.config.get("audit_rate_by_group", {}),
                "error_budget": cal["error_budget"] if cal else t.config.get("error_budget"),
                "calibrated_at": cal["created"] if cal else None, "versions": versions,
                "languages": langs}


# -- request handling -----------------------------------------------------------------------------
def make_handler(app: App):
    routes: List[Tuple[str, str, Callable]] = []

    def route(method: str, pattern: str):
        def deco(fn):
            routes.append((method, pattern, fn))
            return fn
        return deco

    # state and tasks
    @route("GET", "/api/state")
    def _state(q, body):
        return app.state()

    @route("POST", "/api/tasks")
    def _create(q, body):
        name = (body.get("name") or "").strip()
        try:
            backend = body.get("backend") or "laya-onnx"
            instance = app._backend_factory(name) if app._backend_factory else backend
            Task.create(name, body.get("questions") or {}, backend=instance,
                        backend_options=body.get("options") or {}, root=app.root,
                        audit_rate=float(body.get("audit_rate", 0.02)))
        except (ZetError, ValueError) as e:
            raise ApiError(str(e))
        app.forget(name)
        return app.detail(name)

    @route("GET", "/api/tasks/{t}")
    def _detail(q, body, t):
        return app.detail(t)

    # predictions
    @route("POST", "/api/tasks/{t}/predict")
    def _predict(q, body, t):
        texts = [x for x in (body.get("texts") or []) if str(x).strip()]
        if not texts:
            raise ApiError("write a message to decide on")
        lang = (body.get("language") or "").strip()
        states = [{"body": str(x).strip(), **({"language": lang} if lang else {})} for x in texts]
        task = app.task(t)

        def work(report):
            with app.lock(t):
                report("Loading the model (the first run downloads it, about 1.3 GB)")
                results = task.predict(states)
                return [r.to_dict() | {"state": s} for r, s in zip(results, states)]
        return app.jobs.start("predict", work, t)

    @route("POST", "/api/tasks/{t}/calibrate")
    def _calibrate(q, body, t):
        task = app.task(t)
        args = {"error_budget": float(body.get("error_budget", 0.05)),
                "alpha": float(body["alpha"]) if body.get("alpha") not in (None, "") else None,
                "test_fraction": float(body.get("test_fraction", 0.3)), "seed": int(body.get("seed", 0))}

        def work(report):
            with app.lock(t):
                report("Loading the model (the first run downloads it, about 1.3 GB)")
                cal = task.calibrate(**args, progress=lambda d, n, m: report(m, d, n))
                return {"version": cal["version"]}
        return app.jobs.start("calibrate", work, t)

    @route("GET", "/api/tasks/{t}/report")
    def _report(q, body, t):
        try:
            return app.task(t).report(print_it=False)
        except ZetError as e:
            return {"calibrated": False, "stale": str(e)}

    # examples
    @route("GET", "/api/tasks/{t}/examples")
    def _examples(q, body, t):
        task = app.task(t)
        exs = task.examples()
        offset, limit = int(q.get("offset", 0)), int(q.get("limit", 50))
        counts = {qid: {} for qid in task.questions}
        langs: Dict[str, int] = {}
        for ex in exs:
            for qid, lab in ex["labels"].items():
                counts[qid][lab] = counts[qid].get(lab, 0) + 1
            lang = ex.get("language") or "detected"
            langs[lang] = langs.get(lang, 0) + 1
        rows = list(reversed(exs))[offset:offset + limit]
        return {"total": len(exs), "rows": rows, "label_counts": counts, "languages": langs}

    @route("POST", "/api/tasks/{t}/examples")
    def _add_examples(q, body, t):
        task = app.task(t)
        try:
            with app.lock(t):
                if body.get("csv") is not None:
                    suffix = ".jsonl" if str(body.get("filename", "")).lower().endswith((".jsonl", ".json")) else ".csv"
                    with tempfile.TemporaryDirectory() as d:
                        p = Path(d) / ("upload" + suffix)
                        p.write_text(body["csv"], encoding="utf-8")
                        n = task.add_examples(p)
                else:
                    n = task.add_examples(body.get("rows") or [])
        except (ValueError, OSError, ZetError) as e:
            raise ApiError(str(e).replace("upload.csv", body.get("filename") or "file"))
        return {"added": n, "total": len(task.examples())}

    @route("POST", "/api/tasks/{t}/examples/remove")
    def _remove_examples(q, body, t):
        task = app.task(t)
        with app.lock(t):
            n = (task.restore_examples if body.get("restore") else task.remove_examples)(body.get("ids") or [])
        return {"changed": n, "total": len(task.examples())}

    @route("GET", "/api/tasks/{t}/template.csv")
    def _template(q, body, t):
        task = app.task(t)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["body", *task.questions, "language"])
        example = []
        for qdef in task.questions.values():
            crit = qdef.get("criteria")
            if qdef["type"] == "choice":
                example.append(list(crit)[0] if crit else "")
            elif qdef["type"] == "score":
                example.append(crit[0])
            else:
                example.append("no")
        w.writerow(["Write the message here", *example, "en"])
        return ("text/csv; charset=utf-8", buf.getvalue().encode("utf-8"), f"{t}-template.csv")

    # review
    @route("GET", "/api/tasks/{t}/review")
    def _review(q, body, t):
        task = app.task(t)
        cal, stale = app._calibration(task)
        return {"calibrated": cal is not None, "stale": stale, "queue": task.review_queue() if cal else []}

    @route("GET", "/api/tasks/{t}/history")
    def _history(q, body, t):
        task = app.task(t)
        preds = list(reversed(task.predictions()))
        verdicts = task.verdicts()
        offset, limit = int(q.get("offset", 0)), int(q.get("limit", 50))
        rows = [{**r, "verdicts": verdicts.get(r["id"], {})} for r in preds[offset:offset + limit]]
        return {"total": len(preds), "rows": rows}

    @route("POST", "/api/tasks/{t}/verdict")
    def _verdict(q, body, t):
        task = app.task(t)
        pid, labels = body.get("id"), body.get("labels") or {}
        if not pid or not labels:
            raise ApiError("choose an answer first")
        try:
            with app.lock(t):
                rec = task.prediction(pid)
                same = [qid for qid, v in labels.items() if rec["answers"][qid]["answer"] == v]
                diff = {qid: v for qid, v in labels.items() if qid not in same}
                if same:
                    task.confirm(pid, *same)
                if diff:
                    task.correct(pid, **diff)
        except (ZetError, ValueError, KeyError) as e:
            raise ApiError(str(e))
        return {"ok": True, "confirmed": same, "corrected": list(diff)}

    # settings
    @route("POST", "/api/tasks/{t}/audit-rate")
    def _audit(q, body, t):
        task = app.task(t)
        try:
            task.set_audit_rate(float(body["rate"]), body.get("group") or None)
            if body.get("remove") and body.get("group"):
                task.config.get("audit_rate_by_group", {}).pop(body["group"], None)
                task._save_config()
        except (ZetError, ValueError, KeyError) as e:
            raise ApiError(str(e))
        return app.detail(t)

    @route("POST", "/api/tasks/{t}/backend")
    def _backend(q, body, t):
        task = app.task(t)
        with app.lock(t):
            task.set_backend(body.get("name") or "laya-onnx", body.get("options") or {})
        app.forget(t)
        return app.detail(t)

    @route("POST", "/api/tasks/{t}/delete")
    def _delete(q, body, t):
        task = app.task(t)
        if body.get("confirm") != t:
            raise ApiError("type the task's name to confirm")
        with app.lock(t):
            moved = task.delete()
        app.forget(t)
        return {"deleted": t, "moved_to": str(moved)}

    # jobs and tools
    @route("GET", "/api/jobs/{j}")
    def _job(q, body, j):
        return app.jobs.get(j)

    @route("POST", "/api/tools/check-model")
    def _check(q, body):
        from ..backends import LayaOnnxBackend
        model = body.get("model") or "multilingual"
        english = body.get("english") or None

        def work(report):
            import numpy
            import onnxruntime
            import tokenizers
            from ..smoke import QUESTIONS, STATES
            report("Loading the model (the first run downloads it, about 1.3 GB)")
            backend = LayaOnnxBackend(model=model, english=english)
            t0 = time.perf_counter()
            preds = backend.predict(STATES, QUESTIONS)
            dt = time.perf_counter() - t0
            return {"versions": {"python": sys.version.split()[0], "onnxruntime": onnxruntime.__version__,
                                 "numpy": numpy.__version__, "tokenizers": tokenizers.__version__},
                    "ms_per_decision": round(1000 * dt / (len(STATES) * len(QUESTIONS)), 1),
                    "torch_loaded": "torch" in sys.modules, "questions": QUESTIONS,
                    "results": [{"text": s["body"], "language": p.meta.get("language"),
                                 "checkpoint": p.meta.get("checkpoint"), "probs": p.probs}
                                for s, p in zip(STATES, preds)]}
        return app.jobs.start("check-model", work)

    @route("POST", "/api/tools/benchmark")
    def _bench(q, body):
        if not BENCHMARK.is_file():
            raise ApiError("the benchmark script is only available in a copy of the Zet repository")
        languages = body.get("languages") or ["en-US", "sv-SE"]
        per_language = int(body.get("per_language", 300))
        model = body.get("model") or "multilingual"
        budget = float(body.get("error_budget", 0.05))

        def work(report):
            import datetime as dt
            import importlib.util
            from ..backends import LayaOnnxBackend
            spec = importlib.util.spec_from_file_location("zet_massive", BENCHMARK)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            report("Downloading MASSIVE (about 40 MB, first time only)")
            archive = m.download(Path.home() / ".cache" / "zet" / "massive")
            report("Reading languages")
            data = {loc: m.load_locale(archive, loc) for loc in languages}
            report("Running both benchmark tasks (loads the model first)")
            with tempfile.TemporaryDirectory() as work_dir:
                tasks = m.run(LayaOnnxBackend(model=model), data, per_language, 0, budget, Path(work_dir))
            res = {"zet": __version__, "model": model, "languages": languages, "per_language": per_language,
                   "seed": 0, "error_budget": budget, "date": dt.date.today().isoformat(),
                   "data": "MASSIVE 1.1 dev (CC BY 4.0)", "tasks": tasks}
            out = BENCHMARK.parent
            (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            (out / "results.md").write_text(m.markdown(res), encoding="utf-8")
            return res | {"saved_to": str(out)}
        return app.jobs.start("benchmark", work)

    compiled = []
    for method, pattern, fn in routes:
        parts = pattern.strip("/").split("/")
        compiled.append((method, parts, fn))

    def dispatch(method: str, path: str):
        parts = [unquote(p) for p in path.strip("/").split("/")]
        for m, pat, fn in compiled:
            if m != method or len(pat) != len(parts):
                continue
            args = []
            for a, b in zip(pat, parts):
                if a.startswith("{"):
                    args.append(b)
                elif a != b:
                    break
            else:
                return fn, args
        return None, None

    class Handler(BaseHTTPRequestHandler):
        server_version = f"Zet/{__version__}"

        def log_message(self, fmt, *args):  # quiet console
            pass

        def _send(self, status: int, ctype: str, data: bytes, filename: Optional[str] = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status: int, obj: Any):
            self._send(status, "application/json; charset=utf-8",
                       json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))

        def _handle(self, method: str):
            url = urlparse(self.path)
            if method == "GET" and not url.path.startswith("/api/"):
                page = STATIC / "index.html"
                return self._send(200, "text/html; charset=utf-8", page.read_bytes())
            if method == "POST" and self.headers.get("X-Zet") != "1":
                return self._json(403, {"error": "missing X-Zet header"})
            fn, args = dispatch(method, url.path)
            if fn is None:
                return self._json(404, {"error": f"no such endpoint: {method} {url.path}"})
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    try:
                        body = json.loads(self.rfile.read(n).decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        return self._json(400, {"error": "request body is not JSON"})
            try:
                out = fn(q, body, *args)
            except ApiError as e:
                return self._json(e.status, {"error": str(e)})
            except Exception as e:  # never kill the server; show the message
                traceback.print_exc()
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            if isinstance(out, tuple):
                return self._send(200, out[0], out[1], out[2])
            return self._json(200, out)

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def serve(root: Optional[str] = None, port: int = 8765, open_browser: bool = True,
          app: Optional[App] = None) -> ThreadingHTTPServer:
    app = app or App(Path(root) if root else None)
    handler = make_handler(app)
    server = None
    for p in [port] + list(range(port + 1, port + 20)) + [0]:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), handler)
            break
        except OSError:
            continue
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    server.app = app
    server.url = url
    if open_browser:
        print(f"Zet is running at {url}  (tasks in {app.root}). Close this window to stop it.")
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    return server


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="zet ui", description="Open the Zet interface in your browser.")
    ap.add_argument("--root", default=None, help="folder with task folders (default: $ZET_ROOT or ~/zet)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    server = serve(args.root, args.port, not args.no_browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
