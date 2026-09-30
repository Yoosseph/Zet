"""The `zet` command line. Every command works on a task folder under --root (default: ./zet)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
import yaml

from . import __version__
from .task import Task, ZetError

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode=None,
                  help="Zet: typed decisions over text that know when to ask a human.")

ROOT = typer.Option(None, "--root", help="Folder that holds task folders (default: $ZET_ROOT or ~/zet).")


def _fail(msg: str) -> None:
    typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(1)


def _load(name: str, root: str) -> Task:
    try:
        return Task.load(name, root=root)
    except ZetError as e:
        _fail(str(e))


def _utf8_stdout() -> None:
    # Windows consoles default to a legacy code page; Swedish text must print intact.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


@app.callback()
def _main(version: bool = typer.Option(False, "--version", help="Print Zet's version and exit.")):
    _utf8_stdout()
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def init(name: str = typer.Argument(..., help="Task name, e.g. support."),
         questions: Path = typer.Option(..., "--questions", "-q", help="YAML or JSON file with the questions."),
         backend: str = typer.Option("laya-onnx", help="laya-onnx, laya-http or fake."),
         model: Optional[str] = typer.Option(None, help="laya-onnx: auto, multilingual, english or a folder."),
         url: Optional[str] = typer.Option(None, help="laya-http: the Laya server URL."),
         root: str = ROOT, overwrite: bool = typer.Option(False, help="Replace an existing task.")):
    """Create a task.

    Example:  zet init support --questions questions.yaml --model multilingual
    """
    try:
        qs = yaml.safe_load(questions.read_text(encoding="utf-8"))
    except OSError as e:
        _fail(f"cannot read {questions}: {e}")
    opts = {}
    if model:
        opts["model"] = model
    if url:
        opts["url"] = url
    try:
        t = Task.create(name, qs, backend=backend, backend_options=opts, root=root, overwrite=overwrite)
    except (ZetError, ValueError) as e:
        _fail(str(e))
    typer.echo(f"created task {name!r} at {t.path} with {len(qs)} questions")


@app.command("add-examples")
def add_examples(name: str = typer.Argument(..., help="Task name."),
                 file: Path = typer.Argument(..., help="CSV (body, question columns, optional language) or JSONL."),
                 root: str = ROOT):
    """Add labeled examples.

    Example:  zet add-examples support labeled.csv
    """
    t = _load(name, root)
    try:
        n = t.add_examples(file)
    except (ValueError, OSError) as e:
        _fail(str(e))
    typer.echo(f"added {n} examples ({len(t.examples())} in total)")


@app.command()
def calibrate(name: str = typer.Argument(..., help="Task name."),
              error_budget: float = typer.Option(0.05, help="Max error rate among sure answers."),
              alpha: Optional[float] = typer.Option(None, help="Prediction-set miss rate; default: the budget."),
              test_fraction: float = typer.Option(0.3, help="Share held out for the reported numbers."),
              seed: int = typer.Option(0, help="Split seed."), root: str = ROOT):
    """Calibrate on the labeled examples and print the report.

    Example:  zet calibrate support --error-budget 0.05
    """
    t = _load(name, root)
    try:
        t.calibrate(error_budget=error_budget, alpha=alpha, test_fraction=test_fraction, seed=seed)
    except (ZetError, ValueError, RuntimeError, FileNotFoundError) as e:
        _fail(str(e))
    t.report()


@app.command()
def predict(name: str = typer.Argument(..., help="Task name."),
            text: Optional[str] = typer.Argument(None, help="The text to decide on."),
            file: Optional[Path] = typer.Option(None, "--file", "-f", help="JSONL of states, one per line."),
            language: Optional[str] = typer.Option(None, help="The text's language, if known (e.g. sv)."),
            as_json: bool = typer.Option(False, "--json", help="Print JSON lines instead of a summary."),
            root: str = ROOT):
    """Answer the task's questions for a text (or a JSONL file of states).

    Example:  zet predict support "Kan ni skicka om fakturan? Inte bråttom." --language sv
    """
    t = _load(name, root)
    if (text is None) == (file is None):
        _fail("give a text or --file, not both")
    if file is not None:
        states = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        states = [{"body": text, **({"language": language} if language else {})}]
    try:
        results = t.predict(states)
    except (ZetError, RuntimeError, FileNotFoundError) as e:
        _fail(str(e))
    for r in results:
        if as_json:
            typer.echo(json.dumps(r.to_dict(), ensure_ascii=False))
            continue
        typer.echo(f"{r.id}  [{r.language}]")
        for qid, a in r.answers.items():
            opts = "" if a.options == [a.answer] else f"  options: {', '.join(a.options)}"
            audit = "  AUDIT: please check" if a.audit else ""
            typer.echo(f"  {qid:18s} {a.answer:12s} {a.status:6s} p={a.probs[a.answer]:.2f}{opts}{audit}")
        for w in r.warnings:
            typer.echo(f"  ! {w}")


@app.command()
def correct(name: str = typer.Argument(..., help="Task name."),
            prediction_id: str = typer.Argument(..., help="The prediction id printed by zet predict."),
            labels: List[str] = typer.Argument(..., help="question=label pairs."),
            root: str = ROOT):
    """Record the true answer(s) for a prediction.

    Example:  zet correct support p1a2b3c4d5e6 department=sales urgency=0
    """
    t = _load(name, root)
    parsed = {}
    for item in labels:
        if "=" not in item:
            _fail(f"{item!r} is not question=label")
        k, v = item.split("=", 1)
        parsed[k.strip()] = v.strip()
    try:
        t.correct(prediction_id, **parsed)
    except (ZetError, ValueError) as e:
        _fail(str(e))
    typer.echo(f"recorded correction for {prediction_id}")


@app.command()
def confirm(name: str = typer.Argument(..., help="Task name."),
            prediction_id: str = typer.Argument(..., help="The prediction id printed by zet predict."),
            root: str = ROOT):
    """Record that a prediction's answers were right (counts for audits and drift).

    Example:  zet confirm support p1a2b3c4d5e6
    """
    t = _load(name, root)
    try:
        t.confirm(prediction_id)
    except ZetError as e:
        _fail(str(e))
    typer.echo(f"confirmed {prediction_id}")


@app.command()
def report(name: str = typer.Argument(..., help="Task name."),
           as_json: bool = typer.Option(False, "--json", help="Print the numbers as JSON."),
           root: str = ROOT):
    """Show held-out numbers per question and language: coverage, automation, sure-answer error.

    Example:  zet report support
    """
    t = _load(name, root)
    try:
        r = t.report(print_it=not as_json)
    except ZetError as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps(r, ensure_ascii=False, indent=1))


@app.command()
def ui(root: str = ROOT, port: int = typer.Option(8765, help="Port on this computer."),
       no_browser: bool = typer.Option(False, "--no-browser", help="Don't open a browser window.")):
    """Open the Zet interface in your browser. Everything Zet does can be done there.

    Example:  zet ui
    """
    from .ui.server import serve
    server = serve(root, port, not no_browser)
    if no_browser:
        typer.echo(f"Zet is running at {server.url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    app()


if __name__ == "__main__":
    main()
