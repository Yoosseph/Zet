"""MASSIVE benchmark for Zet 0.1: how Laya + Zet calibration behave across languages.

    python benchmarks/massive/run.py                                  # en-US and sv-SE, 300 each
    python benchmarks/massive/run.py --languages en-US sv-SE de-DE fr-FR --per-language 300

Data: MASSIVE 1.1 (Amazon, CC BY 4.0), derived from SLURP (CC BY 4.0). Downloaded at run time
from Amazon's S3 bucket and cached; never committed. Uses the `dev` partition only: Laya chose
its language routing on `test` (docs/decisions.md, Q16).

The same utterance ids are sampled in every language (MASSIVE is parallel), so differences
between languages come from the language, not from the content.

Two questions, each its own task:
- scenario18: all 18 scenarios (Laya's weakest case: many options)
- scenario6: 6 clearly distinct scenarios, on utterances from those scenarios only

Writes benchmarks/massive/results.json and results.md. Numbers are Zet's held-out numbers; the
labels are MASSIVE's human labels.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"

SCENARIOS = {
    "alarm": "setting, checking or removing alarms", "audio": "volume and audio settings",
    "calendar": "calendar events and reminders", "cooking": "recipes and cooking",
    "datetime": "the date, time or time zones", "email": "reading, writing or managing email",
    "general": "chit-chat, jokes, repeats, confirmations", "iot": "smart-home devices, lights, cleaning robots",
    "lists": "to-do and shopping lists", "music": "music preferences and information about music",
    "news": "news", "play": "playing music, podcasts, radio, audiobooks or games",
    "qa": "factual questions, definitions, maths, stock prices", "recommendation": "recommendations for places, movies, events",
    "social": "social media posts and complaints", "takeaway": "ordering food or takeaway",
    "transport": "trains, taxis, tickets and traffic", "weather": "the weather",
}
SUBSET6 = ["alarm", "cooking", "email", "news", "transport", "weather"]


def questions(names: Sequence[str]) -> Dict[str, Dict]:
    return {"scenario": {"type": "choice", "instructions": "What is this voice-assistant request about?",
                         "criteria": {n: SCENARIOS[n] for n in names}}}


def download(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / URL.rsplit("/", 1)[1]
    if not path.is_file():
        print(f"downloading {URL} ...", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(path)
    return path


def load_locale(archive: Path, locale: str) -> List[Dict]:
    with tarfile.open(archive, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(f"/{locale}.jsonl")), None)
        if member is None:
            raise SystemExit(f"{locale} not in {archive.name}")
        lines = tar.extractfile(member).read().decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def sample_ids(rows: List[Dict], n: int, seed: int, scenarios: Optional[Sequence[str]] = None) -> List[str]:
    ids = sorted(r["id"] for r in rows if r["partition"] == "dev"
                 and (scenarios is None or r["scenario"] in scenarios))
    return random.Random(seed).sample(ids, min(n, len(ids)))


def run(backend, data: Dict[str, List[Dict]], per_language: int, seed: int, error_budget: float,
        workdir: Path) -> Dict:
    from zet import Task

    first = next(iter(data.values()))
    by_id = {loc: {r["id"]: r for r in rows} for loc, rows in data.items()}
    out = {}
    for name, names in (("scenario18", sorted(SCENARIOS)), ("scenario6", SUBSET6)):
        ids = sample_ids(first, per_language, seed, None if name == "scenario18" else names)
        ids = [i for i in ids if all(i in by_id[loc] for loc in data)]  # parallel across languages
        task = Task.create(name, questions(names), backend=backend, root=workdir, overwrite=True)
        task.add_examples([{"state": {"body": by_id[loc][i]["utt"]}, "labels": {"scenario": by_id[loc][i]["scenario"]},
                            "language": loc} for loc in data for i in ids])
        task.calibrate(error_budget=error_budget, seed=seed)
        rep = task.report(print_it=False)
        out[name] = {"options": len(names), "per_language": len(ids), "report": rep["questions"]["scenario"],
                     "split": rep["split"]}
    return out


def markdown(res: Dict) -> str:
    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}%"

    def size(x):
        return "-" if x is None else f"{x:.2f}"

    lines = ["# MASSIVE benchmark (Zet 0.1)", "",
             f"Model: `{res['model']}` · error budget {res['error_budget']:.0%} · seed {res['seed']} · "
             f"MASSIVE 1.1 `dev`, up to {res['per_language']} parallel utterances per language · {res['date']}", "",
             "Every number is measured on the held-out 30% of the sample. Labels are MASSIVE's human labels.",
             "The benchmark demonstrates the method; Zet's guarantee for your task comes from your own "
             "calibration data.", ""]
    for name, r in res["tasks"].items():
        lines += [f"## {name} ({r['options']} options; {r['per_language']} sampled per language)", "",
                  "| language | held out | coverage | avg set size | automated | error among sure | upper bound (95%) |",
                  "|---|---|---|---|---|---|---|"]
        for row in r["report"]["groups"] + [r["report"]["overall"]]:
            lines.append(f"| {row['group']} | {row['n_test']} | {pct(row['coverage'])} | "
                         f"{size(row['avg_set_size'])} | "
                         f"{pct(row['automation'])} | {pct(row['sure_error'])} | {pct(row['sure_error_upper'])} |")
        lines += ["", "Laya-only means using every top answer from the same checkpoint. Zet keeps that answer "
                  "when sure and sends unsure answers to review.", "",
                  "| language | Laya-only wrong | Zet wrong automatically | sent to review | Laya wrong in review |",
                  "|---|---:|---:|---:|---:|"]
        for row in r["report"]["groups"] + [r["report"]["overall"]]:
            reviewed = row["n_test"] - row["n_sure"]
            automatic_errors = (f"{row['sure_errors']}/{row['n_sure']}" if row["n_sure"]
                                else "0 (no automatic answers)")
            lines.append(f"| {row['group']} | {row['laya_errors']}/{row['n_test']} | "
                         f"{automatic_errors} | {reviewed} | "
                         f"{row['review_errors']}/{reviewed} |")
        lines += ["", "The model errors in review are mistakes a correct human review could catch; "
                  "they are not measured human corrections.", ""]
    lines += ["Data: MASSIVE (FitzGerald et al., 2022), CC BY 4.0, derived from SLURP (Bastianelli et al., "
              "2020), CC BY 4.0."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", nargs="+", default=["en-US", "sv-SE"])
    ap.add_argument("--per-language", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="multilingual", help="multilingual, english or a local folder")
    ap.add_argument("--error-budget", type=float, default=0.05)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "zet" / "massive")
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args(argv)

    from zet import LayaOnnxBackend, __version__

    archive = download(args.cache)
    data = {loc: load_locale(archive, loc) for loc in args.languages}
    backend = LayaOnnxBackend(model=args.model)
    with tempfile.TemporaryDirectory() as work:
        tasks = run(backend, data, args.per_language, args.seed, args.error_budget, Path(work))
    res = {"zet": __version__, "model": args.model, "languages": args.languages,
           "per_language": args.per_language, "seed": args.seed, "error_budget": args.error_budget,
           "date": dt.date.today().isoformat(), "data": "MASSIVE 1.1 dev (CC BY 4.0)", "tasks": tasks}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.out / "results.md").write_text(markdown(res), encoding="utf-8")
    print(markdown(res))


if __name__ == "__main__":
    sys.exit(main())
