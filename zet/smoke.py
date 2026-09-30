"""End-to-end smoke run: real Laya weights, ONNX Runtime, no PyTorch.

    python -m zet.smoke                       # multilingual checkpoint
    python -m zet.smoke --model english
    python -m zet.smoke --model C:\\models\\laya-split   # a local directory

Downloads the weights on first run (english ~1.7 GB, multilingual ~1.3 GB), answers English and
Swedish emails for all three question types, and writes a JSON report with versions, timings,
the Hub revision and SHA-256 of every weight file. Those last two are the values to pin in
zet/backends/weights.py once the run looks right.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this email?",
                   "criteria": {"billing": "invoices, payments, charges",
                                "technical": "bugs, crashes, login problems",
                                "sales": "buying, upgrading, pricing", "other": "anything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this email?",
                "criteria": ["not urgent", "soon", "critical"]},
    "refund_requested": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
}
STATES = [
    {"body": "Could you resend last month's invoice? No rush."},
    {"body": "The app crashes every time I log in and I have a demo in an hour!"},
    {"body": "I don't want a refund, I just want the app to work."},
    {"body": "Kan ni skicka om förra månadens faktura? Inte bråttom."},
    {"body": "Appen kraschar varje gång jag loggar in och jag har en demo om en timme!"},
    {"body": "Jag vill inte ha pengarna tillbaka, bara en fungerande app."},
]


def _versions() -> dict:
    import numpy
    import onnxruntime
    import tokenizers

    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "onnxruntime": onnxruntime.__version__, "numpy": numpy.__version__,
            "tokenizers": tokenizers.__version__}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m zet.smoke", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="multilingual", help="english, multilingual, or a local directory")
    ap.add_argument("--revision", default=None, help="Hub revision to download")
    ap.add_argument("--out", default=None, help="report path (default: zet-smoke-<model>.json)")
    args = ap.parse_args(argv)

    from . import __version__
    from .backends import LayaOnnxBackend
    from .backends.weights import sha256_file

    print("Zet", __version__, json.dumps(_versions()))
    backend = LayaOnnxBackend(model=args.model, revision=args.revision)

    t0 = time.perf_counter()
    ck = backend._checkpoint(backend.checkpoint_for(STATES[0]) if args.model == "auto" else args.model)
    t_load = time.perf_counter() - t0
    print(f"loaded {ck.name} ({ck.format}) from {ck.dir} in {t_load:.1f}s; revision {ck.revision}")

    t1 = time.perf_counter()
    preds = backend.predict(STATES, QUESTIONS)
    t_pred = time.perf_counter() - t1

    rows = []
    for st, pr in zip(STATES, preds):
        top = {q: max(p, key=p.get) for q, p in pr.probs.items()}
        print(f"\n[{pr.meta['language']}] {st['body']}")
        for q, p in pr.probs.items():
            print(f"  {q:17s} -> {top[q]:10s} " + "  ".join(f"{k}={v:.3f}" for k, v in p.items()))
        rows.append({"state": st, "probs": pr.probs, "meta": pr.meta})

    torch_loaded = "torch" in sys.modules
    files = sorted(p for p in Path(ck.dir).rglob("*") if p.is_file())
    report = {
        "zet": __version__, "versions": _versions(), "model": args.model, "checkpoint": ck.name,
        "format": ck.format, "revision": ck.revision,
        "sha256": {str(p.relative_to(ck.dir)).replace("\\", "/"): sha256_file(p) for p in files},
        "load_seconds": round(t_load, 2), "predict_seconds": round(t_pred, 3),
        "rows": len(STATES) * len(QUESTIONS), "torch_imported": torch_loaded, "results": rows,
    }
    out = Path(args.out or f"zet-smoke-{Path(args.model).name}.json")
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    ms = 1000 * t_pred / report["rows"]
    print(f"\n{report['rows']} rows in {t_pred:.2f}s ({ms:.0f} ms/row); torch imported: {torch_loaded}")
    print(f"report: {out.resolve()}")
    if torch_loaded:
        print("FAIL: torch was imported; the core must run without it.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
