"""Record Laya's PyTorch outputs for the equivalence test. Run where PyTorch works (WSL):

    pip install laya
    python scripts/make_laya_reference.py --checkpoint multilingual --revision <laya commit>
    python scripts/make_laya_reference.py --checkpoint english --revision <laya commit>

Use the Laya revision the ONNX weights were exported from when you know it; otherwise `main`
(the script records which commit that resolved to). Runs on Windows too if PyTorch installs; use
a separate venv so Zet's own environment stays torch-free.

Writes tests/fixtures/equivalence/reference-<checkpoint>.json with, for every case and question:
Laya's token row (ids and marker positions, built by Laya's own `build_sequence` and tokenizer),
and Laya's answer from `Agent.system_one` (probabilities rounded to 4 decimals by Laya). Commit it;
`pytest --run-model` then checks Zet against it on any machine, without PyTorch.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "equivalence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=["english", "multilingual"], required=True)
    ap.add_argument("--revision", required=True, help="convaiinnovations/laya commit SHA")
    ap.add_argument("--repo", default="convaiinnovations/laya")
    args = ap.parse_args()

    import laya
    import torch
    from laya.agent import Agent
    from laya.common import build_sequence

    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    questions = cases["questions"]
    subfolder = "multilingual" if args.checkpoint == "multilingual" else None
    agent = Agent(args.repo, device="cpu", subfolder=subfolder, revision=args.revision)
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)

    out = []
    for state in cases["states"]:
        answers = agent.system_one(state, questions)["answers"]
        rows = {}
        for qid, qdef in questions.items():
            ids, markers = build_sequence(agent.tok, state, Agent._to_internal(qdef), max_len, head_max_len,
                                          truncate_left=isinstance(state, list))
            rows[qid] = {"ids": ids, "markers": markers}
        out.append({"state": state, "rows": rows, "answers": answers})

    # Record the commit the revision resolved to, so "main" today stays reproducible later.
    from huggingface_hub import HfApi
    resolved = HfApi().model_info(args.repo, revision=args.revision).sha
    ref = {"checkpoint": args.checkpoint, "repo": args.repo, "revision": resolved,
           "revision_requested": args.revision,
           "laya": getattr(laya, "__version__", "unknown"), "torch": torch.__version__,
           "platform": platform.platform(), "questions": questions, "cases": out}
    path = HERE / f"reference-{args.checkpoint}.json"
    path.write_text(json.dumps(ref, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
