"""Where Laya's ONNX weights come from, and how they are pinned.

Hosted defaults (decision F1, open: see docs/decisions.md):
- multilingual: `soyelmismo/laya-multilingual-onnx`, the full-precision `model-fp32.onnx`, pinned
  to commit 0966c4f. Its top answers match Laya's PyTorch path on every equivalence case.
- english: no full-precision ONNX export is hosted anywhere since `distinctinteractive/laya-onnx`
  went offline (2026-09-29). Export one yourself and pass its folder as `model=`.

Any local folder works too: `model.onnx` (or `model-fp32.onnx`) for the single-graph format,
`encoder.onnx` + `head.onnx` for the split format, plus `rl_agent_config.json` and the tokenizer.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class WeightSource:
    repo: str
    subfolder: str               # "" for the repo root
    revision: str                # a commit SHA; a Hub commit is content-addressed, so this pins the files
    files: Tuple[str, ...]       # patterns to download, relative to the subfolder
    sha256: Dict[str, str] = field(default_factory=dict)  # optional extra check: file name -> digest


SOURCES: Dict[str, WeightSource] = {
    "multilingual": WeightSource(
        "soyelmismo/laya-multilingual-onnx", "", "0966c4fa58da6878b39e7e14cb5e93313b82d828",
        ("model-fp32.onnx", "tokenizer.json", "rl_agent_config.json", "tokenizer/*")),
}

NO_SOURCE = {
    "english": (
        "No hosted full-precision ONNX export of Laya's English checkpoint exists (the previous one, "
        "distinctinteractive/laya-onnx, went offline). Export it with Laya's "
        "laya-ts/scripts/export_onnx.py and pass the folder: LayaOnnxBackend(model=r'C:\\path\\to\\folder'), "
        "or use model='multilingual'."),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(model_dir: Path, sha256: Dict[str, str]) -> None:
    for name, want in sha256.items():
        p = model_dir / name
        if not p.is_file():
            raise FileNotFoundError(f"pinned file {name} is missing from {model_dir}")
        got = sha256_file(p)
        if got != want.lower():
            raise ValueError(f"{name}: SHA-256 {got} does not match the pinned {want}")


def resolve(model: str, revision: Optional[str] = None, cache_dir: Optional[str] = None,
            sha256: Optional[Dict[str, str]] = None) -> Path:
    """A local directory holding one checkpoint's ONNX files.

    `model` is "english", "multilingual" or a local directory path.
    """
    local = Path(os.path.expandvars(model)).expanduser()
    if local.is_dir():
        if sha256:
            verify(local, sha256)
        return local
    if model in NO_SOURCE:
        raise FileNotFoundError(NO_SOURCE[model])
    if model not in SOURCES:
        raise ValueError(f"unknown model {model!r}: use 'multilingual', 'english' with a local folder, "
                         "or a local directory path")
    src = SOURCES[model]
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode
    from huggingface_hub import snapshot_download

    prefix = f"{src.subfolder}/" if src.subfolder else ""
    try:
        root = snapshot_download(src.repo, revision=revision or src.revision, cache_dir=cache_dir,
                                 allow_patterns=[prefix + f for f in src.files])
    except Exception as e:  # network down, repo gone, rate limit: say what to do instead
        raise RuntimeError(
            f"could not download Laya weights for {model!r} from {src.repo}@{(revision or src.revision)[:7]}: "
            f"{type(e).__name__}: {e}. If the repo is gone, point Zet at a local export: "
            f"LayaOnnxBackend(model=r'C:\\path\\to\\folder').") from e
    path = Path(root) / src.subfolder if src.subfolder else Path(root)
    pins = sha256 if sha256 is not None else src.sha256
    if pins:
        verify(path, pins)
    return path


def resolved_revision(model_dir: Path) -> Optional[str]:
    """The Hub commit a snapshot directory belongs to (…/snapshots/<sha>[/<subfolder>]), if any."""
    parts = Path(model_dir).absolute().parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None
