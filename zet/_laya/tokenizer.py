"""Torch-free tokenizer for Laya checkpoints, built on the `tokenizers` library.

Laya loads its tokenizer with `transformers.AutoTokenizer` (laya/onnx_agent.py:151). Zet reads the
same `tokenizer.json` directly. Parity of token ids with Laya is checked by the model tests
(`tests/test_laya_equivalence.py`), which compare whole input rows built by both.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer

# Fallback names, tried in order, when tokenizer_config.json does not name a special token.
_FALLBACKS = {
    "cls_token": ("[CLS]", "<cls>", "<s>", "<bos>"),
    "sep_token": ("[SEP]", "<sep>", "</s>", "<eos>"),
    "mask_token": ("[MASK]", "<mask>"),
    "pad_token": ("[PAD]", "<pad>"),
}


def _find(model_dir: Path, name: str) -> Optional[Path]:
    for p in (model_dir / name, model_dir / "tokenizer" / name):
        if p.is_file():
            return p
    return None


def _token_str(value) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


class LayaTokenizer:
    """The subset of a tokenizer that Laya's sequence builder uses."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        tok_path = _find(model_dir, "tokenizer.json")
        if tok_path is None:
            raise FileNotFoundError(f"no tokenizer.json in {model_dir} or {model_dir / 'tokenizer'}")
        self._tok = Tokenizer.from_file(str(tok_path))
        # A tokenizer.json may carry truncation or padding settings; Laya never applies them.
        self._tok.no_truncation()
        self._tok.no_padding()

        cfg = {}
        cfg_path = _find(model_dir, "tokenizer_config.json")
        if cfg_path is not None:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        for attr, fallbacks in _FALLBACKS.items():
            candidates = [_token_str(cfg.get(attr))] + list(fallbacks)
            for cand in candidates:
                if cand is not None and self._tok.token_to_id(cand) is not None:
                    setattr(self, attr, cand)
                    setattr(self, attr + "_id", self._tok.token_to_id(cand))
                    break
            else:
                raise ValueError(f"tokenizer in {model_dir} has no usable {attr}")

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Token ids without special tokens; `max_length` keeps the first ids, as HF truncation does."""
        ids = self._tok.encode(text, add_special_tokens=False).ids
        return ids[:max_length] if max_length is not None else ids
