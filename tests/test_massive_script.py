"""The MASSIVE benchmark script, offline: a small fake archive in MASSIVE's layout + FakeBackend."""
from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

from zet import FakeBackend

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("massive_run", ROOT / "benchmarks" / "massive" / "run.py")
massive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(massive)


def fake_archive(tmp_path: Path) -> Path:
    path = tmp_path / "amazon-massive-dataset-1.1.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for loc in ("en-US", "sv-SE"):
            rows = []
            for i in range(900):
                scen = sorted(massive.SCENARIOS)[i % 18]
                rows.append({"id": str(i), "locale": loc, "partition": "dev" if i % 7 else "test",
                             "scenario": scen, "intent": f"{scen}_x", "utt": f"{loc} utterance {i}"})
            blob = "\n".join(json.dumps(r) for r in rows).encode()
            info = tarfile.TarInfo(f"1.1/data/{loc}.jsonl")
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    return path


def test_loads_dev_only_and_samples_parallel_ids(tmp_path):
    arch = fake_archive(tmp_path)
    en = massive.load_locale(arch, "en-US")
    ids = massive.sample_ids(en, 100, seed=0)
    assert len(ids) == 100 and all(int(i) % 7 for i in ids)          # dev partition only
    assert massive.sample_ids(en, 100, seed=0) == ids                # fixed seed
    six = massive.sample_ids(en, 1000, 0, massive.SUBSET6)
    assert {en[int(i)]["scenario"] for i in six} == set(massive.SUBSET6)


def test_run_and_markdown(tmp_path):
    arch = fake_archive(tmp_path)
    data = {loc: massive.load_locale(arch, loc) for loc in ("en-US", "sv-SE")}
    tasks = massive.run(FakeBackend(seed=2), data, 200, 0, 0.1, tmp_path / "work")
    assert tasks["scenario18"]["options"] == 18 and tasks["scenario6"]["options"] == 6
    groups = {g["group"] for g in tasks["scenario18"]["report"]["groups"]}
    assert groups == {"en-US", "sv-SE"}
    res = {"model": "fake", "error_budget": 0.1, "seed": 0, "per_language": 200, "date": "2026-09-29",
           "tasks": tasks}
    md = massive.markdown(res)
    assert "| sv-SE |" in md and "CC BY 4.0" in md and "## scenario6 (6 options; 200 sampled per language)" in md
    assert "Laya-only wrong" in md and "Laya wrong in review" in md
