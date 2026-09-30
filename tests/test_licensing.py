"""Licensing is correct from the first commit: every copied file carries both header lines and is
listed in THIRD_PARTY_NOTICES.md, which holds the full license text of each source."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPIED = ["zet/_laya/sequence.py", "zet/_laya/questions.py", "zet/_laya/collate.py", "zet/_laya/lang.py"]


def test_copied_files_carry_both_headers():
    for rel in COPIED:
        head = (ROOT / rel).read_text(encoding="utf-8").splitlines()[:4]
        text = "\n".join(head)
        assert "Copyright the Laya authors" in text, rel
        assert "Modified from Laya (Apache-2.0), https://github.com/NandhaKishorM/laya, by the Zet authors" in text, rel


def test_every_copied_file_is_listed_with_full_license():
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for rel in COPIED:
        assert f"`{rel}`" in notices, rel
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in notices
    assert "END OF TERMS AND CONDITIONS" in notices


def test_no_other_file_claims_laya_origin_without_being_listed():
    for p in (ROOT / "zet").rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        if "Modified from Laya" in p.read_text(encoding="utf-8"):
            assert rel in COPIED, f"{rel} says it is copied from Laya but is not in COPIED/notices"


def test_license_at_root_is_apache():
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")[:200]
