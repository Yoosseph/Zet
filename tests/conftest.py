"""Tests marked @pytest.mark.model need real Laya weights and run only with --run-model
(or ZET_RUN_MODEL=1). Everything else runs anywhere, offline, with no model download."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "tiny_model"

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": ["billing", "technical", "sales", "other"]},
    "urgency": {"type": "score", "instructions": "How urgent is it?",
                "criteria": ["not urgent", "soon", "critical"]},
    "refund_requested": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
}

STATES = [
    {"body": "Could you resend last month's invoice? No rush."},
    {"body": "Kan ni skicka om förra månadens faktura? Inte bråttom."},
    "The app crashes on login. I don't want a refund, just a fix.",
    ["Hi, my invoice is wrong.", "Please fix it today."],
]


def pytest_addoption(parser):
    parser.addoption("--run-model", action="store_true", default=False,
                     help="run tests that download and run real Laya weights")


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs real Laya weights (enable with --run-model)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-model") or os.environ.get("ZET_RUN_MODEL") == "1":
        return
    skip = pytest.mark.skip(reason="needs real Laya weights; run with --run-model")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
