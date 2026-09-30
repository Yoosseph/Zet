"""What `Task.predict` returns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional


@dataclass
class Answer:
    """One question's answer.

    answer:  the most probable option
    status:  "sure" (its error rate is within the task's budget) or "unsure" (send to a human)
    options: the prediction set: options that together contain the truth at the calibrated rate
    probs:   the model's probability for every option
    audit:   True when this sure answer was sampled for a human spot check
    calibration_group: which calibration was applied ("sv", "sv/billing", "pooled", ...)
    """
    question: str
    answer: str
    status: str
    options: List[str]
    probs: Dict[str, float]
    audit: bool = False
    calibration_group: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def sure(self) -> bool:
        return self.status == "sure"

    def to_dict(self) -> Dict[str, Any]:
        d = {"answer": self.answer, "status": self.status, "options": self.options,
             "probs": self.probs, "audit": self.audit, "calibration_group": self.calibration_group}
        if self.warnings:
            d["warnings"] = self.warnings
        return d


@dataclass
class Result(Mapping):
    """All answers for one state; index it by question key: result["department"].answer."""
    id: str
    answers: Dict[str, Answer]
    language: str
    calibrated: bool
    version: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.answers)

    def __len__(self) -> int:
        return len(self.answers)

    @property
    def all_sure(self) -> bool:
        return all(a.sure for a in self.answers.values())

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "language": self.language, "calibrated": self.calibrated,
                "version": self.version, "answers": {q: a.to_dict() for q, a in self.answers.items()},
                "warnings": self.warnings}
