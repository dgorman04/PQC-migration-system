"""Static score → priority bucket mapping. Loads
`python/config/risk_thresholds.yaml` (README §7.5). Shared by evaluation and,
eventually, the migration optimiser (§7.11) and the inference service's
response bucketing (§7.6) — the config file is the one source of truth for
all three.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PRIORITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "risk_thresholds.yaml"


@dataclass(frozen=True)
class ThresholdTable:
    # ordered highest-priority first: (priority, min_score)
    buckets: list[tuple[str, float]]

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_PATH) -> "ThresholdTable":
        data = yaml.safe_load(Path(path).read_text())
        buckets = [(b["priority"], float(b["min_score"])) for b in data["buckets"]]
        scores = [score for _, score in buckets]
        if scores != sorted(scores, reverse=True):
            raise ValueError(
                f"{path}: buckets must be listed highest min_score first so "
                "'first match wins' is correct"
            )
        return cls(buckets=buckets)

    def priority_for(self, score: float) -> str:
        for priority, min_score in self.buckets:
            if score >= min_score:
                return priority
        # Every table should end in a bucket with min_score 0.0; this is a
        # defensive fallback in case a caller loads a malformed table.
        return self.buckets[-1][0]

    def rank(self, priority: str) -> int:
        """0 = most urgent. Used to tell an "under-scored" miss (the model
        said less urgent than reality) from an "over-scored" one.
        """
        for index, (name, _) in enumerate(self.buckets):
            if name == priority:
                return index
        raise ValueError(f"unknown priority {priority!r}, expected one of {[b[0] for b in self.buckets]}")
