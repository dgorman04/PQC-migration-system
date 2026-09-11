"""Transparent, documented rule-based label function for the quantum-migration
risk model (spec §6, README §7.5).

The label produced here is NOT ground truth — it is a deliberately simple,
fully auditable scoring function used to generate *training* labels at a
scale no manual labelling effort could match. XGBoost learns the non-linear
structure in the noisy version of this signal; the model is then judged
against a small, *separately* hand-labelled validation set
(`data/expert_validation.csv`, README §7.5) which this function never sees
and never influences. That split — and the reasoning below — belongs
verbatim in the dissertation methodology chapter; this file is its source of
truth, so keep the two in sync if the weights change.

Five weighted factors:
  exposure      — is this asset reachable, and does it present a cert at all
  criticality   — the business-declared importance of the asset
  algo_weakness — how quantum-broken the observed algorithm/key actually is
  complexity    — harder-to-migrate assets are riskier to leave exposed for
                  longer, so complexity raises the label (this is a
                  deliberate, debatable modelling choice — defend it, or
                  change the weight to 0 and document why instead)
  business      — what kind of data / regulation sits behind the asset

Plus a small bump for a documented prior security incident, and Gaussian
noise so the label is not a deterministic function of the inputs — otherwise
XGBoost has nothing to learn beyond memorising this formula.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLabelWeights:
    exposure: float = 0.25
    criticality: float = 0.30
    algo_weakness: float = 0.25
    complexity: float = 0.10
    business: float = 0.10
    incident_bump: float = 0.05
    noise_std: float = 0.05

    def __post_init__(self) -> None:
        core = self.exposure + self.criticality + self.algo_weakness + self.complexity + self.business
        if abs(core - 1.0) > 1e-6:
            raise ValueError(
                "exposure + criticality + algo_weakness + complexity + business "
                f"must sum to 1.0, got {core}"
            )


_CRITICALITY_SCORE = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.15}
_COMPLEXITY_SCORE = {"HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2}


def _exposure(row: pd.Series) -> float:
    return 0.6 * float(row["internet_accessible"]) + 0.4 * float(row["has_public_cert"])


def _algo_weakness(row: pd.Series) -> float:
    """1.0 for anything quantum-vulnerable (Shor's algorithm doesn't care how
    large the key is), a small residual for non-vulnerable algorithms (other
    classical weaknesses may still exist and are out of scope for this
    model), 0.0-1.0 either way.
    """
    if not bool(row["quantum_vulnerable"]):
        return 0.1
    return 1.0


def _business(row: pd.Series) -> float:
    return (
        0.5 * float(row["financial_data"])
        + 0.3 * float(row["customer_data"])
        + 0.2 * float(row["regulated_system"])
    )


def compute_risk_label(row: pd.Series, weights: RiskLabelWeights = RiskLabelWeights()) -> float:
    """The label for one asset's feature row, in [0, 1], before noise."""
    criticality = _CRITICALITY_SCORE.get(str(row["criticality"]).upper(), 0.4)
    complexity = _COMPLEXITY_SCORE.get(str(row["migration_complexity"]).upper(), 0.5)

    score = (
        weights.exposure * _exposure(row)
        + weights.criticality * criticality
        + weights.algo_weakness * _algo_weakness(row)
        + weights.complexity * complexity
        + weights.business * _business(row)
    )
    if bool(row.get("previous_incident_flag", False)):
        score += weights.incident_bump
    return float(np.clip(score, 0.0, 1.0))


def label_dataframe(
    df: pd.DataFrame,
    weights: RiskLabelWeights = RiskLabelWeights(),
    seed: int = 42,
    label_col: str = "risk_score",
) -> pd.DataFrame:
    """Returns a copy of `df` with `label_col` added. Deterministic given `seed`
    — re-running with the same seed reproduces the same labels, which matters
    for the dissertation's reproducibility claims (README §7.13).
    """
    rng = np.random.default_rng(seed)
    base = df.apply(lambda row: compute_risk_label(row, weights), axis=1)
    noise = rng.normal(loc=0.0, scale=weights.noise_std, size=len(df))
    out = df.copy()
    out[label_col] = np.clip(base.to_numpy() + noise, 0.0, 1.0)
    return out
