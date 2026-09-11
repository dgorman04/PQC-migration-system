"""Unit tests for pqc_riskmodel.labels — the rule-based label function
(README §7.5). Sanity checks only: this function's job is to be transparent
and auditable, not clever, so the tests mostly confirm "worse asset scores
higher" rather than exact values.
"""
import pandas as pd
import pytest

from pqc_riskmodel.labels import RiskLabelWeights, compute_risk_label, label_dataframe


def make_row(**overrides) -> pd.Series:
    base = dict(
        internet_accessible=0,
        has_public_cert=0,
        quantum_vulnerable=False,
        key_size=2048,
        criticality="LOW",
        migration_complexity="LOW",
        financial_data=0,
        customer_data=0,
        regulated_system=0,
        previous_incident_flag=0,
    )
    base.update(overrides)
    return pd.Series(base)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        RiskLabelWeights(exposure=0.9, criticality=0.9, algo_weakness=0.1, complexity=0.1, business=0.1)


def test_worst_case_scores_higher_than_best_case():
    worst = make_row(
        internet_accessible=1,
        has_public_cert=1,
        quantum_vulnerable=True,
        criticality="CRITICAL",
        migration_complexity="HIGH",
        financial_data=1,
        customer_data=1,
        regulated_system=1,
        previous_incident_flag=1,
    )
    best = make_row()  # all the safe/benign defaults above

    assert compute_risk_label(worst) > compute_risk_label(best)


def test_score_is_always_in_unit_range():
    for criticality in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        row = make_row(criticality=criticality, quantum_vulnerable=True, previous_incident_flag=1)
        score = compute_risk_label(row)
        assert 0.0 <= score <= 1.0


def test_label_dataframe_is_deterministic_given_seed():
    df = pd.DataFrame([make_row(criticality="HIGH").to_dict() for _ in range(20)])
    a = label_dataframe(df, seed=7)
    b = label_dataframe(df, seed=7)
    pd.testing.assert_series_equal(a["risk_score"], b["risk_score"])


def test_label_dataframe_adds_noise_across_identical_rows():
    df = pd.DataFrame([make_row(criticality="HIGH").to_dict() for _ in range(20)])
    labelled = label_dataframe(df, seed=1)
    # identical inputs should not all get the exact same label — otherwise
    # there is nothing non-linear for XGBoost to learn (see module docstring)
    assert labelled["risk_score"].nunique() > 1
