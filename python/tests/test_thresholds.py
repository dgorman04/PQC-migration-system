"""Unit tests for pqc_riskmodel.thresholds (README §7.5).

Exists mainly to catch path-resolution mistakes early: DEFAULT_PATH is
computed from `__file__`'s location relative to the repo, which is exactly
the kind of thing that looks right on inspection and is wrong until you
actually run it (this test caught a real off-by-one-directory bug the first
time it ran).
"""
import pytest

from pqc_riskmodel.thresholds import ThresholdTable


def test_default_path_loads_the_real_config_file():
    # No path argument — exercises DEFAULT_PATH, not a passed-in fixture.
    table = ThresholdTable.from_yaml()
    assert table.buckets[0][0] == "CRITICAL"
    assert table.buckets[-1] == ("LOW", 0.0)


def test_priority_for_picks_first_match_highest_first():
    table = ThresholdTable.from_yaml()
    assert table.priority_for(0.9) == "CRITICAL"
    assert table.priority_for(0.75) == "CRITICAL"  # boundary is inclusive
    assert table.priority_for(0.6) == "HIGH"
    assert table.priority_for(0.3) == "MEDIUM"
    assert table.priority_for(0.0) == "LOW"


def test_rank_orders_critical_as_most_urgent():
    table = ThresholdTable.from_yaml()
    assert table.rank("CRITICAL") < table.rank("HIGH") < table.rank("MEDIUM") < table.rank("LOW")


def test_rank_rejects_unknown_priority():
    table = ThresholdTable.from_yaml()
    with pytest.raises(ValueError):
        table.rank("NOT_A_PRIORITY")


def test_buckets_out_of_order_are_rejected(tmp_path):
    bad_yaml = tmp_path / "bad_thresholds.yaml"
    bad_yaml.write_text(
        "buckets:\n"
        "  - priority: LOW\n"
        "    min_score: 0.0\n"
        "  - priority: CRITICAL\n"
        "    min_score: 0.75\n"
    )
    with pytest.raises(ValueError):
        ThresholdTable.from_yaml(bad_yaml)
