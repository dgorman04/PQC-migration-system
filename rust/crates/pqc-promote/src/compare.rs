//! Ported from the reference `promotion-controller/crates/promotion-policy/src/compare.rs`
//! (README.md §5.4). Same logic; `Eval` replaces the reference's
//! `BundleManifest`-backed lookup since this project has no bundle-manifest
//! concept — a run's dataset id, evaluation time, and metrics are read
//! straight off an MLflow run by `main.rs` via `crate::mlflow`.

use std::collections::BTreeMap;
use std::fmt;

use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Champion,
    Candidate,
}

impl fmt::Display for Side {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Champion => "champion",
            Self::Candidate => "candidate",
        })
    }
}

/// What `compare()` needs about one side. Built from an MLflow run's
/// params/metrics — this module knows nothing about MLflow itself.
#[derive(Debug, Clone)]
pub struct Eval {
    pub dataset_id: String,
    pub evaluated_at: DateTime<Utc>,
    pub metrics: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct Champion<'a>(pub &'a Eval);

#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a>(pub &'a Eval);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NotComparable {
    #[error("the {side} has no evaluation record, so there is nothing to compare")]
    MissingEval { side: Side },

    #[error(
        "the two models are evaluated with different datasets - champion on `{champion}`, candidate on `{candidate}`. they must have the same dataset to be compared."
    )]
    DatasetMismatch { champion: String, candidate: String },

    #[error("metric `{metric}` is included in the {included_in} but not the other side")]
    MetricMissing { metric: String, included_in: Side },

    #[error("neither side has any metrics")]
    NoMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricPair {
    pub champion: f64,
    pub candidate: f64,
}

impl MetricPair {
    pub fn gap(&self) -> f64 {
        self.candidate - self.champion
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub dataset_id: String,
    pub candidate_evaluated_at: DateTime<Utc>,
    metrics: BTreeMap<String, MetricPair>,
}

impl Comparison {
    pub fn metric(&self, name: &str) -> Option<&MetricPair> {
        self.metrics.get(name)
    }

    pub fn metrics(&self) -> impl Iterator<Item = (&str, &MetricPair)> {
        self.metrics
            .iter()
            .map(|(name, pair)| (name.as_str(), pair))
    }
}

/// `champion` is `None` when this deployment has never been promoted before
/// — the caller (`main.rs`) treats that as a bootstrap case, not by calling
/// this function differently, so it still comes through here as a
/// `MissingEval` refusal for a consistent, single code path.
pub fn compare(
    champion: Option<Champion<'_>>,
    candidate: Candidate<'_>,
) -> Result<Comparison, NotComparable> {
    let champion = champion
        .ok_or(NotComparable::MissingEval {
            side: Side::Champion,
        })?
        .0;
    let candidate = candidate.0;

    if champion.dataset_id != candidate.dataset_id {
        return Err(NotComparable::DatasetMismatch {
            champion: champion.dataset_id.clone(),
            candidate: candidate.dataset_id.clone(),
        });
    }

    if champion.metrics.is_empty() && candidate.metrics.is_empty() {
        return Err(NotComparable::NoMetrics);
    }

    for metric in champion.metrics.keys() {
        if !candidate.metrics.contains_key(metric) {
            return Err(NotComparable::MetricMissing {
                metric: metric.clone(),
                included_in: Side::Champion,
            });
        }
    }
    for metric in candidate.metrics.keys() {
        if !champion.metrics.contains_key(metric) {
            return Err(NotComparable::MetricMissing {
                metric: metric.clone(),
                included_in: Side::Candidate,
            });
        }
    }

    let metrics: BTreeMap<String, MetricPair> = champion
        .metrics
        .iter()
        .map(|(name, champion_value)| {
            let pair = MetricPair {
                champion: *champion_value,
                candidate: candidate.metrics[name],
            };
            (name.clone(), pair)
        })
        .collect();

    Ok(Comparison {
        dataset_id: candidate.dataset_id.clone(),
        candidate_evaluated_at: candidate.evaluated_at,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(dataset_id: &str, metrics: &[(&str, f64)]) -> Eval {
        Eval {
            dataset_id: dataset_id.to_string(),
            evaluated_at: Utc::now(),
            metrics: metrics.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    #[test]
    fn missing_champion_is_refused() {
        let candidate = eval("ds-1", &[("f1", 0.9)]);
        let err = compare(None, Candidate(&candidate)).unwrap_err();
        assert_eq!(
            err,
            NotComparable::MissingEval {
                side: Side::Champion
            }
        );
    }

    #[test]
    fn dataset_mismatch_is_refused() {
        let champion = eval("ds-1", &[("f1", 0.9)]);
        let candidate = eval("ds-2", &[("f1", 0.9)]);
        let err = compare(Some(Champion(&champion)), Candidate(&candidate)).unwrap_err();
        assert!(matches!(err, NotComparable::DatasetMismatch { .. }));
    }

    #[test]
    fn matching_metrics_produce_a_comparison() {
        let champion = eval("ds-1", &[("f1", 0.80)]);
        let candidate = eval("ds-1", &[("f1", 0.85)]);
        let comparison = compare(Some(Champion(&champion)), Candidate(&candidate)).unwrap();
        let pair = comparison.metric("f1").unwrap();
        assert!((pair.gap() - 0.05).abs() < 1e-9);
    }
}
