//! Ported from the reference
//! `promotion-controller/crates/promotion-policy/src/decision.rs`
//! (README.md §5.4). Trimmed: the reference also built a `DecisionRecord` /
//! `MetricEvidence` audit-trail object from `promotion-core`; this project
//! has nothing that persists promotion history yet, so `Decision` here
//! carries just the outcome and the comparison it was decided from. Add that
//! back if/when something needs to read promotion history programmatically.

use std::fmt;

use chrono::{DateTime, Utc};

use crate::compare::{Comparison, NotComparable};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Gate {
    Comparability,
    AbsoluteFloor,
    RelativeImprovement,
    Regression,
    Age,
    EnvironmentGating,
}

impl fmt::Display for Gate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Comparability => "comparability",
            Self::AbsoluteFloor => "absolute floor",
            Self::RelativeImprovement => "relative improvement",
            Self::Regression => "regression",
            Self::Age => "age",
            Self::EnvironmentGating => "environment gating",
        })
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Rejected {
    #[error("{metric} is {observed:.4}, must be above {floor:.4} for this environment")]
    BelowFloor {
        metric: String,
        observed: f64,
        floor: f64,
    },

    #[error(
        "{metric} moves {gap:+.4} against the champion. the policy requires at least {required:+.4}"
    )]
    NotEnoughBetter {
        metric: String,
        gap: f64,
        required: f64,
    },

    #[error(
        "{metric} is {regression:.4} less than in the champion. it can lose at most {allowed:.4}"
    )]
    GuardedRegression {
        metric: String,
        regression: f64,
        allowed: f64,
    },

    #[error("metric `{metric}` is referenced by policy but is not in the comparison")]
    MetricNotEvaluated { metric: String, gate: Gate },
}

impl Rejected {
    pub fn gate(&self) -> Gate {
        match self {
            Self::BelowFloor { .. } => Gate::AbsoluteFloor,
            Self::NotEnoughBetter { .. } => Gate::RelativeImprovement,
            Self::GuardedRegression { .. } => Gate::Regression,
            Self::MetricNotEvaluated { gate, .. } => *gate,
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Parked {
    #[error("the candidate's evaluation is {age_days} days old; policy allows at most {max_days} days — re-evaluate before promoting")]
    EvaluationTooOld { age_days: i64, max_days: i64 },

    #[error(
        "{environment} last promoted a model too recently; policy requires waiting until {until}"
    )]
    RateLimited {
        environment: String,
        interval_hours: i64,
        until: DateTime<Utc>,
    },

    #[error("{0}")]
    NotComparable(#[from] NotComparable),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Approve,
    Reject(Rejected),
    Park(Parked),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub outcome: Outcome,
    pub comparison: Option<Comparison>,
}

impl Decision {
    pub fn approved(&self) -> bool {
        matches!(self.outcome, Outcome::Approve)
    }
}
