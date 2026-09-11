//! Model promotion gate engine — the replacement for the reference
//! `promotion-controller` workspace (14 crates + 3 binaries). Same idea (a
//! declarative gate engine that approves/rejects/parks a candidate model
//! with a reason per gate, plus an MLflow client to act on the decision),
//! ported to one crate because this project needs neither a persistent
//! service, S3 bundle storage, nor a reconcile agent — see README.md §5.1,
//! §5.4, §7.7.
//!
//! Module layout mirrors the reference crate's own file split
//! (`compare.rs` / `decision.rs` / `gates.rs` / `policy.rs` / `evaluate.rs`)
//! one-for-one, plus `mlflow.rs` (ported from the reference
//! `promotion-mlflow` crate) which the reference kept as a separate crate.

pub mod compare;
pub mod decision;
pub mod evaluate;
pub mod gates;
pub mod mlflow;
pub mod policy;

pub use compare::{Candidate, Champion, Comparison, Eval, MetricPair, NotComparable, Side};
pub use decision::{Decision, Gate, Outcome, Parked, Rejected};
pub use evaluate::evaluate;
pub use mlflow::{Error as MlflowError, MlflowClient, ModelVersionRef, ResolvedRun};
pub use policy::{
    DeploymentPolicy, EnvironmentDefaults, EnvironmentPolicy, InvalidPolicy, NotInPolicy, OnPass,
    Policy, Rules,
};
