//! HTTP surface: `/healthz` and the single deployment's `/infer` endpoint.
//! Shape borrows from the reference `inference-api` (`instance_id` in,
//! `instance_id` + result out) but trimmed — one deployment, one profile
//! (`api_only`), no `execution`/`timing` envelope, no per-instance
//! error/ok split (a bad instance fails the whole batch with 400/500,
//! since there is no Kafka DLQ here to make partial-batch semantics worth
//! the complexity). See README.md §7.6.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapter::GradientBoostedAdapter;
use crate::error::Error;
use crate::thresholds::ThresholdTable;

pub struct AppState {
    pub adapter: GradientBoostedAdapter,
    pub thresholds: ThresholdTable,
}

#[derive(Debug, Deserialize)]
pub struct InferRequest {
    pub instances: Vec<Instance>,
}

#[derive(Debug, Deserialize)]
pub struct Instance {
    pub instance_id: String,
    pub features: BTreeMap<String, Value>,
}

#[derive(Debug, Serialize)]
pub struct InferResponse {
    pub results: Vec<InstanceResult>,
}

#[derive(Debug, Serialize)]
pub struct InstanceResult {
    pub instance_id: String,
    pub risk_score: f32,
    pub priority: String,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/api/v1/deployments/quantum-risk/infer", post(infer))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn infer(
    State(state): State<Arc<AppState>>,
    Json(request): Json<InferRequest>,
) -> Result<Json<InferResponse>, Error> {
    if request.instances.is_empty() {
        return Err(Error::BadRequest("instances must not be empty".to_string()));
    }
    if request.instances.len() > 128 {
        return Err(Error::BadRequest(
            "at most 128 instances per request".to_string(),
        ));
    }

    let raw_features: Vec<_> = request
        .instances
        .iter()
        .map(|instance| instance.features.clone())
        .collect();
    let scores = state.adapter.predict(&raw_features)?;

    let results = request
        .instances
        .iter()
        .zip(scores)
        .map(|(instance, score)| InstanceResult {
            instance_id: instance.instance_id.clone(),
            risk_score: score,
            priority: state.thresholds.priority_for(score as f64).to_string(),
        })
        .collect();

    Ok(Json(InferResponse { results }))
}
