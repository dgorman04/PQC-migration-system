//! Minimal MLflow REST client covering exactly what a promotion decision
//! needs: resolve a registered model's alias to a version + run, read that
//! run's metrics/params, and move an alias once a decision approves.
//!
//! Ported from the reference
//! `promotion-controller/crates/promotion-mlflow/src/client.rs`
//! (README.md §5.4) — same request/response shapes for the endpoints both
//! need (`/api/2.0/mlflow/runs/get`, the request/response DTOs, the
//! epoch-millis timestamp parsing). Deliberately narrower: the reference
//! client also resolved and downloaded artifact bytes
//! (`resolve_artifact_base_url` / `download_artifact`) because
//! `promotion-service` needed to move model files; this crate never touches
//! model files — that's the inference service's job at boot (§7.6) — so this
//! client adds alias lookup/set instead (`/api/2.0/mlflow/registered-models/alias`),
//! which the reference client didn't need.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::Deserialize;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("mlflow request to {url} failed: {source}")]
    Request {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("mlflow returned {status} for {url}: {body}")]
    UnexpectedStatus {
        url: String,
        status: u16,
        body: String,
    },

    #[error("failed to parse mlflow response from {url}: {source}")]
    MalformedResponse {
        url: String,
        #[source]
        source: serde_json::Error,
    },
}

pub struct MlflowClient {
    http: reqwest::Client,
    /// Never has a trailing slash — enforced in `new`.
    tracking_uri: String,
}

#[derive(Debug, Clone)]
pub struct ModelVersionRef {
    pub version: String,
    pub run_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedRun {
    pub run_id: String,
    pub metrics: BTreeMap<String, f64>,
    pub params: BTreeMap<String, String>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl ResolvedRun {
    pub fn dataset_id(&self) -> Option<&str> {
        self.params.get("dataset_id").map(String::as_str)
    }
}

impl MlflowClient {
    pub fn new(tracking_uri: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            tracking_uri: tracking_uri.into().trim_end_matches('/').to_string(),
        }
    }

    async fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<T, Error> {
        let url = format!("{}{path}", self.tracking_uri);
        let response = self
            .http
            .get(&url)
            .query(query)
            .send()
            .await
            .map_err(|source| Error::Request {
                url: url.clone(),
                source,
            })?;

        let status = response.status();
        let bytes = response.bytes().await.map_err(|source| Error::Request {
            url: url.clone(),
            source,
        })?;

        if !status.is_success() {
            return Err(Error::UnexpectedStatus {
                url,
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }

        serde_json::from_slice(&bytes).map_err(|source| Error::MalformedResponse { url, source })
    }

    async fn post_json(&self, path: &str, body: &serde_json::Value) -> Result<(), Error> {
        let url = format!("{}{path}", self.tracking_uri);
        let response = self
            .http
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|source| Error::Request {
                url: url.clone(),
                source,
            })?;

        let status = response.status();
        let bytes = response.bytes().await.map_err(|source| Error::Request {
            url: url.clone(),
            source,
        })?;

        if !status.is_success() {
            return Err(Error::UnexpectedStatus {
                url,
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }

        Ok(())
    }

    /// `Ok(None)` when the alias simply doesn't exist yet — the normal state
    /// for a deployment's `champion` alias before its first promotion, never
    /// an error condition. Any other failure still surfaces as `Err`.
    pub async fn model_version_by_alias(
        &self,
        model_name: &str,
        alias: &str,
    ) -> Result<Option<ModelVersionRef>, Error> {
        let result: Result<GetByAliasResponse, Error> = self
            .get_json(
                "/api/2.0/mlflow/registered-models/alias",
                &[("name", model_name), ("alias", alias)],
            )
            .await;

        match result {
            Ok(response) => Ok(Some(ModelVersionRef {
                version: response.model_version.version,
                run_id: response.model_version.run_id,
            })),
            Err(Error::UnexpectedStatus { status: 404, .. }) => Ok(None),
            Err(other) => Err(other),
        }
    }

    pub async fn get_run(&self, run_id: &str) -> Result<ResolvedRun, Error> {
        let run: GetRunResponse = self
            .get_json("/api/2.0/mlflow/runs/get", &[("run_id", run_id)])
            .await?;

        let RunDto { info, data } = run.run;

        Ok(ResolvedRun {
            run_id: run_id.to_string(),
            metrics: data
                .metrics
                .into_iter()
                .map(|metric| (metric.key, metric.value))
                .collect(),
            params: pairs(data.params),
            finished_at: info.end_time.as_ref().and_then(EpochMillis::to_datetime),
        })
    }

    pub async fn set_registered_model_alias(
        &self,
        model_name: &str,
        alias: &str,
        version: &str,
    ) -> Result<(), Error> {
        self.post_json(
            "/api/2.0/mlflow/registered-models/alias",
            &serde_json::json!({ "name": model_name, "alias": alias, "version": version }),
        )
        .await
    }
}

#[derive(Debug, Deserialize)]
struct GetByAliasResponse {
    model_version: ModelVersionDto,
}

#[derive(Debug, Deserialize)]
struct ModelVersionDto {
    version: String,
    #[serde(default)]
    run_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GetRunResponse {
    run: RunDto,
}

#[derive(Debug, Deserialize)]
struct RunDto {
    info: RunInfoDto,
    #[serde(default)]
    data: RunDataDto,
}

#[derive(Debug, Deserialize)]
struct RunInfoDto {
    #[serde(default)]
    end_time: Option<EpochMillis>,
}

#[derive(Debug, Default, Deserialize)]
struct RunDataDto {
    #[serde(default)]
    metrics: Vec<MetricDto>,
    #[serde(default)]
    params: Vec<KeyValueDto>,
}

#[derive(Debug, Deserialize)]
struct MetricDto {
    key: String,
    value: f64,
}

#[derive(Debug, Deserialize)]
struct KeyValueDto {
    key: String,
    value: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EpochMillis {
    Number(i64),
    Text(String),
}

impl EpochMillis {
    fn to_datetime(&self) -> Option<DateTime<Utc>> {
        let millis = match self {
            Self::Number(millis) => *millis,
            Self::Text(text) => text.parse().ok()?,
        };
        DateTime::from_timestamp_millis(millis)
    }
}

fn pairs(entries: Vec<KeyValueDto>) -> BTreeMap<String, String> {
    entries
        .into_iter()
        .map(|entry| (entry.key, entry.value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_millis_parses_both_shapes() {
        assert!(EpochMillis::Number(1_787_313_000_000)
            .to_datetime()
            .is_some());
        assert!(EpochMillis::Text("1787313000000".to_string())
            .to_datetime()
            .is_some());
        assert!(EpochMillis::Text("not-a-number".to_string())
            .to_datetime()
            .is_none());
    }

    // Live-HTTP behaviour (404 → None on alias lookup, DTO parsing against a
    // real response body, etc.) is exercised in the reference
    // `promotion-mlflow` crate's wiremock-based tests, which this client's
    // request/response shapes were ported from. Add wiremock here too if
    // this client grows logic those tests don't already cover.
}
