//! Resolves a model by MLflow registered-model alias and downloads its
//! `.onnx` artifact — the boot-time fallback when `PQC_MODEL_PATH` isn't
//! set (README.md §7.6). This is the other half of the reference
//! `promotion-mlflow/src/client.rs` from the half `pqc-promote` ported
//! (§7.7): that crate resolves runs and sets aliases but never downloads
//! artifacts; this one resolves a run's `artifact_uri` and downloads
//! `model_onnx/model.onnx` from it. Small deliberate duplication between
//! the two crates rather than a shared `pqc-mlflow` crate — see README §5.1.

use serde::Deserialize;

use crate::error::Error;

pub struct MlflowClient {
    http: reqwest::Client,
    tracking_uri: String,
}

pub struct ResolvedArtifact {
    pub bytes: Vec<u8>,
    pub run_id: String,
    pub version: String,
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
            .map_err(|source| Error::ModelLoad(format!("request to {url} failed: {source}")))?;

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|source| Error::ModelLoad(format!("reading response from {url}: {source}")))?;

        if !status.is_success() {
            return Err(Error::ModelLoad(format!(
                "mlflow returned {status} for {url}: {}",
                String::from_utf8_lossy(&bytes)
            )));
        }

        serde_json::from_slice(&bytes)
            .map_err(|source| Error::ModelLoad(format!("parsing response from {url}: {source}")))
    }

    /// Resolves `model_name`'s `alias` to a version, resolves that
    /// version's run, downloads `model_onnx/model.onnx` from the run's
    /// artifacts, and returns the raw bytes.
    pub async fn download_model_by_alias(
        &self,
        model_name: &str,
        alias: &str,
    ) -> Result<ResolvedArtifact, Error> {
        let alias_response: GetByAliasResponse = self
            .get_json(
                "/api/2.0/mlflow/registered-models/alias",
                &[("name", model_name), ("alias", alias)],
            )
            .await?;
        let version = alias_response.model_version.version;
        let run_id = alias_response.model_version.run_id.ok_or_else(|| {
            Error::ModelLoad(format!(
                "model `{model_name}` version `{version}` has no run_id"
            ))
        })?;

        let run_response: GetRunResponse = self
            .get_json("/api/2.0/mlflow/runs/get", &[("run_id", run_id.as_str())])
            .await?;
        let artifact_base_url = resolve_artifact_base_url(
            &self.tracking_uri,
            &run_id,
            &run_response.run.info.artifact_uri,
        )?;

        let download_url = format!("{artifact_base_url}/model_onnx/model.onnx");
        let response = self
            .http
            .get(&download_url)
            .send()
            .await
            .map_err(|source| {
                Error::ModelLoad(format!("request to {download_url} failed: {source}"))
            })?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(|source| {
            Error::ModelLoad(format!("reading model bytes from {download_url}: {source}"))
        })?;
        if !status.is_success() {
            return Err(Error::ModelLoad(format!(
                "mlflow returned {status} downloading {download_url}"
            )));
        }

        Ok(ResolvedArtifact {
            bytes: bytes.to_vec(),
            run_id,
            version,
        })
    }
}

/// Rewrites a run's `mlflow-artifacts:` URI into an HTTP base URL under the
/// tracking server: `{tracking_uri}/api/2.0/mlflow-artifacts/artifacts/<path>`.
/// Ported from the reference `promotion-mlflow` client — see its own
/// doc comment for why this endpoint isn't part of MLflow's documented
/// stable REST contract, only exercised against a real server, not just
/// wiremock tests.
fn resolve_artifact_base_url(
    tracking_uri: &str,
    run_id: &str,
    artifact_uri: &str,
) -> Result<String, Error> {
    let Some(rest) = artifact_uri.strip_prefix("mlflow-artifacts:") else {
        return Err(Error::ModelLoad(format!(
            "run `{run_id}` has an artifact_uri this client doesn't know how to resolve: {artifact_uri}"
        )));
    };
    let path = rest.trim_start_matches('/');
    Ok(format!(
        "{tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{path}"
    ))
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
}

#[derive(Debug, Deserialize)]
struct RunInfoDto {
    artifact_uri: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_artifact_base_url_rewrites_the_mlflow_artifacts_scheme() {
        let resolved = resolve_artifact_base_url(
            "http://mlflow:5000",
            "run-abc",
            "mlflow-artifacts:/12/run-abc/artifacts",
        )
        .unwrap();
        assert_eq!(
            resolved,
            "http://mlflow:5000/api/2.0/mlflow-artifacts/artifacts/12/run-abc/artifacts"
        );
    }

    #[test]
    fn resolve_artifact_base_url_rejects_a_non_proxied_scheme() {
        let err = resolve_artifact_base_url(
            "http://mlflow:5000",
            "run-abc",
            "s3://bucket/12/run-abc/artifacts",
        );
        assert!(err.is_err());
    }
}
