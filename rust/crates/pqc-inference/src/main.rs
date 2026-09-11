//! Boots the quantum-risk inference service (README.md §7.6).
//!
//! Model resolution: if the env var named by `config.model.path_env`
//! (default `PQC_MODEL_PATH`) is set, loads that local `.onnx` file —
//! the path used by `scripts/train_local_demo.py`'s output and by CI.
//! Otherwise resolves `config.mlflow.model_name`'s alias
//! (`config.mlflow.alias_env`, default `champion`) against
//! `config.mlflow.tracking_uri_env` (default `http://localhost:5000`) and
//! downloads it. Run from `rust/crates/pqc-inference/` — config paths are
//! relative to CWD, not to this crate's manifest.

use std::env;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pqc_inference::{
    config, AppState, Config, Error, GradientBoostedAdapter, MlflowClient, OnnxModel,
    ThresholdTable,
};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    if let Err(error) = run().await {
        eprintln!("fatal: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Error> {
    let config = Config::load(config::config_path())?;

    let (model_bytes, model_path) = load_model_bytes(&config).await?;
    let model = OnnxModel::load_from_bytes(
        &model_bytes,
        &config.model.input_name,
        &config.model.output_name,
    )?;

    let thresholds_path = resolve_thresholds_path(model_path.as_deref());
    tracing::info!(path = %thresholds_path.display(), "loading risk thresholds");
    let thresholds = ThresholdTable::from_yaml_path(&thresholds_path)?;

    let adapter = GradientBoostedAdapter::new(model, config);
    let state = Arc::new(AppState {
        adapter,
        thresholds,
    });

    let bind_addr = config::bind_addr();
    let addr: SocketAddr = bind_addr.parse().map_err(|source| {
        Error::Config(format!(
            "invalid PQC_INFERENCE_BIND_ADDR `{bind_addr}`: {source}"
        ))
    })?;

    tracing::info!(%addr, "pqc-inference listening");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|source| Error::Config(format!("binding {addr}: {source}")))?;
    axum::serve(listener, pqc_inference::router(state))
        .await
        .map_err(|source| Error::Config(source.to_string()))?;

    Ok(())
}

async fn load_model_bytes(config: &Config) -> Result<(Vec<u8>, Option<PathBuf>), Error> {
    match env::var(&config.model.path_env) {
        Ok(path_str) => {
            let path = PathBuf::from(&path_str);
            tracing::info!(path = %path.display(), "loading model from local file");
            let bytes = std::fs::read(&path)
                .map_err(|source| Error::ModelLoad(format!("{}: {source}", path.display())))?;
            Ok((bytes, Some(path)))
        }
        Err(_) => {
            let tracking_uri = env::var(&config.mlflow.tracking_uri_env)
                .unwrap_or_else(|_| "http://localhost:5000".to_string());
            let alias =
                env::var(&config.mlflow.alias_env).unwrap_or_else(|_| "champion".to_string());
            tracing::info!(
                %tracking_uri,
                model_name = %config.mlflow.model_name,
                %alias,
                "{} unset — loading model from MLflow",
                config.model.path_env
            );

            let client = MlflowClient::new(tracking_uri);
            let resolved = client
                .download_model_by_alias(&config.mlflow.model_name, &alias)
                .await?;
            tracing::info!(run_id = %resolved.run_id, version = %resolved.version, "resolved model version");
            Ok((resolved.bytes, None))
        }
    }
}

/// `PQC_THRESHOLDS_PATH` wins if set. Otherwise, when the model came from a
/// local file, looks for a `risk_thresholds.yaml` sitting next to it — the
/// layout `scripts/train_local_demo.py` produces. Falls back to a
/// repo-relative guess that only makes sense when running from the crate
/// directory against the demo model — set `PQC_THRESHOLDS_PATH` explicitly
/// for anything else, including the MLflow path (no local directory to
/// look next to).
fn resolve_thresholds_path(model_path: Option<&Path>) -> PathBuf {
    if let Ok(explicit) = env::var("PQC_THRESHOLDS_PATH") {
        return PathBuf::from(explicit);
    }
    if let Some(model_path) = model_path {
        if let Some(dir) = model_path.parent() {
            let sibling = dir.join("risk_thresholds.yaml");
            if sibling.exists() {
                return sibling;
            }
        }
    }
    PathBuf::from("../../../models/quantum-risk/risk_thresholds.yaml")
}
