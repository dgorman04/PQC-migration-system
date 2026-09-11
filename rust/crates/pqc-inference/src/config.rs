//! Loads `config/quantum-risk.yaml` (README.md §7.6). Deliberately one
//! config file for one deployment — see the comment at the top of that
//! file for why this collapses the reference project's
//! deployment/adapter/serving split.

use std::env;

use serde::Deserialize;

use crate::error::Error;
use crate::features::FeatureSpec;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Head {
    Regression,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub path_env: String,
    pub input_name: String,
    pub output_name: String,
    pub head: Head,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MlflowConfig {
    pub tracking_uri_env: String,
    pub model_name: String,
    pub alias_env: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub mlflow: MlflowConfig,
    pub features: Vec<FeatureSpec>,
}

impl Config {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|source| Error::Config(format!("reading {}: {source}", path.display())))?;
        serde_yaml::from_str(&text)
            .map_err(|source| Error::Config(format!("parsing {}: {source}", path.display())))
    }
}

/// Path to `config/quantum-risk.yaml`, overridable so the binary doesn't
/// have to be launched from a specific working directory.
pub fn config_path() -> String {
    env::var("PQC_INFERENCE_CONFIG").unwrap_or_else(|_| "config/quantum-risk.yaml".to_string())
}

pub fn bind_addr() -> String {
    env::var("PQC_INFERENCE_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8081".to_string())
}
