//! The `gradient_boosted` adapter — the Rust counterpart to the reference
//! `inference-adapters::gradient_boosted_models` module, trimmed to the
//! `regression` head only (our model is an `XGBRegressor`, README §7.5).
//! Ties feature encoding + the ONNX session + head postprocessing together.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::config::{Config, Head};
use crate::error::Error;
use crate::features;
use crate::onnx::OnnxModel;

pub struct GradientBoostedAdapter {
    model: OnnxModel,
    config: Config,
}

impl GradientBoostedAdapter {
    pub fn new(model: OnnxModel, config: Config) -> Self {
        Self { model, config }
    }

    pub fn predict(&self, instances: &[BTreeMap<String, Value>]) -> Result<Vec<f32>, Error> {
        let rows = instances
            .iter()
            .map(|raw| features::encode(&self.config.features, raw))
            .collect::<Result<Vec<_>, _>>()?;

        let raw_scores = self.model.predict_batch(&rows)?;

        let scores = match self.config.model.head {
            // The model was trained to emit an already-clipped [0, 1] score
            // (README §7.5), but re-clip defensively at the serving
            // boundary rather than trust the artifact blindly.
            Head::Regression => raw_scores
                .into_iter()
                .map(|score| score.clamp(0.0, 1.0))
                .collect(),
        };

        Ok(scores)
    }
}
