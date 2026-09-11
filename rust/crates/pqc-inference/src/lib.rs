//! ONNX serving for the quantum-risk model. See README.md §7.6 for the
//! reference this was ported/trimmed from and what was deliberately left
//! out (Kafka, the transformer adapter, `*_extracted` feature kinds).

pub mod adapter;
pub mod api;
pub mod config;
pub mod error;
pub mod features;
pub mod mlflow;
pub mod onnx;
pub mod thresholds;

pub use adapter::GradientBoostedAdapter;
pub use api::{router, AppState};
pub use config::Config;
pub use error::Error;
pub use mlflow::MlflowClient;
pub use onnx::OnnxModel;
pub use thresholds::ThresholdTable;
