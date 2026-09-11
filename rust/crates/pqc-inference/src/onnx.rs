//! ONNX session wrapper — the Rust counterpart to the reference
//! `inference-runtime` crate, trimmed to exactly what a batch regression
//! forward pass needs. See README.md §7.6.

use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Value as OrtValue;

use crate::error::Error;

/// `ort::Session::run` takes `&mut self` (it's not internally synchronized),
/// but this is shared across concurrent Axum handlers behind an `Arc`, so
/// it's wrapped in a `Mutex` — same reasoning and same consequence as the
/// reference `inference-runtime`'s `Mutex<ort::Session>`: inference for
/// this deployment is serialized, one model, one session, one lock. If that
/// ever shows up as a throughput ceiling, the fix is running more instances
/// behind a load balancer, not fighting the lock.
pub struct OnnxModel {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
}

impl OnnxModel {
    pub fn load_from_file(path: &Path, input_name: &str, output_name: &str) -> Result<Self, Error> {
        let session = Session::builder()
            .map_err(|source| Error::ModelLoad(source.to_string()))?
            .commit_from_file(path)
            .map_err(|source| Error::ModelLoad(format!("{}: {source}", path.display())))?;

        Ok(Self {
            session: Mutex::new(session),
            input_name: input_name.to_string(),
            output_name: output_name.to_string(),
        })
    }

    pub fn load_from_bytes(
        bytes: &[u8],
        input_name: &str,
        output_name: &str,
    ) -> Result<Self, Error> {
        let session = Session::builder()
            .map_err(|source| Error::ModelLoad(source.to_string()))?
            .commit_from_memory(bytes)
            .map_err(|source| Error::ModelLoad(source.to_string()))?;

        Ok(Self {
            session: Mutex::new(session),
            input_name: input_name.to_string(),
            output_name: output_name.to_string(),
        })
    }

    /// `rows` is one feature vector per instance, already encoded and in
    /// the config's declared column order. Returns one raw score per row,
    /// in the same order.
    pub fn predict_batch(&self, rows: &[Vec<f32>]) -> Result<Vec<f32>, Error> {
        let n_rows = rows.len();
        let n_cols = rows.first().map(|row| row.len()).unwrap_or(0);

        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let array = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
            .map_err(|source| Error::Inference(format!("building input tensor: {source}")))?;

        let input_value = OrtValue::from_array(array)
            .map_err(|source| Error::Inference(format!("wrapping input tensor: {source}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::Inference("onnx session lock poisoned".to_string()))?;

        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => input_value])
            .map_err(|source| Error::Inference(format!("session.run: {source}")))?;

        let output = outputs.get(self.output_name.as_str()).ok_or_else(|| {
            Error::Inference(format!(
                "output `{}` not present in response",
                self.output_name
            ))
        })?;

        let (_shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|source| Error::Inference(format!("extracting output tensor: {source}")))?;

        Ok(data.to_vec())
    }
}
