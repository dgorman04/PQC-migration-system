use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("config error: {0}")]
    Config(String),

    #[error("failed to load model: {0}")]
    ModelLoad(String),

    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("inference failed: {0}")]
    Inference(String),
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match &self {
            Error::BadRequest(_) => StatusCode::BAD_REQUEST,
            Error::Config(_) | Error::ModelLoad(_) | Error::Inference(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        let body = Json(json!({ "error": self.to_string() }));
        (status, body).into_response()
    }
}
