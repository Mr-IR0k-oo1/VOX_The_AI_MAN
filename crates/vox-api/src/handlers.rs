//! Route handlers for the frozen VOX endpoint set.
//!
//! Phase 0: request validation is active; pipeline execution is not
//! implemented yet and returns `501 Not Implemented`.

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

use crate::error::ApiError;
use vox_types::{QueryRequest, RetrieveRequest, VoiceRequest};

/// Service health payload.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// Placeholder metrics until per-stage latency export lands.
pub async fn metrics() -> Response {
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        "# HELP vox_up Whether the VOX service is up.\n# TYPE vox_up gauge\nvox_up 1\n",
    )
        .into_response()
}

pub async fn retrieve(
    Json(request): Json<RetrieveRequest>,
) -> Result<Json<RetrievalResponseStub>, ApiError> {
    request
        .validate()
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    Err(ApiError::not_implemented(
        "retrieval pipeline is not wired yet (Phase 0 freeze)",
    ))
}

pub async fn query(Json(request): Json<QueryRequest>) -> Result<Json<QueryResponseStub>, ApiError> {
    request
        .validate()
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    Err(ApiError::not_implemented(
        "text-to-answer pipeline is not wired yet (Phase 0 freeze)",
    ))
}

pub async fn voice_query(
    Json(request): Json<VoiceRequest>,
) -> Result<Json<QueryResponseStub>, ApiError> {
    request
        .validate()
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    Err(ApiError::not_implemented(
        "voice-to-answer pipeline is not wired yet (Phase 0 freeze)",
    ))
}

/// Placeholder success types; replaced by real domain responses when the
/// orchestrator is wired in.
#[derive(Debug, Serialize)]
pub struct RetrievalResponseStub {
    pub results: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct QueryResponseStub {
    pub answer: String,
}
