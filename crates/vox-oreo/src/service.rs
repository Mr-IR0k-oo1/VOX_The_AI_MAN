//! Standalone OREO retrieval service.
//!
//! Exposes the frozen wire contract on its own port so the engine can run
//! independently of VOX (`oreo serve`), while `vox-api` can either talk to it
//! over HTTP or embed it in-process. The request/response shapes are exactly
//! `vox_types::{Query, RetrievalResponse}`.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use vox_types::{Query, RetrievalResponse};

use crate::engine::OreoEngine;
use crate::error::OreoError;

/// Shared service state.
#[derive(Clone)]
pub struct ServiceState {
    engine: Arc<OreoEngine>,
}

/// Builds the standalone router over `engine`.
pub fn build_router(engine: Arc<OreoEngine>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/retrieve", post(retrieve))
        .with_state(ServiceState { engine })
}

/// Liveness probe with index sizes.
async fn health(State(state): State<ServiceState>) -> Response {
    let dense = state.engine.dense_count().await.unwrap_or(0);
    let sparse = state.engine.sparse_count().unwrap_or(0);
    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "service": "oreo",
            "version": env!("CARGO_PKG_VERSION"),
            "dense_points": dense,
            "bm25_docs": sparse,
        })),
    )
        .into_response()
}

/// `POST /v1/retrieve` — hybrid retrieval over the indexed corpus.
///
/// # Errors
/// Never returns `Err` directly; failures map onto [`ServiceError`].
async fn retrieve(
    State(state): State<ServiceState>,
    Json(query): Json<Query>,
) -> Result<Json<RetrievalResponse>, ServiceError> {
    let response = state.engine.retrieve(query).await?;
    Ok(Json(response))
}

/// Maps engine failures onto HTTP semantics (mirrors `vox-api`).
#[derive(Debug)]
pub struct ServiceError(OreoError);

impl From<OreoError> for ServiceError {
    fn from(err: OreoError) -> Self {
        Self(err)
    }
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        let (status, code) = match &self.0 {
            OreoError::Validation(_) => (StatusCode::UNPROCESSABLE_ENTITY, "invalid_request"),
            _ => (StatusCode::SERVICE_UNAVAILABLE, "retrieval_unavailable"),
        };
        tracing::warn!(error = %self.0, status = %status, "retrieve failed");
        (
            status,
            Json(json!({ "error": code, "detail": self.0.to_string() })),
        )
            .into_response()
    }
}
