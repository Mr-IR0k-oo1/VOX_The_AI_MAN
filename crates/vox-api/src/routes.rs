//! HTTP handlers and error mapping.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use vox_core::{AnswerResponse, RetrieveRequest};
use vox_pipeline::PipelineError;
use vox_retrieval::RetrievalError;

use crate::metrics::init_metrics;
use crate::state::AppState;

/// Liveness probe.
pub async fn health(State(state): State<std::sync::Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "service": "vox",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": state.started_at.elapsed().as_secs(),
    }))
}

/// Prometheus exposition endpoint.
pub async fn metrics() -> impl IntoResponse {
    let body = init_metrics().render();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        body,
    )
}

/// Text query → grounded answer or refusal.
///
/// # Errors
/// Never returns `Err` directly; failures are mapped into responses via
/// [`ApiError`].
pub async fn query(
    State(state): State<std::sync::Arc<AppState>>,
    Json(request): Json<RetrieveRequest>,
) -> Result<Json<AnswerResponse>, ApiError> {
    metrics::counter!("vox_queries_total").increment(1);
    let response = state.pipeline.run(request).await.map_err(ApiError)?;
    if let Some(total) = response.timings_ms.total {
        metrics::histogram!("vox_query_latency_ms").record(total);
    }
    Ok(Json(response))
}

/// Voice ingestion endpoint, reserved for the STT phase.
pub async fn voice_query() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "not_implemented",
            "detail": "voice ingestion (STT) lands in a later phase",
        })),
    )
        .into_response()
}

/// Maps pipeline failures onto HTTP semantics.
#[derive(Debug)]
pub struct ApiError(PipelineError);

impl From<PipelineError> for ApiError {
    fn from(err: PipelineError) -> Self {
        Self(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = match &self.0 {
            PipelineError::Validation(_) => (StatusCode::UNPROCESSABLE_ENTITY, "invalid_request"),
            PipelineError::Retrieval(RetrievalError::Validation(_)) => {
                (StatusCode::UNPROCESSABLE_ENTITY, "invalid_request")
            }
            PipelineError::Retrieval(RetrievalError::Timeout { .. }) => {
                (StatusCode::GATEWAY_TIMEOUT, "retrieval_timeout")
            }
            PipelineError::Retrieval(_) => (StatusCode::SERVICE_UNAVAILABLE, "retrieval_unavailable"),
            PipelineError::Llm(_) => (StatusCode::BAD_GATEWAY, "llm_error"),
        };
        tracing::warn!(error = %self.0, status = %status, "request failed");
        (status, Json(json!({ "error": code, "detail": self.0.to_string() }))).into_response()
    }
}
