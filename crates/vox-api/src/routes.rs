//! HTTP handlers and error mapping for the frozen v1 endpoints.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use vox_core::PipelineError;
use vox_ingest::IngestError;
use vox_llm::LlmError;
use vox_retrieval::RetrievalError;
use vox_stt::SttError;
use vox_types::{
    AnswerResponse, Query, QueryIntent, RetrievalResponse, ValidationError, VoiceRequest,
    VoiceResponse,
};

use crate::metrics::init_metrics;
use crate::state::AppState;

/// Liveness probe.
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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

/// Hybrid retrieval over the configured backend (`POST /v1/retrieve`).
///
/// # Errors
/// Never returns `Err` directly; failures are mapped into responses via
/// [`ApiError`].
pub async fn retrieve(
    State(state): State<Arc<AppState>>,
    Json(query): Json<Query>,
) -> Result<Json<RetrievalResponse>, ApiError> {
    metrics::counter!("vox_retrieve_requests_total").increment(1);
    let response = state.retrieval.retrieve(query).await?;
    Ok(Json(response))
}

/// Text query → grounded answer or refusal (`POST /v1/query`).
///
/// # Errors
/// Never returns `Err` directly; failures are mapped into responses via
/// [`ApiError`].
pub async fn query(
    State(state): State<Arc<AppState>>,
    Json(request): Json<Query>,
) -> Result<Json<AnswerResponse>, ApiError> {
    metrics::counter!("vox_queries_total").increment(1);
    let response = state.pipeline.run(request).await?;
    if let Some(total) = response.timings_ms.total {
        metrics::histogram!("vox_query_latency_ms").record(total);
    }
    Ok(Json(response))
}

/// Voice query → transcript → grounded answer or refusal
/// (`POST /v1/voice/query`).
///
/// Phase 0 wiring: ingestion validation and the STT boundary are exercised,
/// but no real recognizer exists yet, so requests fail with `501` at the STT
/// stage until one is integrated.
///
/// # Errors
/// Never returns `Err` directly; failures are mapped into responses via
/// [`ApiError`].
pub async fn voice_query(
    State(state): State<Arc<AppState>>,
    Json(request): Json<VoiceRequest>,
) -> Result<Json<VoiceResponse>, ApiError> {
    metrics::counter!("vox_voice_requests_total").increment(1);
    request.validate()?;
    let audio = vox_ingest::decode_audio(&request.audio_base64, request.format)?;
    let transcript = state
        .stt
        .transcribe(&audio.bytes, audio.format, request.language)
        .await?;

    let answer = state
        .pipeline
        .run(Query {
            query: transcript.text.clone(),
            language: transcript.language,
            top_k: request.top_k,
            intent: QueryIntent::Unknown,
        })
        .await?;
    if let Some(total) = answer.timings_ms.total {
        metrics::histogram!("vox_voice_latency_ms").record(total);
    }

    Ok(Json(VoiceResponse {
        transcript,
        answer: answer.answer,
        grounded: answer.grounded,
        refusal_reason: answer.refusal_reason,
        timings_ms: answer.timings_ms,
    }))
}

/// Maps domain failures onto HTTP semantics.
#[derive(Debug)]
pub enum ApiError {
    /// Input failed domain validation.
    Validation(ValidationError),
    /// The retrieval boundary failed.
    Retrieval(RetrievalError),
    /// The generation backend failed.
    Llm(LlmError),
    /// The speech-to-text boundary failed.
    Stt(SttError),
    /// The voice payload failed ingestion checks.
    Ingest(IngestError),
}

impl From<ValidationError> for ApiError {
    fn from(err: ValidationError) -> Self {
        Self::Validation(err)
    }
}

impl From<PipelineError> for ApiError {
    fn from(err: PipelineError) -> Self {
        match err {
            PipelineError::Validation(err) => Self::Validation(err),
            PipelineError::Retrieval(err) => Self::Retrieval(err),
            PipelineError::Llm(err) => Self::Llm(err),
        }
    }
}

impl From<RetrievalError> for ApiError {
    fn from(err: RetrievalError) -> Self {
        Self::Retrieval(err)
    }
}

impl From<LlmError> for ApiError {
    fn from(err: LlmError) -> Self {
        Self::Llm(err)
    }
}

impl From<SttError> for ApiError {
    fn from(err: SttError) -> Self {
        Self::Stt(err)
    }
}

impl From<IngestError> for ApiError {
    fn from(err: IngestError) -> Self {
        Self::Ingest(err)
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Validation(err) => write!(f, "{err}"),
            ApiError::Retrieval(err) => write!(f, "{err}"),
            ApiError::Llm(err) => write!(f, "{err}"),
            ApiError::Stt(err) => write!(f, "{err}"),
            ApiError::Ingest(err) => write!(f, "{err}"),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = match &self {
            ApiError::Validation(_) => (StatusCode::UNPROCESSABLE_ENTITY, "invalid_request"),
            ApiError::Ingest(_) => (StatusCode::UNPROCESSABLE_ENTITY, "invalid_audio"),
            ApiError::Retrieval(RetrievalError::Validation(_)) => {
                (StatusCode::UNPROCESSABLE_ENTITY, "invalid_request")
            }
            ApiError::Retrieval(RetrievalError::Timeout { .. }) => {
                (StatusCode::GATEWAY_TIMEOUT, "retrieval_timeout")
            }
            ApiError::Retrieval(_) => (StatusCode::SERVICE_UNAVAILABLE, "retrieval_unavailable"),
            ApiError::Llm(_) => (StatusCode::BAD_GATEWAY, "llm_error"),
            ApiError::Stt(SttError::NotImplemented) => {
                (StatusCode::NOT_IMPLEMENTED, "stt_not_implemented")
            }
            ApiError::Stt(_) => (StatusCode::BAD_GATEWAY, "stt_error"),
        };
        tracing::warn!(error = %self, status = %status, "request failed");
        (
            status,
            Json(json!({ "error": code, "detail": self.to_string() })),
        )
            .into_response()
    }
}
