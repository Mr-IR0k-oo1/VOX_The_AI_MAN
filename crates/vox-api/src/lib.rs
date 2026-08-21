//! VOX API server: Axum wiring, configuration, and process bootstrap.
//!
//! The binary (`main.rs`) is thin; everything here is reusable so integration
//! tests exercise the exact router that production serves.

pub mod config;
pub mod logging;
pub mod metrics;
pub mod routes;
pub mod state;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use vox_core::{PipelineConfig, VoiceRagPipeline};
use vox_llm::{ExtractiveProvider, LlmProvider};
use vox_retrieval::{HttpRetrievalClient, MockRetrievalClient, RetrievalClient};
use vox_stt::{SpeechRecognizer, StubSpeechRecognizer};

pub use crate::config::{Config, ConfigError, LogFormat, RetrievalMode};
pub use crate::state::AppState;

/// The endpoints frozen in Phase 0, as `(method, path)` pairs.
pub const FROZEN_ENDPOINTS: [(&str, &str); 5] = [
    ("POST", "/v1/retrieve"),
    ("POST", "/v1/query"),
    ("POST", "/v1/voice/query"),
    ("GET", "/health"),
    ("GET", "/metrics"),
];

/// Builds the application router over `state`.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/metrics", get(routes::metrics))
        .route("/v1/retrieve", post(routes::retrieve))
        .route("/v1/query", post(routes::query))
        .route("/v1/voice/query", post(routes::voice_query))
        .with_state(state)
}

/// Builds [`AppState`] from `config`, selecting the backend implementations.
///
/// # Errors
/// Returns [`ConfigError`] when the HTTP retrieval backend cannot be built.
pub fn build_state(config: &Config) -> Result<AppState, ConfigError> {
    let retrieval: Arc<dyn RetrievalClient> = match config.retrieval.mode {
        RetrievalMode::Mock => Arc::new(MockRetrievalClient::new(config.retrieval.mock_delay)),
        RetrievalMode::Http => {
            let client = HttpRetrievalClient::new(
                config.retrieval.base_url.clone(),
                config.retrieval.timeout,
                config.retrieval.retry,
            )
            .map_err(|err| ConfigError::HttpClient(err.to_string()))?;
            Arc::new(client)
        }
    };

    let stt: Arc<dyn SpeechRecognizer> = Arc::new(StubSpeechRecognizer);
    let llm: Arc<dyn LlmProvider> = Arc::new(ExtractiveProvider);

    tracing::info!(
        retrieval = retrieval.name(),
        stt = stt.name(),
        llm = llm.name(),
        timeout_ms = config.retrieval.timeout.as_millis() as u64,
        "backends selected"
    );

    let pipeline = VoiceRagPipeline::new(
        Arc::clone(&retrieval),
        Arc::clone(&llm),
        PipelineConfig {
            grounding_min_score: config.pipeline.grounding_min_score,
        },
    );

    Ok(AppState {
        pipeline: Arc::new(pipeline),
        retrieval,
        stt,
        started_at: std::time::Instant::now(),
    })
}
