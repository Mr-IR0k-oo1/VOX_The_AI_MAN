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
use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use vox_pipeline::{ExtractiveLlmClient, PipelineConfig, VoiceRagPipeline};
use vox_retrieval::{HttpRetrievalClient, MockRetrievalClient, RetrievalClient};

pub use crate::config::{Config, ConfigError, LogFormat, RetrievalMode};
pub use crate::state::AppState;

use crate::routes::{handlers_placeholder_unused};

/// Builds the application router over `state`.
#[must_use]
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/metrics", get(routes::metrics))
        .route("/v1/query", post(routes::query))
        .route("/v1/voice/query", post(routes::voice_query))
        .with_state(state)
}

/// Builds [`AppState`] from `config`, selecting the retrieval backend.
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

    tracing::info!(
        backend = retrieval.name(),
        timeout_ms = config.retrieval.timeout.as_millis() as u64,
        "retrieval backend selected"
    );

    let pipeline = VoiceRagPipeline::new(
        retrieval,
        Arc::new(ExtractiveLlmClient),
        PipelineConfig {
            grounding_min_score: config.pipeline.grounding_min_score,
        },
    );

    Ok(AppState {
        pipeline: Arc::new(pipeline),
        started_at: std::time::Instant::now(),
    })
}

const _: () = {
    // Compile-time guard: keep imports honest.
    let _ = Duration::ZERO;
};
