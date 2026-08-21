//! VOX API server: Axum wiring, configuration, and process bootstrap.
//!
//! The binary (`main.rs`) is thin; everything here is reusable so integration
//! tests exercise the exact router that production serves.

pub mod config;
pub mod ids;
pub mod logging;
pub mod metrics;
pub mod routes;
pub mod state;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use vox_core::VoxPipeline;
use vox_guard::GuardService;
use vox_llm::{ExtractiveProvider, LlmProvider, OpenAiCompatibleProvider};
use vox_retrieval::{
    EmbeddedOreoClient, MockRetrievalClient, OREORetrievalClient, RetrievalClient,
};
use vox_stt::{MockRecognizer, SarvamRecognizer, SpeechRecognizer};

pub use crate::config::{Config, ConfigError, LlmMode, LogFormat, RetrievalMode, SttMode};
pub use crate::state::AppState;

/// The endpoints frozen in Phase 0, as `(method, path)` pairs.
pub const FROZEN_ENDPOINTS: [(&str, &str); 5] = [
    ("POST", "/v1/retrieve"),
    ("POST", "/v1/query"),
    ("POST", "/v1/voice/query"),
    ("GET", "/health"),
    ("GET", "/metrics"),
];

/// Maximum accepted JSON body size.
///
/// Driven by the voice endpoint: base64 inflates audio by ~4/3, so a
/// 10 MB recording (the ingest limit) arrives as a ~14 MB JSON body.
pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Builds the application router over `state`.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(routes::demo_ui))
        .route("/demo", get(routes::demo_ui))
        .route("/health", get(routes::health))
        .route("/metrics", get(routes::metrics))
        .route("/v1/retrieve", post(routes::retrieve))
        .route("/v1/query", post(routes::query))
        .route("/v1/voice/query", post(routes::voice_query))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

/// Builds [`AppState`] from `config`, selecting the backend implementations.
///
/// # Errors
/// Returns [`ConfigError`] when an HTTP backend client or the embedded
/// OREO engine cannot be built.
pub fn build_state(config: &Config) -> Result<AppState, ConfigError> {
    let retrieval: Arc<dyn RetrievalClient> = match config.retrieval.mode {
        RetrievalMode::Mock => Arc::new(MockRetrievalClient::new(config.retrieval.mock_delay)),
        RetrievalMode::Http => {
            let client = OREORetrievalClient::new(
                config.retrieval.base_url.clone(),
                config.retrieval.timeout,
                config.retrieval.retry,
            )
            .map_err(|err| ConfigError::HttpClient(err.to_string()))?;
            Arc::new(client)
        }
        RetrievalMode::Oreo => {
            let oreo_config = config
                .retrieval
                .oreo
                .clone()
                .ok_or_else(|| ConfigError::OreoEngine("oreo settings missing".into()))?;
            let engine = vox_oreo::OreoEngine::new(oreo_config)
                .map_err(|err| ConfigError::OreoEngine(err.to_string()))?;
            Arc::new(EmbeddedOreoClient::new(Arc::new(engine)))
        }
    };

    let stt: Arc<dyn SpeechRecognizer> = match config.stt.mode {
        SttMode::Mock => Arc::new(MockRecognizer::new()),
        SttMode::Sarvam => {
            let client = SarvamRecognizer::new(
                config.stt.sarvam_api_key.clone(),
                config.stt.sarvam_model.clone(),
                config.stt.sarvam_base_url.clone(),
                config.stt.sarvam_timeout,
            )
            .map_err(|err| ConfigError::HttpClient(err.to_string()))?;
            Arc::new(client)
        }
    };

    let llm: Arc<dyn LlmProvider> = match config.llm.mode {
        LlmMode::Extractive => Arc::new(ExtractiveProvider),
        LlmMode::OpenAi => {
            let client = OpenAiCompatibleProvider::new(
                config.llm.api_key.clone(),
                config.llm.model.clone(),
                config.llm.base_url.clone(),
                config.llm.timeout,
            )
            .map_err(|err| ConfigError::HttpClient(err.to_string()))?;
            Arc::new(client)
        }
    };

    // Input-guard topic vocabulary: the mock corpus knows its own domain;
    // the HTTP and embedded-OREO backends index whatever corpus they were
    // given, which config cannot know, so off-topic checking is disabled
    // (empty vocabulary) until a real corpus manifest supplies it.
    let guards = Arc::new(GuardService::new(match config.retrieval.mode {
        RetrievalMode::Mock => MockRetrievalClient::topic_vocabulary(),
        RetrievalMode::Http | RetrievalMode::Oreo => Vec::new(),
    }));

    tracing::info!(
        retrieval = retrieval.name(),
        stt = stt.name(),
        llm = llm.name(),
        retrieval_timeout_ms = config.retrieval.timeout.as_millis() as u64,
        stt_timeout_ms = config.stt.sarvam_timeout.as_millis() as u64,
        llm_timeout_ms = config.llm.timeout.as_millis() as u64,
        grounding_min_score = config.grounding.min_score,
        "backends selected"
    );

    let pipeline = VoxPipeline::new(
        Arc::clone(&stt),
        Arc::clone(&retrieval),
        Arc::clone(&llm),
        config.grounding,
        guards,
    );

    Ok(AppState {
        pipeline: Arc::new(pipeline),
        retrieval,
        started_at: std::time::Instant::now(),
    })
}
