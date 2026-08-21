//! Shared application state for request handlers.

use std::sync::Arc;
use std::time::Instant;

use vox_core::VoiceRagPipeline;
use vox_retrieval::RetrievalClient;
use vox_stt::SpeechRecognizer;

/// State shared with all Axum handlers.
pub struct AppState {
    /// The user-facing text pipeline.
    pub pipeline: Arc<VoiceRagPipeline>,
    /// Retrieval boundary, also served directly at `POST /v1/retrieve`.
    pub retrieval: Arc<dyn RetrievalClient>,
    /// Speech-to-text boundary for the voice flow.
    pub stt: Arc<dyn SpeechRecognizer>,
    /// Process start time, used by `/health`.
    pub started_at: Instant,
}
