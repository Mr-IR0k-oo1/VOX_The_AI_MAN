//! Shared application state for request handlers.

use std::time::Instant;

use vox_pipeline::VoiceRagPipeline;

/// State shared with all Axum handlers.
pub struct AppState {
    /// The user-facing pipeline.
    pub pipeline: std::sync::Arc<VoiceRagPipeline>,
    /// Process start time, used by `/health`.
    pub started_at: Instant,
}
