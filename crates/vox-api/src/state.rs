//! Shared application state for request handlers.

use std::sync::Arc;
use std::time::Instant;

use vox_core::VoxPipeline;

/// State shared with all Axum handlers.
pub struct AppState {
    /// The full pipeline (STT → language → analysis → retrieval).
    pub pipeline: Arc<VoxPipeline>,
    /// Retrieval boundary, also served directly at `POST /v1/retrieve`.
    pub retrieval: Arc<dyn vox_retrieval::RetrievalClient>,
    /// Process start time, used by `/health`.
    pub started_at: Instant,
}
