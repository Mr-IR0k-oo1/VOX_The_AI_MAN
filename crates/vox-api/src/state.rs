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
    /// Underlying embedded OREO engine (when running in `oreo` mode).
    pub oreo_engine: Option<Arc<vox_oreo::OreoEngine>>,
    /// Process start time, used by `/health`.
    pub started_at: Instant,
}

impl AppState {
    /// If backed by an embedded OREO engine, indexes the default multilingual sample corpus.
    ///
    /// # Errors
    /// Returns [`vox_oreo::OreoError`] if document indexing or vector insertion fails.
    pub async fn index_sample_corpus(&self) -> Result<(), vox_oreo::OreoError> {
        if let Some(engine) = &self.oreo_engine {
            engine
                .index_documents(vox_oreo::ingest::sample_corpus())
                .await?;
        }
        Ok(())
    }
}
