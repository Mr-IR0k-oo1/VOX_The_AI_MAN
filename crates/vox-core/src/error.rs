//! Pipeline orchestration errors.

use thiserror::Error;
use vox_ingest::IngestError;
use vox_llm::LlmError;
use vox_retrieval::RetrievalError;
use vox_stt::SttError;
use vox_types::ValidationError;

/// Failures that abort a pipeline run.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Input failed validation.
    #[error(transparent)]
    Validation(#[from] ValidationError),
    /// The voice payload failed ingestion checks.
    #[error(transparent)]
    Ingest(#[from] IngestError),
    /// The speech-to-text boundary failed.
    #[error(transparent)]
    Stt(#[from] SttError),
    /// The retrieval boundary failed after retries.
    #[error(transparent)]
    Retrieval(#[from] RetrievalError),
    /// The answer-generation boundary failed after retries.
    #[error(transparent)]
    Llm(#[from] LlmError),
}
