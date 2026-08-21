//! Pipeline orchestration errors.

use thiserror::Error;
use vox_core::CoreError;
use vox_retrieval::RetrievalError;

/// Failures that abort a pipeline run.
///
/// A refusal (insufficient evidence, empty generation) is *not* an error: it
/// is a successful run whose [`vox_core::AnswerResponse`] carries
/// `grounded = false` plus a reason.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Input failed validation.
    #[error(transparent)]
    Validation(#[from] CoreError),
    /// The retrieval boundary failed after retries.
    #[error(transparent)]
    Retrieval(#[from] RetrievalError),
    /// The generation backend failed.
    #[error("llm generation failed: {0}")]
    Llm(String),
}
