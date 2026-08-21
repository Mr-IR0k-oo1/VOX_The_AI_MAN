//! Pipeline orchestration errors.

use thiserror::Error;
use vox_llm::LlmError;
use vox_retrieval::RetrievalError;
use vox_types::ValidationError;

/// Failures that abort a pipeline run.
///
/// A refusal (insufficient evidence, empty generation) is *not* an error: it
/// is a successful run whose [`vox_types::AnswerResponse`] carries
/// `grounded = false` plus a reason.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Input failed validation.
    #[error(transparent)]
    Validation(#[from] ValidationError),
    /// The retrieval boundary failed after retries.
    #[error(transparent)]
    Retrieval(#[from] RetrievalError),
    /// The generation backend failed.
    #[error(transparent)]
    Llm(#[from] LlmError),
}
