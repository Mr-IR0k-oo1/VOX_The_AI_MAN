//! Answer-generation substitution boundary for VOX.
//!
//! The pipeline depends only on [`LlmProvider`]; concrete providers live
//! behind this trait and the pipeline never talks to an LLM SDK directly.
//! Prompt construction ([`prompt::build_prompt`]) is shared so every provider
//! receives the same grounding instructions.

pub mod extractive;
pub mod openai;
pub mod prompt;

pub use crate::extractive::ExtractiveProvider;
pub use crate::openai::OpenAiCompatibleProvider;
pub use crate::prompt::{build_prompt, Prompt};
use async_trait::async_trait;
use reqwest::StatusCode;
use thiserror::Error;
use vox_types::{Language, RetrievedDocument};

/// Maximum attempts per call: the initial try plus one retry.
pub const MAX_RETRIES: u32 = 1;

/// Structured input to the generation stage.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    /// The (normalized) question to answer.
    pub question: String,
    /// Language the answer must be written in.
    pub language: Language,
    /// Retrieved evidence the answer must be grounded in.
    pub evidence: Vec<RetrievedDocument>,
}

impl LlmRequest {
    /// Builds a generation request from pipeline state.
    #[must_use]
    pub fn new(
        question: impl Into<String>,
        language: Language,
        evidence: Vec<RetrievedDocument>,
    ) -> Self {
        Self {
            question: question.into(),
            language,
            evidence,
        }
    }
}

/// Structured output of the generation stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmResponse {
    /// Generated answer text. May be empty; callers treat that as a refusal
    /// signal until the guardrail phase lands.
    pub answer: String,
    /// Model identifier reported by the backend, when available.
    pub model: Option<String>,
}

/// Failures raised by an answer-generation backend.
#[derive(Debug, Error)]
pub enum LlmError {
    /// The backend failed to produce any output.
    #[error("llm backend failed: {0}")]
    Backend(String),
    /// The backend call exceeded its timeout.
    #[error("llm request timed out after {timeout_ms} ms")]
    Timeout {
        /// Configured timeout in milliseconds.
        timeout_ms: u64,
    },
    /// The backend answered with a non-success status.
    #[error("llm backend returned HTTP {status}")]
    HttpStatus {
        /// Upstream status code.
        status: StatusCode,
    },
    /// Connection, DNS, TLS, or body-transport failure.
    #[error("network error contacting llm backend: {0}")]
    Network(#[from] reqwest::Error),
    /// The backend answered 2xx but the body violates the contract.
    #[error("invalid response from llm backend: {0}")]
    InvalidResponse(&'static str),
}

impl LlmError {
    /// Whether retrying the exact same request can plausibly succeed.
    ///
    /// Only safe transient conditions are retried: network failures,
    /// timeouts, upstream 5xx, and 429. Client errors and contract
    /// violations are not.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            LlmError::Network(_) | LlmError::Timeout { .. } => true,
            LlmError::HttpStatus { status } => {
                status.is_server_error() || *status == StatusCode::TOO_MANY_REQUESTS
            }
            LlmError::Backend(_) | LlmError::InvalidResponse(_) => false,
        }
    }
}

/// Substitution boundary for answer generation.
///
/// Implementations must be safe to share across threads and should apply
/// their own timeout and bounded retry policy.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Backend name used in logs and metrics.
    fn name(&self) -> &'static str;

    /// Generates a grounded answer for `request`.
    ///
    /// Implementations receive already-validated evidence and must not invent
    /// facts beyond it; grounding enforcement stays with the caller.
    ///
    /// # Errors
    /// Returns [`LlmError`] when the backend fails to produce output. An
    /// empty answer is reported as success and handled downstream as a
    /// refusal signal.
    async fn generate(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError>;
}
