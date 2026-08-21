//! Error type for the retrieval boundary.

use thiserror::Error;
use vox_core::CoreError;

/// Failures that can occur when talking to a retrieval backend.
#[derive(Debug, Error)]
pub enum RetrievalError {
    /// The request violated input rules and was never sent upstream.
    #[error(transparent)]
    Validation(#[from] CoreError),
    /// The upstream call exceeded its per-attempt timeout.
    #[error("retrieval request timed out after {timeout_ms} ms")]
    Timeout {
        /// Configured timeout in milliseconds.
        timeout_ms: u64,
    },
    /// The upstream service answered with a non-success status.
    #[error("retrieval service returned HTTP {status}")]
    HttpStatus {
        /// Upstream status code.
        status: reqwest::StatusCode,
    },
    /// Connection, DNS, TLS, or body-transport failure.
    #[error("network error contacting retrieval service: {0}")]
    Network(#[from] reqwest::Error),
    /// The upstream answered 2xx but the body violates the contract.
    #[error("invalid response from retrieval service: {0}")]
    InvalidResponse(&'static str),
}

impl RetrievalError {
    /// Whether retrying the exact same request can plausibly succeed.
    ///
    /// Only transient conditions are retryable: network failures, timeouts,
    /// upstream 5xx, and 429. Client errors and contract violations are not.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            RetrievalError::Network(_) | RetrievalError::Timeout { .. } => true,
            RetrievalError::HttpStatus { status } => {
                status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS
            }
            RetrievalError::Validation(_) | RetrievalError::InvalidResponse(_) => false,
        }
    }
}
