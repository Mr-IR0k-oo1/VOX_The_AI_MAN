//! Speech-to-text substitution boundary for VOX.
//!
//! The voice pipeline depends only on [`SpeechRecognizer`]; concrete engines
//! plug in behind it. Phase 1 ships a deterministic [`mock::MockRecognizer`]
//! for tests and local development, and a [`sarvam::SarvamRecognizer`] client
//! for Sarvam AI's `POST /speech-to-text` service.

pub mod mock;
pub mod sarvam;

use async_trait::async_trait;
use thiserror::Error;
use vox_types::{AudioFormat, Language, Transcript};

pub use crate::mock::MockRecognizer;
pub use crate::sarvam::SarvamRecognizer;

/// Failures raised by a speech-to-text backend.
#[derive(Debug, Error)]
pub enum SttError {
    /// No real recognizer is wired up yet.
    #[error("speech-to-text is not implemented yet")]
    NotImplemented,
    /// The upstream call exceeded its timeout.
    #[error("speech-to-text request timed out after {timeout_ms} ms")]
    Timeout {
        /// Configured timeout in milliseconds.
        timeout_ms: u64,
    },
    /// The upstream service answered with a non-success status.
    #[error("speech-to-text service returned HTTP {status}")]
    HttpStatus {
        /// Upstream status code.
        status: reqwest::StatusCode,
    },
    /// Connection, DNS, TLS, or body-transport failure.
    #[error("network error contacting speech-to-text service: {0}")]
    Network(#[from] reqwest::Error),
    /// The upstream answered 2xx but the body violates the contract.
    #[error("invalid response from speech-to-text service: {0}")]
    InvalidResponse(&'static str),
    /// The recognizer backend failed for a reason not covered above.
    #[error("stt backend failed: {0}")]
    Backend(String),
}

impl SttError {
    /// Whether retrying the exact same request can plausibly succeed.
    ///
    /// Only transient conditions are retryable: network failures, timeouts,
    /// upstream 5xx, and 429. Client errors and contract violations are not.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            SttError::Network(_) | SttError::Timeout { .. } => true,
            SttError::HttpStatus { status } => {
                status.is_server_error() || *status == reqwest::StatusCode::TOO_MANY_REQUESTS
            }
            SttError::NotImplemented | SttError::InvalidResponse(_) | SttError::Backend(_) => false,
        }
    }
}

/// Substitution boundary for speech-to-text engines.
///
/// Implementations must be safe to share across threads.
#[async_trait]
pub trait SpeechRecognizer: Send + Sync {
    /// Backend name used in logs and metrics.
    fn name(&self) -> &'static str;

    /// Transcribes `audio` bytes of `format`, optionally hinted toward
    /// `language`.
    ///
    /// Implementations report the language they detected via
    /// [`Transcript::detected_language`] when the engine provides one; the
    /// caller resolves the final pipeline language.
    ///
    /// # Errors
    /// Returns [`SttError`] when the backend cannot produce a transcript.
    async fn transcribe(
        &self,
        audio: &[u8],
        format: AudioFormat,
        language: Option<Language>,
    ) -> Result<Transcript, SttError>;
}
