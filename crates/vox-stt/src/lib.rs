//! Speech-to-text substitution boundary for VOX.
//!
//! The voice pipeline depends only on [`SpeechRecognizer`]; concrete engines
//! (e.g. a hosted Indic ASR service) plug in behind it. Phase 0 ships the
//! trait plus a stub that reports [`SttError::NotImplemented`].

use async_trait::async_trait;
use thiserror::Error;
use vox_types::{AudioFormat, Language, Transcript};

/// Failures raised by a speech-to-text backend.
#[derive(Debug, Error)]
pub enum SttError {
    /// No real recognizer is wired up yet.
    #[error("speech-to-text is not implemented yet")]
    NotImplemented,
    /// The recognizer backend failed.
    #[error("stt backend failed: {0}")]
    Backend(String),
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
    /// # Errors
    /// Returns [`SttError`] when the backend cannot produce a transcript.
    async fn transcribe(
        &self,
        audio: &[u8],
        format: AudioFormat,
        language: Option<Language>,
    ) -> Result<Transcript, SttError>;
}

/// Placeholder recognizer used until a real STT engine is integrated.
#[derive(Debug, Clone, Copy, Default)]
pub struct StubSpeechRecognizer;

#[async_trait]
impl SpeechRecognizer for StubSpeechRecognizer {
    fn name(&self) -> &'static str {
        "stub"
    }

    async fn transcribe(
        &self,
        _audio: &[u8],
        _format: AudioFormat,
        _language: Option<Language>,
    ) -> Result<Transcript, SttError> {
        Err(SttError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_should_report_not_implemented() {
        let recognizer = StubSpeechRecognizer;
        let err = recognizer
            .transcribe(b"audio", AudioFormat::Wav, Some(Language::Hi))
            .await
            .expect_err("stub must not transcribe");
        assert!(matches!(err, SttError::NotImplemented));
    }

    #[test]
    fn stub_should_report_its_name() {
        assert_eq!(StubSpeechRecognizer.name(), "stub");
    }
}
