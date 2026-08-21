//! Deterministic mock recognizer for tests and local development.

use async_trait::async_trait;
use vox_types::{AudioFormat, Language, Transcript};

use crate::{SpeechRecognizer, SttError};

/// Default transcript produced by [`MockRecognizer::new`].
pub const DEFAULT_TRANSCRIPT: &str = "This is a mock transcript for voice pipeline testing.";

/// Deterministic in-process recognizer.
///
/// Never touches a network and always succeeds, which makes pipeline tests
/// reproducible. The transcript is fixed at construction time; the audio
/// payload is ignored.
#[derive(Debug, Clone)]
pub struct MockRecognizer {
    transcript: Transcript,
}

impl MockRecognizer {
    /// Creates the default mock: an English test sentence with full
    /// confidence.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transcript: Transcript {
                text: DEFAULT_TRANSCRIPT.to_owned(),
                detected_language: Some(Language::En),
                confidence: Some(1.0),
            },
        }
    }

    /// Creates a mock that always returns `text`, reporting `language` (or
    /// none, to exercise downstream script detection) at full confidence.
    #[must_use]
    pub fn with_transcript(text: impl Into<String>, language: Option<Language>) -> Self {
        Self {
            transcript: Transcript {
                text: text.into(),
                detected_language: language,
                confidence: Some(1.0),
            },
        }
    }
}

impl Default for MockRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SpeechRecognizer for MockRecognizer {
    fn name(&self) -> &'static str {
        "mock"
    }

    async fn transcribe(
        &self,
        _audio: &[u8],
        _format: AudioFormat,
        _language: Option<Language>,
    ) -> Result<Transcript, SttError> {
        Ok(self.transcript.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn default_mock_should_return_the_english_test_sentence() {
        let recognizer = MockRecognizer::new();
        let transcript = recognizer
            .transcribe(b"audio", AudioFormat::Wav, None)
            .await
            .expect("transcribe");
        assert_eq!(transcript.text, DEFAULT_TRANSCRIPT);
        assert_eq!(transcript.detected_language, Some(Language::En));
        assert_eq!(transcript.confidence, Some(1.0));
    }

    #[tokio::test]
    async fn custom_mock_should_be_deterministic_and_may_report_no_language() {
        let recognizer = MockRecognizer::with_transcript("வணக்கம்", None);
        let first = recognizer
            .transcribe(b"a", AudioFormat::Mp3, Some(Language::Ta))
            .await
            .expect("first");
        let second = recognizer
            .transcribe(b"a", AudioFormat::Mp3, Some(Language::Ta))
            .await
            .expect("second");
        assert_eq!(first.text, "வணக்கம்");
        assert_eq!(first.detected_language, None);
        assert_eq!(first, second);
    }

    #[test]
    fn mock_should_report_its_name() {
        assert_eq!(MockRecognizer::new().name(), "mock");
    }
}
