//! Voice ingestion contracts (`POST /v1/voice/query`).

use serde::{Deserialize, Serialize};

use crate::audio::AudioFormat;
use crate::decision::Answerability;
use crate::document::RetrievedDocument;
use crate::error::{ValidationError, MAX_TOP_K};
use crate::language::Language;
use crate::latency::LatencyMetrics;
use crate::query::Query;

/// Request body of `POST /v1/voice/query`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceRequest {
    /// Raw audio bytes, base64-encoded.
    pub audio_base64: String,
    /// Container format of the audio payload.
    pub format: AudioFormat,
    /// Optional language hint for the recognizer; detected when absent.
    #[serde(default)]
    pub language: Option<Language>,
    /// Maximum number of documents to retrieve for the transcribed query.
    #[serde(default = "default_top_k")]
    pub top_k: u8,
}

fn default_top_k() -> u8 {
    5
}

impl VoiceRequest {
    /// Validates externally supplied input before any decoding happens.
    ///
    /// # Errors
    /// - [`ValidationError::EmptyAudio`] when `audio_base64` is missing or
    ///   whitespace-only.
    /// - [`ValidationError::InvalidTopK`] when `top_k` is `0` or above
    ///   [`MAX_TOP_K`].
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.audio_base64.trim().is_empty() {
            return Err(ValidationError::EmptyAudio);
        }
        if self.top_k == 0 || self.top_k > MAX_TOP_K {
            return Err(ValidationError::InvalidTopK(self.top_k));
        }
        Ok(())
    }
}

/// Output of the speech-to-text stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    /// Recognized text.
    pub text: String,
    /// Language identified by the recognizer, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_language: Option<Language>,
    /// Recognizer confidence in `[0, 1]`, when reported by the backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl Transcript {
    /// Creates a transcript without detection metadata.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            detected_language: None,
            confidence: None,
        }
    }
}

/// Response body of `POST /v1/voice/query`.
#[derive(Debug, Clone, Serialize)]
pub struct VoiceResponse {
    /// Server-generated identifier for this request.
    pub request_id: String,
    /// What was heard.
    pub transcript: Transcript,
    /// Resolved pipeline language (hint → STT → script fallback).
    pub language: Language,
    /// The analyzed query (normalized text + intent).
    pub query: Query,
    /// Mock/sample evidence retrieved for the query.
    pub evidence: Vec<RetrievedDocument>,
    /// Evidence-sufficiency verdict from the grounding stage.
    pub answerability: Answerability,
    /// Generated answer; absent when generation produced nothing usable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    /// Machine-readable refusal reason; present exactly when the pipeline
    /// refused to answer instead of generating.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal_reason: Option<String>,
    /// Per-stage latencies in milliseconds (includes `stt`) plus total.
    pub metrics: LatencyMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(audio: &str, top_k: u8) -> VoiceRequest {
        VoiceRequest {
            audio_base64: audio.to_owned(),
            format: AudioFormat::Wav,
            language: Some(Language::Hi),
            top_k,
        }
    }

    #[test]
    fn validate_should_accept_a_well_formed_request() {
        assert_eq!(request("aGVsbG8=", 5).validate(), Ok(()));
    }

    #[test]
    fn validate_should_reject_blank_audio() {
        let req = request("   ", 5);
        assert_eq!(req.validate(), Err(ValidationError::EmptyAudio));
    }

    #[test]
    fn validate_should_reject_out_of_range_top_k() {
        let req = request("aGk=", 0);
        assert_eq!(req.validate(), Err(ValidationError::InvalidTopK(0)));
    }

    #[test]
    fn deserialization_should_default_optional_fields() {
        let req: VoiceRequest =
            serde_json::from_str(r#"{"audio_base64":"aGk=","format":"mp3"}"#).expect("deserialize");
        assert_eq!(req.language, None);
        assert_eq!(req.top_k, 5);
    }

    #[test]
    fn transcript_should_omit_unset_metadata() {
        let transcript = Transcript::new("hello");
        let json = serde_json::to_value(&transcript).expect("serialize");
        assert_eq!(json, serde_json::json!({ "text": "hello" }));
    }
}
