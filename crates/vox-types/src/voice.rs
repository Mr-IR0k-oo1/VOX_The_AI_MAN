//! Voice ingestion contracts (`POST /v1/voice/query`).

use serde::{Deserialize, Serialize};

use crate::audio::AudioFormat;
use crate::error::{ValidationError, MAX_TOP_K};
use crate::language::Language;
use crate::latency::LatencyMetrics;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    /// Recognized text.
    pub text: String,
    /// Language the transcript was recognized in.
    pub language: Language,
    /// Recognizer confidence in `[0, 1]`, when reported by the backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// Response body of `POST /v1/voice/query`.
#[derive(Debug, Clone, Serialize)]
pub struct VoiceResponse {
    /// What was heard.
    pub transcript: Transcript,
    /// Grounded answer text; empty when the request was refused.
    pub answer: String,
    /// Whether the answer is grounded in retrieved evidence.
    pub grounded: bool,
    /// Machine-readable refusal reason; present only when `grounded` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal_reason: Option<String>,
    /// Per-stage latencies in milliseconds (includes the `stt` stage).
    pub timings_ms: LatencyMetrics,
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
        assert_eq!(
            request("   ", 5).validate(),
            Err(ValidationError::EmptyAudio)
        );
    }

    #[test]
    fn validate_should_reject_out_of_range_top_k() {
        assert_eq!(
            request("aGk=", 0).validate(),
            Err(ValidationError::InvalidTopK(0))
        );
    }

    #[test]
    fn deserialization_should_default_optional_fields() {
        let req: VoiceRequest =
            serde_json::from_str(r#"{"audio_base64":"aGk=","format":"mp3"}"#).expect("deserialize");
        assert_eq!(req.language, None);
        assert_eq!(req.top_k, 5);
    }
}
