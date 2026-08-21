//! Voice ingestion boundary: decode and sanity-check uploaded audio before
//! it reaches a speech recognizer.
//!
//! Phase 0 scope: base64 decoding, emptiness and size-cap enforcement, and
//! best-effort container sniffing. Real transcoding/streaming lands with the
//! voice ingestion phase.

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use thiserror::Error;
use vox_types::AudioFormat;

/// Hard cap on decoded audio size accepted per request (10 MiB).
pub const MAX_AUDIO_BYTES: usize = 10 * 1024 * 1024;

/// Failures raised while ingesting a voice payload.
#[derive(Debug, Error)]
pub enum IngestError {
    /// The payload is not valid base64.
    #[error("audio payload is not valid base64: {0}")]
    Base64(String),
    /// The payload decoded to zero bytes.
    #[error("audio payload is empty")]
    EmptyAudio,
    /// The decoded payload exceeds [`MAX_AUDIO_BYTES`].
    #[error("audio payload too large: {size_bytes} bytes (max {max_bytes})")]
    TooLarge {
        /// Decoded size in bytes.
        size_bytes: usize,
        /// Configured cap in bytes.
        max_bytes: usize,
    },
}

/// Validated audio ready for transcription.
#[derive(Debug, Clone)]
pub struct AudioInput {
    /// Declared container format.
    pub format: AudioFormat,
    /// Decoded audio bytes.
    pub bytes: Vec<u8>,
}

/// Decodes and validates a base64-encoded audio payload.
///
/// # Errors
/// - [`IngestError::Base64`] when the payload is not valid base64.
/// - [`IngestError::EmptyAudio`] when it decodes to zero bytes.
/// - [`IngestError::TooLarge`] when it exceeds [`MAX_AUDIO_BYTES`].
pub fn decode_audio(audio_base64: &str, format: AudioFormat) -> Result<AudioInput, IngestError> {
    let bytes = STANDARD
        .decode(audio_base64.trim())
        .map_err(|err| IngestError::Base64(err.to_string()))?;
    if bytes.is_empty() {
        return Err(IngestError::EmptyAudio);
    }
    let size_bytes = bytes.len();
    if size_bytes > MAX_AUDIO_BYTES {
        return Err(IngestError::TooLarge {
            size_bytes,
            max_bytes: MAX_AUDIO_BYTES,
        });
    }
    Ok(AudioInput { format, bytes })
}

/// Best-effort container sniffing from magic bytes.
///
/// Used for diagnostics and future auto-detection; the declared
/// [`AudioFormat`] remains authoritative in Phase 0.
#[must_use]
pub fn detect_format(bytes: &[u8]) -> Option<AudioFormat> {
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE" {
        Some(AudioFormat::Wav)
    } else if bytes.starts_with(b"fLaC") {
        Some(AudioFormat::Flac)
    } else if bytes.starts_with(b"ID3")
        || (bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0)
    {
        Some(AudioFormat::Mp3)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_should_roundtrip_base64_payloads() {
        let input = decode_audio("aGVsbG8=", AudioFormat::Wav).expect("decode");
        assert_eq!(input.bytes, b"hello");
        assert_eq!(input.format, AudioFormat::Wav);
    }

    #[test]
    fn decode_should_reject_invalid_base64() {
        let err = decode_audio("!!!not-base64!!!", AudioFormat::Mp3).expect_err("must reject");
        assert!(matches!(err, IngestError::Base64(_)));
    }

    #[test]
    fn decode_should_reject_empty_payloads() {
        // '=' alone decodes to zero bytes.
        let err = decode_audio("", AudioFormat::Flac).expect_err("must reject");
        assert!(matches!(err, IngestError::EmptyAudio));
    }

    #[test]
    fn decode_should_enforce_the_size_cap() {
        let oversized = vec![0u8; MAX_AUDIO_BYTES + 1];
        let encoded = STANDARD.encode(oversized);
        let err = decode_audio(&encoded, AudioFormat::Wav).expect_err("must reject");
        assert!(matches!(
            err,
            IngestError::TooLarge {
                max_bytes: MAX_AUDIO_BYTES,
                ..
            }
        ));
    }

    #[test]
    fn detect_should_recognize_wav_flac_and_mp3_sync() {
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(b"xxxxWAVE");
        assert_eq!(detect_format(&wav), Some(AudioFormat::Wav));
        assert_eq!(detect_format(b"fLaC..."), Some(AudioFormat::Flac));
        assert_eq!(detect_format(b"ID3\x04"), Some(AudioFormat::Mp3));
        assert_eq!(detect_format(&[0xFF, 0xFB, 0x90]), Some(AudioFormat::Mp3));
        assert_eq!(detect_format(b"unknown"), None);
        assert_eq!(detect_format(&[]), None);
    }
}
