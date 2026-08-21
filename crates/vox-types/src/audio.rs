//! Audio container formats accepted at the voice ingestion boundary.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::ValidationError;

/// Container formats VOX accepts for voice input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// RIFF WAVE (`wav`).
    Wav,
    /// MPEG layer-3 (`mp3`).
    Mp3,
    /// FLAC (`flac`).
    Flac,
}

impl AudioFormat {
    /// Every format VOX accepts.
    pub const ALL: [AudioFormat; 3] = [AudioFormat::Wav, AudioFormat::Mp3, AudioFormat::Flac];

    /// The canonical lowercase wire token for this format.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Flac => "flac",
        }
    }
}

impl fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AudioFormat {
    type Err = ValidationError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        AudioFormat::ALL
            .into_iter()
            .find(|format| format.as_str() == s)
            .ok_or_else(|| ValidationError::UnsupportedAudioFormat(s.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_should_roundtrip_the_wire_token() {
        let json = serde_json::to_value(AudioFormat::Wav).expect("serialize");
        assert_eq!(json, serde_json::json!("wav"));
        let parsed: AudioFormat = serde_json::from_value(json).expect("deserialize");
        assert_eq!(parsed, AudioFormat::Wav);
    }

    #[test]
    fn parse_should_reject_unknown_tokens() {
        assert_eq!(
            "ogg".parse::<AudioFormat>(),
            Err(ValidationError::UnsupportedAudioFormat("ogg".to_owned()))
        );
    }

    #[test]
    fn display_should_match_wire_token() {
        assert_eq!(AudioFormat::Flac.to_string(), "flac");
    }
}
