//! Language codes recognized by VOX.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// Languages VOX currently accepts, identified by ISO 639-1 code.
///
/// The set mirrors the Indian languages targeted by the STT provider; extend
/// deliberately as STT/retrieval coverage grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    /// Assamese (`as`)
    As,
    /// Bengali (`bn`)
    Bn,
    /// English (`en`)
    En,
    /// Gujarati (`gu`)
    Gu,
    /// Hindi (`hi`)
    Hi,
    /// Kannada (`kn`)
    Kn,
    /// Malayalam (`ml`)
    Ml,
    /// Marathi (`mr`)
    Mr,
    /// Odia (`or`)
    Or,
    /// Punjabi (`pa`)
    Pa,
    /// Tamil (`ta`)
    Ta,
    /// Telugu (`te`)
    Te,
    /// Urdu (`ur`)
    Ur,
}

impl Language {
    /// Every language VOX accepts.
    pub const ALL: [Language; 13] = [
        Language::As,
        Language::Bn,
        Language::En,
        Language::Gu,
        Language::Hi,
        Language::Kn,
        Language::Ml,
        Language::Mr,
        Language::Or,
        Language::Pa,
        Language::Ta,
        Language::Te,
        Language::Ur,
    ];

    /// The ISO 639-1 code for this language (the wire representation).
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Language::As => "as",
            Language::Bn => "bn",
            Language::En => "en",
            Language::Gu => "gu",
            Language::Hi => "hi",
            Language::Kn => "kn",
            Language::Ml => "ml",
            Language::Mr => "mr",
            Language::Or => "or",
            Language::Pa => "pa",
            Language::Ta => "ta",
            Language::Te => "te",
            Language::Ur => "ur",
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl FromStr for Language {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Language::ALL
            .into_iter()
            .find(|lang| lang.code() == s)
            .ok_or_else(|| CoreError::UnsupportedLanguage(s.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_should_accept_known_codes_case_sensitively() {
        assert_eq!("ta".parse::<Language>(), Ok(Language::Ta));
        assert_eq!("hi".parse::<Language>(), Ok(Language::Hi));
    }

    #[test]
    fn parse_should_reject_unknown_codes_with_the_offending_value() {
        assert_eq!(
            "xx".parse::<Language>(),
            Err(CoreError::UnsupportedLanguage("xx".to_owned()))
        );
    }

    #[test]
    fn serde_should_roundtrip_the_wire_code() {
        let json = serde_json::to_value(Language::Ta).expect("serialize");
        assert_eq!(json, serde_json::json!("ta"));
        let parsed: Language = serde_json::from_value(json).expect("deserialize");
        assert_eq!(parsed, Language::Ta);
    }

    #[test]
    fn display_should_match_code() {
        assert_eq!(Language::Ur.to_string(), "ur");
    }
}
