//! Validation errors for externally supplied input.

use thiserror::Error;

/// Upper bound for `top_k` on any request.
pub const MAX_TOP_K: u8 = 20;

/// Errors produced when validating external input against domain rules.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidationError {
    /// The query string is missing or only whitespace.
    #[error("query must be a non-empty string")]
    EmptyQuery,
    /// The audio payload is missing or only whitespace.
    #[error("audio payload must be a non-empty base64 string")]
    EmptyAudio,
    /// `top_k` is outside the accepted range `1..={MAX_TOP_K}`.
    #[error("top_k must be between 1 and {MAX_TOP_K}, got {0}")]
    InvalidTopK(u8),
    /// The language code is not one VOX recognizes.
    #[error("unsupported language code: {0}")]
    UnsupportedLanguage(String),
    /// The audio format token is not one VOX recognizes.
    #[error("unsupported audio format: {0}")]
    UnsupportedAudioFormat(String),
}
