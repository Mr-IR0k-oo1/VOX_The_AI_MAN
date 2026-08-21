//! Validation errors for externally supplied input.

use thiserror::Error;

use crate::types::MAX_TOP_K;

/// Errors produced when validating external input against domain rules.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    /// The query string is missing or only whitespace.
    #[error("query must be a non-empty string")]
    EmptyQuery,
    /// `top_k` is outside the accepted range `1..={MAX_TOP_K}`.
    #[error("top_k must be between 1 and {MAX_TOP_K}, got {0}")]
    InvalidTopK(u8),
    /// The language code is not one VOX recognizes.
    #[error("unsupported language code: {0}")]
    UnsupportedLanguage(String),
}
