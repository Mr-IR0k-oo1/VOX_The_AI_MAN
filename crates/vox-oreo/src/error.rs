//! Error type for the OREO retrieval engine.

use thiserror::Error;
use vox_types::ValidationError;

/// Failures raised by ingestion, indexing, or retrieval inside OREO.
#[derive(Debug, Error)]
pub enum OreoError {
    /// The request violated input rules and was never processed.
    #[error(transparent)]
    Validation(#[from] ValidationError),
    /// A configuration value is missing or contradictory.
    #[error("invalid retrieval engine configuration: {0}")]
    Config(String),
    /// Filesystem failure while reading a corpus or opening an index.
    #[error("io failure: {0}")]
    Io(#[from] std::io::Error),
    /// The corpus could not be parsed.
    #[error("corpus parse failure at {location}: {message}")]
    Corpus {
        /// Where the failure happened (file, line).
        location: String,
        /// Underlying parse message.
        message: String,
    },
    /// The full-text index failed.
    #[error("bm25 index failure: {0}")]
    Index(String),
    /// The vector store answered with a non-success status.
    #[error("vector store returned HTTP {status}: {body}")]
    VectorStore {
        /// Upstream status code.
        status: u16,
        /// Response body (truncated by the caller).
        body: String,
    },
    /// Connection/transport failure against the vector store.
    #[error("vector store network error: {0}")]
    VectorStoreNetwork(String),
}

impl From<tantivy::TantivyError> for OreoError {
    fn from(err: tantivy::TantivyError) -> Self {
        Self::Index(err.to_string())
    }
}
