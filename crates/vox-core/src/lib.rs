//! Shared domain types for VOX: language codes, the retrieval request/response
//! contract, the user-facing answer contract, per-stage timings, and
//! validation errors.
//!
//! These types are the single source of truth for the wire contracts shared by
//! the API, pipeline, and retrieval client crates.

pub mod error;
pub mod language;
pub mod timings;
pub mod types;

pub use error::CoreError;
pub use language::Language;
pub use timings::{ms, StageTimings};
pub use types::{AnswerResponse, RetrievedChunk, RetrieveRequest, RetrieveResponse, MAX_TOP_K};
