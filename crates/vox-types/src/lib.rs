//! Shared domain types for VOX: language codes, audio formats, the query and
//! retrieval contracts, voice ingestion contracts, answer contracts, stage
//! verdicts, per-stage latency metrics, and validation errors.
//!
//! These types are the single source of truth for the wire contracts shared
//! by every VOX crate. They carry no behavior beyond validation of their own
//! invariants.

pub mod answer;
pub mod audio;
pub mod decision;
pub mod document;
pub mod error;
pub mod language;
pub mod latency;
pub mod query;
pub mod voice;

pub use answer::AnswerResponse;
pub use audio::AudioFormat;
pub use decision::{Answerability, GuardrailDecision};
pub use document::{RetrievalResponse, RetrievalTimings, RetrievedDocument};
pub use error::{ValidationError, MAX_TOP_K};
pub use language::Language;
pub use latency::{ms, LatencyMetrics};
pub use query::{Query, QueryIntent};
pub use voice::{Transcript, VoiceRequest, VoiceResponse};
