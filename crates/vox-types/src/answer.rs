//! User-facing answer contracts (`POST /v1/query`).

use serde::Serialize;

use crate::language::Language;
use crate::latency::LatencyMetrics;

/// Response body of `POST /v1/query`.
#[derive(Debug, Clone, Serialize)]
pub struct AnswerResponse {
    /// The original query text, echoed back.
    pub query: String,
    /// Language the query was processed in.
    pub language: Language,
    /// Grounded answer text; empty when the request was refused.
    pub answer: String,
    /// Whether the answer is grounded in retrieved evidence.
    pub grounded: bool,
    /// Machine-readable refusal reason; present only when `grounded` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal_reason: Option<String>,
    /// Per-stage latencies in milliseconds.
    pub timings_ms: LatencyMetrics,
}
