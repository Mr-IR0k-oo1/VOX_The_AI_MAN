//! Structured pipeline responses for the v1 endpoints.

use serde::Serialize;

use crate::decision::Answerability;
use crate::document::RetrievedDocument;
use crate::language::Language;
use crate::latency::LatencyMetrics;
use crate::query::Query;

/// Response body of `POST /v1/query`.
#[derive(Debug, Clone, Serialize)]
pub struct QueryResponse {
    /// Server-generated identifier for this request.
    pub request_id: String,
    /// Resolved pipeline language (hint → script fallback).
    pub language: Language,
    /// The analyzed query (normalized text + intent).
    pub query: Query,
    /// Mock/sample evidence retrieved for the query.
    pub evidence: Vec<RetrievedDocument>,
    /// Evidence-sufficiency verdict from the grounding stage.
    pub answerability: Answerability,
    /// Generated answer; absent when generation produced nothing usable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    /// Machine-readable refusal reason; present exactly when the pipeline
    /// refused to answer instead of generating.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal_reason: Option<String>,
    /// Per-stage latencies in milliseconds plus total.
    pub metrics: LatencyMetrics,
}
