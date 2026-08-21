//! Per-request state carried through the pipeline stages.

use vox_types::{
    Answerability, Language, LatencyMetrics, Query, QueryResponse, RetrievedDocument, Transcript,
};

/// Everything the pipeline knows about the request it is processing.
///
/// Built up stage by stage (STT → language → query analysis → retrieval →
/// grounding preparation → generation) and converted into the endpoint
/// response at the end.
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Server-generated identifier for this request.
    pub request_id: String,
    /// What was heard; present on the voice flow only.
    pub transcript: Option<Transcript>,
    /// Resolved pipeline language.
    pub language: Language,
    /// The analyzed query (normalized text + intent).
    pub query: Query,
    /// Evidence returned by the retrieval boundary.
    pub evidence: Vec<RetrievedDocument>,
    /// Preliminary evidence-sufficiency verdict (grounding preparation).
    pub answerability: Answerability,
    /// Generated answer, when the LLM produced usable output.
    pub answer: Option<String>,
    /// Per-stage latency measurements (snapshot up to generation).
    pub metrics: LatencyMetrics,
}

impl PipelineContext {
    /// Converts a completed text-flow context into the endpoint response.
    #[must_use]
    pub fn into_query_response(self) -> QueryResponse {
        QueryResponse {
            request_id: self.request_id,
            language: self.language,
            query: self.query,
            evidence: self.evidence,
            answerability: self.answerability,
            answer: self.answer,
            metrics: self.metrics,
        }
    }

    /// Converts a completed voice-flow context into the endpoint response.
    ///
    /// `transcript` is the STT output carried through this flow; the voice
    /// contract always exposes it alongside the resolved language, analyzed
    /// query, evidence, answer, and metrics.
    #[must_use]
    pub fn into_voice_response(self, transcript: Transcript) -> vox_types::VoiceResponse {
        vox_types::VoiceResponse {
            request_id: self.request_id,
            transcript,
            language: self.language,
            query: self.query,
            evidence: self.evidence,
            answerability: self.answerability,
            answer: self.answer,
            metrics: self.metrics,
        }
    }
}
