//! Retrieval boundary contracts (`POST /v1/retrieve`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One retrieved evidence document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedDocument {
    /// Stable document/chunk identifier from the retrieval backend.
    pub id: String,
    /// Document text.
    pub text: String,
    /// Relevance score; must be finite (`NaN`/infinite scores are rejected).
    pub score: f32,
    /// 1-based rank after fusion/reranking.
    pub rank: u32,
    /// Free-form backend metadata (source document, section, language, ...).
    #[serde(default)]
    pub metadata: Value,
}

/// Response body of `POST /v1/retrieve`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalResponse {
    /// Ranked evidence documents; may be empty when nothing matches.
    pub documents: Vec<RetrievedDocument>,
    /// Retrieval-internal stage timings in milliseconds; omitted when the
    /// backend does not expose them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timings_ms: Option<RetrievalTimings>,
}

/// Latency of each retrieval-engine stage in milliseconds.
///
/// Stages that did not run are omitted from serialization rather than
/// reported as zero, mirroring [`crate::latency::LatencyMetrics`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrievalTimings {
    /// Query embedding duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed: Option<f64>,
    /// Dense (vector) search duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense: Option<f64>,
    /// Sparse (BM25) search duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse: Option<f64>,
    /// Reciprocal Rank Fusion duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fuse: Option<f64>,
    /// Reranking duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<f64>,
    /// Total time inside the retrieval engine.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialization_should_default_metadata_to_null() {
        let doc: RetrievedDocument =
            serde_json::from_str(r#"{"id":"d1","text":"t","score":0.5,"rank":1}"#)
                .expect("deserialize");
        assert_eq!(doc.metadata, Value::Null);
    }

    #[test]
    fn retrieval_response_should_default_to_empty() {
        let response = RetrievalResponse::default();
        assert!(response.documents.is_empty());
        assert_eq!(
            serde_json::to_value(&response).expect("serialize"),
            serde_json::json!({ "documents": [] })
        );
    }

    #[test]
    fn unset_timings_should_be_omitted_from_json() {
        let response = RetrievalResponse::default();
        let json = serde_json::to_value(&response).expect("serialize");
        assert!(json.get("timings_ms").is_none());
    }

    #[test]
    fn set_timings_should_serialize_stage_splits() {
        let response = RetrievalResponse {
            timings_ms: Some(RetrievalTimings {
                embed: Some(1.0),
                total: Some(4.5),
                ..RetrievalTimings::default()
            }),
            ..RetrievalResponse::default()
        };
        let json = serde_json::to_value(&response).expect("serialize");
        assert_eq!(
            json["timings_ms"],
            serde_json::json!({ "embed": 1.0, "total": 4.5 })
        );
    }
}
