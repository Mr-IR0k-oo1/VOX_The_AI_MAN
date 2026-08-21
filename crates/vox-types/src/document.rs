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
}
