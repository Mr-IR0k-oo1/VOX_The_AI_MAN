//! Wire contracts shared between the API, pipeline, and retrieval boundary.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::CoreError;
use crate::language::Language;
use crate::timings::StageTimings;

/// Upper bound for `top_k` on any request.
pub const MAX_TOP_K: u8 = 20;

/// Request body shared by `POST /v1/query` and `POST /v1/retrieve`.
///
/// The identical shape is deliberate: the user-facing pipeline forwards the
/// (normalized) query to the retrieval service over this exact contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrieveRequest {
    /// The query text.
    pub query: String,
    /// Expected language of the query.
    pub language: Language,
    /// Maximum number of chunks to return.
    #[serde(default = "default_top_k")]
    pub top_k: u8,
}

fn default_top_k() -> u8 {
    5
}

impl RetrieveRequest {
    /// Validates externally supplied input.
    ///
    /// # Errors
    /// - [`CoreError::EmptyQuery`] when `query` is missing or whitespace-only.
    /// - [`CoreError::InvalidTopK`] when `top_k` is `0` or above [`MAX_TOP_K`].
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.query.trim().is_empty() {
            return Err(CoreError::EmptyQuery);
        }
        if self.top_k == 0 || self.top_k > MAX_TOP_K {
            return Err(CoreError::InvalidTopK(self.top_k));
        }
        Ok(())
    }
}

/// One retrieved evidence chunk, as returned by `POST /v1/retrieve`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedChunk {
    /// Stable chunk identifier from the retrieval service.
    pub id: String,
    /// Chunk text.
    pub text: String,
    /// Relevance score; must be finite (`NaN`/infinite scores are rejected).
    pub score: f32,
    /// 1-based rank after fusion/reranking.
    pub rank: u32,
    /// Free-form backend metadata (source document, section, ...).
    #[serde(default)]
    pub metadata: Value,
}

/// Response body of `POST /v1/retrieve`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrieveResponse {
    /// Ranked evidence chunks; may be empty when nothing matches.
    pub results: Vec<RetrievedChunk>,
}

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
    pub timings_ms: StageTimings,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(query: &str, top_k: u8) -> RetrieveRequest {
        RetrieveRequest {
            query: query.to_owned(),
            language: Language::Ta,
            top_k,
        }
    }

    #[test]
    fn validate_should_accept_a_well_formed_request() {
        assert_eq!(request("what is gst", 5).validate(), Ok(()));
    }

    #[test]
    fn validate_should_reject_whitespace_only_queries() {
        assert_eq!(
            request("   \n\t", 5).validate(),
            Err(CoreError::EmptyQuery)
        );
    }

    #[test]
    fn validate_should_reject_top_k_above_the_limit() {
        assert_eq!(
            request("q", MAX_TOP_K + 1).validate(),
            Err(CoreError::InvalidTopK(MAX_TOP_K + 1))
        );
    }

    #[test]
    fn validate_should_reject_zero_top_k() {
        assert_eq!(request("q", 0).validate(), Err(CoreError::InvalidTopK(0)));
    }

    #[test]
    fn deserialization_should_default_top_k_to_five() {
        let req: RetrieveRequest =
            serde_json::from_str(r#"{"query":"q","language":"bn"}"#).expect("deserialize");
        assert_eq!(req.top_k, 5);
    }
}
