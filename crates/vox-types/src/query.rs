//! The text query contract shared by `POST /v1/query` and `POST /v1/retrieve`.

use serde::{Deserialize, Serialize};

use crate::error::{ValidationError, MAX_TOP_K};
use crate::language::Language;

/// Coarse classification of what the caller wants done with a query.
///
/// Phase 0 freezes the wire shape only; intent detection itself lands with
/// query analysis. Until then every query defaults to [`QueryIntent::Unknown`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryIntent {
    /// No intent supplied or detected yet.
    #[default]
    Unknown,
    /// A fact is being requested ("What is GST?").
    Factual,
    /// A procedure is being requested ("How do I file an ITR?").
    Procedural,
    /// The query is outside the served domain.
    OffTopic,
}

/// Request body shared by `POST /v1/query` and `POST /v1/retrieve`.
///
/// The identical shape is deliberate: the user-facing pipeline forwards the
/// (normalized) query to the retrieval boundary over this exact contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// The query text.
    pub query: String,
    /// Expected language of the query.
    pub language: Language,
    /// Maximum number of documents to return.
    #[serde(default = "default_top_k")]
    pub top_k: u8,
    /// Caller-supplied intent hint; the server may refine or ignore it.
    #[serde(default)]
    pub intent: QueryIntent,
}

fn default_top_k() -> u8 {
    5
}

impl Query {
    /// Validates externally supplied input.
    ///
    /// # Errors
    /// - [`ValidationError::EmptyQuery`] when `query` is missing or
    ///   whitespace-only.
    /// - [`ValidationError::InvalidTopK`] when `top_k` is `0` or above
    ///   [`MAX_TOP_K`].
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.query.trim().is_empty() {
            return Err(ValidationError::EmptyQuery);
        }
        if self.top_k == 0 || self.top_k > MAX_TOP_K {
            return Err(ValidationError::InvalidTopK(self.top_k));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(query: &str, top_k: u8) -> Query {
        Query {
            query: query.to_owned(),
            language: Language::Ta,
            top_k,
            intent: QueryIntent::Unknown,
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
            Err(ValidationError::EmptyQuery)
        );
    }

    #[test]
    fn validate_should_reject_top_k_above_the_limit() {
        assert_eq!(
            request("q", MAX_TOP_K + 1).validate(),
            Err(ValidationError::InvalidTopK(MAX_TOP_K + 1))
        );
    }

    #[test]
    fn validate_should_reject_zero_top_k() {
        assert_eq!(
            request("q", 0).validate(),
            Err(ValidationError::InvalidTopK(0))
        );
    }

    #[test]
    fn deserialization_should_default_top_k_and_intent() {
        let req: Query =
            serde_json::from_str(r#"{"query":"q","language":"bn"}"#).expect("deserialize");
        assert_eq!(req.top_k, 5);
        assert_eq!(req.intent, QueryIntent::Unknown);
    }

    #[test]
    fn intent_should_roundtrip_snake_case() {
        let json = serde_json::to_value(QueryIntent::OffTopic).expect("serialize");
        assert_eq!(json, serde_json::json!("off_topic"));
    }
}
