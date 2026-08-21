//! The text query contract shared by `POST /v1/query` and `POST /v1/retrieve`.

use serde::{Deserialize, Serialize};

use crate::error::{ValidationError, MAX_TOP_K};
use crate::language::Language;

/// Coarse classification of what the caller wants done with a query.
///
/// Detected with deterministic heuristics during query analysis; a
/// caller-supplied value is honored when present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryIntent {
    /// No intent supplied or detected.
    #[default]
    Unknown,
    /// A fact is being requested ("Who is the finance minister?").
    Factual,
    /// A definition is being requested ("What is GST?").
    Definition,
    /// Two or more things are being compared ("Difference between TCP and UDP").
    Comparison,
    /// A procedure is being requested ("How do I file an ITR?").
    Procedural,
}

/// Request body shared by `POST /v1/query` and `POST /v1/retrieve`.
///
/// The identical shape is deliberate: the user-facing pipeline forwards the
/// (normalized) query to the retrieval boundary over this exact contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// The raw query text as supplied (wire name: `query`).
    #[serde(rename = "query", alias = "text")]
    pub text: String,
    /// Normalized form of [`Query::text`]; server-populated when absent.
    #[serde(default)]
    pub normalized_text: String,
    /// Language hint for the pipeline; auto-detected when absent.
    #[serde(default)]
    pub language: Option<Language>,
    /// Intent hint; refined by query analysis when `Unknown`.
    #[serde(default)]
    pub intent: QueryIntent,
    /// Maximum number of documents to return.
    #[serde(default = "default_top_k")]
    pub top_k: u8,
}

fn default_top_k() -> u8 {
    5
}

impl Query {
    /// Creates a raw query with no normalization yet.
    #[must_use]
    pub fn new(text: impl Into<String>, language: Option<Language>, top_k: u8) -> Self {
        Self {
            text: text.into(),
            normalized_text: String::new(),
            language,
            intent: QueryIntent::Unknown,
            top_k,
        }
    }

    /// Validates externally supplied input.
    ///
    /// # Errors
    /// - [`ValidationError::EmptyQuery`] when `text` is missing or
    ///   whitespace-only.
    /// - [`ValidationError::InvalidTopK`] when `top_k` is `0` or above
    ///   [`MAX_TOP_K`].
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.text.trim().is_empty() {
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

    #[test]
    fn validate_should_accept_a_well_formed_request() {
        assert_eq!(
            Query::new("what is gst", Some(Language::Ta), 5).validate(),
            Ok(())
        );
    }

    #[test]
    fn validate_should_reject_whitespace_only_queries() {
        let query = Query::new("   \n\t", None, 5);
        assert_eq!(query.validate(), Err(ValidationError::EmptyQuery));
    }

    #[test]
    fn validate_should_reject_top_k_above_the_limit() {
        let query = Query::new("q", None, MAX_TOP_K + 1);
        assert_eq!(
            query.validate(),
            Err(ValidationError::InvalidTopK(MAX_TOP_K + 1))
        );
    }

    #[test]
    fn validate_should_reject_zero_top_k() {
        let query = Query::new("q", None, 0);
        assert_eq!(query.validate(), Err(ValidationError::InvalidTopK(0)));
    }

    #[test]
    fn deserialization_should_default_optional_fields() {
        let req: Query =
            serde_json::from_str(r#"{"query":"q","language":"bn"}"#).expect("deserialize");
        assert_eq!(req.normalized_text, "");
        assert_eq!(req.language, Some(Language::Bn));
        assert_eq!(req.intent, QueryIntent::Unknown);
        assert_eq!(req.top_k, 5);
    }

    #[test]
    fn deserialization_should_accept_the_text_alias() {
        let req: Query = serde_json::from_str(r#"{"text":"q"}"#).expect("deserialize");
        assert_eq!(req.text, "q");
    }

    #[test]
    fn serialization_should_emit_the_query_wire_name() {
        let query = Query::new("hello", None, 5);
        let json = serde_json::to_value(&query).expect("serialize");
        assert!(json.get("query").is_some(), "wire key must be `query`");
        assert!(json.get("text").is_none());
    }

    #[test]
    fn intent_should_serialize_snake_case() {
        assert_eq!(
            serde_json::to_value(QueryIntent::Comparison).expect("serialize"),
            serde_json::json!("comparison")
        );
    }
}
