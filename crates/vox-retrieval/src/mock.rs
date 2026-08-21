//! Deterministic in-memory retrieval backend.

use std::time::Duration;

use async_trait::async_trait;
use vox_types::{Query, RetrievalResponse, RetrievedDocument};

use crate::error::RetrievalError;
use crate::RetrievalClient;

/// Placeholder backend used until the real retrieval service is stable.
///
/// Produces ranked synthetic evidence derived from the query so the full
/// pipeline can be exercised end-to-end without Qdrant/Tantivy. Output is
/// deterministic for a given request; an optional artificial delay supports
/// timeout/retry testing.
#[derive(Debug, Clone, Copy)]
pub struct MockRetrievalClient {
    delay: Duration,
}

impl MockRetrievalClient {
    /// Creates a mock backend that sleeps for `delay` before answering.
    #[must_use]
    pub const fn new(delay: Duration) -> Self {
        Self { delay }
    }
}

#[async_trait]
impl RetrievalClient for MockRetrievalClient {
    fn name(&self) -> &'static str {
        "mock"
    }

    async fn retrieve(&self, request: Query) -> Result<RetrievalResponse, RetrievalError> {
        request.validate()?;
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }

        let query = request.query.trim();
        let documents = (0..usize::from(request.top_k))
            .map(|i| RetrievedDocument {
                id: format!("mock-{i:04}"),
                text: format!("Mock evidence chunk {i} relevant to: {query}"),
                score: 0.95 - i as f32 * 0.05,
                rank: u32::try_from(i).expect("index fits u32") + 1,
                metadata: serde_json::json!({
                    "source": "mock",
                    "language": request.language.code(),
                }),
            })
            .collect();

        Ok(RetrievalResponse { documents })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use vox_types::{Language, Query, QueryIntent, ValidationError};

    use super::*;

    fn request(query: &str, top_k: u8) -> Query {
        Query {
            query: query.to_owned(),
            language: Language::Ta,
            top_k,
            intent: QueryIntent::Unknown,
        }
    }

    #[tokio::test]
    async fn retrieve_should_return_deterministic_ranked_results() {
        let client = MockRetrievalClient::new(Duration::ZERO);
        let first = client
            .retrieve(request("  what is gst  ", 3))
            .await
            .expect("first call");
        let second = client
            .retrieve(request("  what is gst  ", 3))
            .await
            .expect("second call");

        assert_eq!(first.documents.len(), 3);
        assert_eq!(first.documents[0].rank, 1);
        assert_eq!(first.documents[2].id, "mock-0002");
        assert_eq!(first.documents, second.documents);
        assert_eq!(
            first.documents[0].text,
            "Mock evidence chunk 0 relevant to: what is gst"
        );
    }

    #[tokio::test]
    async fn retrieve_should_reject_invalid_input_without_delay() {
        let client = MockRetrievalClient::new(Duration::from_secs(60));
        let err = client
            .retrieve(request("   ", 3))
            .await
            .expect_err("should reject");
        assert!(matches!(
            err,
            RetrievalError::Validation(ValidationError::EmptyQuery)
        ));
    }
}
