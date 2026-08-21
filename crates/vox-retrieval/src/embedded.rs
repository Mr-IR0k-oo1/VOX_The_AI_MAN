//! In-process OREO backend: runs the real retrieval engine inside this
//! process, no HTTP hop.
//!
//! Used when `VOX_RETRIEVAL_MODE=oreo`. The engine owns Qdrant/Tantivy
//! internals; this adapter only translates between the [`RetrievalClient`]
//! boundary and [`vox_oreo::OreoEngine`].

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use vox_types::{Query, RetrievalResponse};

use crate::error::RetrievalError;
use crate::RetrievalClient;

/// Embedded OREO retrieval engine backend.
#[derive(Clone)]
pub struct EmbeddedOreoClient {
    engine: Arc<vox_oreo::OreoEngine>,
    /// Artificial floor on response latency; zero in production, useful for
    /// timeout tests.
    delay: Duration,
}

impl EmbeddedOreoClient {
    /// Wraps an assembled engine.
    #[must_use]
    pub const fn new(engine: Arc<vox_oreo::OreoEngine>) -> Self {
        Self {
            engine,
            delay: Duration::ZERO,
        }
    }

    /// Wraps an engine with a simulated per-call delay (testing only).
    #[must_use]
    pub const fn with_delay(engine: Arc<vox_oreo::OreoEngine>, delay: Duration) -> Self {
        Self { engine, delay }
    }

    /// The wrapped engine.
    #[must_use]
    pub fn engine(&self) -> &Arc<vox_oreo::OreoEngine> {
        &self.engine
    }
}

#[async_trait]
impl RetrievalClient for EmbeddedOreoClient {
    fn name(&self) -> &'static str {
        "oreo-embedded"
    }

    async fn retrieve(&self, request: Query) -> Result<RetrievalResponse, RetrievalError> {
        request.validate()?;
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        self.engine
            .retrieve(request)
            .await
            .map_err(|err| match err {
                vox_oreo::OreoError::Validation(validation) => {
                    RetrievalError::Validation(validation)
                }
                other => RetrievalError::Engine(other.to_string()),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use vox_types::{Language, QueryIntent};

    fn test_engine() -> Arc<vox_oreo::OreoEngine> {
        // Parallel tests must never share a Tantivy directory (single writer
        // lock per directory).
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("oreo-embedded-{}-{unique}", std::process::id()));
        let config = vox_oreo::OreoConfig {
            tantivy_dir: dir,
            ..vox_oreo::OreoConfig::default()
        };
        Arc::new(vox_oreo::OreoEngine::new(config).expect("engine"))
    }

    fn request(query: &str) -> Query {
        Query {
            query: query.to_owned(),
            language: Language::En,
            top_k: 5,
            intent: QueryIntent::Unknown,
        }
    }

    #[tokio::test]
    async fn embedded_client_should_return_engine_results_with_timings() {
        let engine = test_engine();
        engine
            .index_documents(vox_oreo::ingest::sample_corpus())
            .await
            .expect("index sample corpus");
        let client = EmbeddedOreoClient::new(engine);

        let response = client
            .retrieve(request("What is GST?"))
            .await
            .expect("retrieve");
        assert!(!response.documents.is_empty());
        assert!(response.timings_ms.is_some());

        let first = &response.documents[0];
        assert_eq!(first.rank, 1);
        assert!(first.score.is_finite());
        assert_eq!(
            first.metadata["document_id"].as_str(),
            Some(first.id.split('#').next().unwrap_or_default())
        );
    }

    #[tokio::test]
    async fn embedded_client_should_reject_invalid_queries_without_indexing() {
        let client = EmbeddedOreoClient::new(test_engine());
        let err = client
            .retrieve(request("   "))
            .await
            .expect_err("blank query rejected");
        assert!(matches!(
            err,
            RetrievalError::Validation(vox_types::ValidationError::EmptyQuery)
        ));
    }

    #[tokio::test]
    async fn empty_index_should_return_empty_results_not_errors() {
        let client = EmbeddedOreoClient::new(test_engine());
        let response = client
            .retrieve(request("anything at all"))
            .await
            .expect("retrieve");
        assert!(response.documents.is_empty());
    }
}
