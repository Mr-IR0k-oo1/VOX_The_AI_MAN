//! Retrieval legs and the hybrid retriever.
//!
//! * [`DenseRetriever`] — embeds the query, searches a [`VectorStore`].
//! * [`SparseRetriever`] — BM25 keyword search over Tantivy.
//! * [`HybridRetriever`] — runs both concurrently and fuses them with
//!   Reciprocal Rank Fusion into an ordered candidate pool (top 20 by
//!   configuration) for downstream reranking.

use async_trait::async_trait;
use std::sync::Arc;

use crate::bm25::TantivyBm25Index;
use crate::chunk::ChunkRecord;
use crate::embed::Embedder;
use crate::error::OreoError;
use crate::fusion::rrf_fuse;
use crate::vector_store::VectorStore;

/// One scored chunk flowing through retrieval/reranking.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredChunk {
    /// Matched chunk with full metadata.
    pub chunk: ChunkRecord,
    /// Relevance score; semantics depend on the producing stage
    /// (cosine, BM25, RRF, or reranker score). Higher is better.
    pub score: f32,
}

/// A single retrieval leg over the indexed corpus.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Leg name used in logs (`dense`, `sparse`, `hybrid`).
    fn name(&self) -> &'static str;

    /// Returns up to `top_k` chunks ordered by descending score.
    ///
    /// # Errors
    /// Returns [`OreoError`] when the underlying index or embedder fails.
    async fn retrieve(&self, query_text: &str, top_k: usize)
        -> Result<Vec<ScoredChunk>, OreoError>;
}

/// Dense leg: hashed/neural embedding + vector store search.
pub struct DenseRetriever {
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn VectorStore>,
}

impl DenseRetriever {
    /// Creates a dense retriever from an embedder and a vector store.
    #[must_use]
    pub const fn new(embedder: Arc<dyn Embedder>, store: Arc<dyn VectorStore>) -> Self {
        Self { embedder, store }
    }
}

#[async_trait]
impl Retriever for DenseRetriever {
    fn name(&self) -> &'static str {
        "dense"
    }

    async fn retrieve(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>, OreoError> {
        let vectors = self.embedder.embed(&[query_text.to_owned()]).await?;
        let Some(query_vector) = vectors.into_iter().next() else {
            return Ok(Vec::new());
        };
        let hits = self.store.search(&query_vector, top_k).await?;
        Ok(hits
            .into_iter()
            .map(|hit| ScoredChunk {
                chunk: hit.chunk,
                score: hit.score,
            })
            .collect())
    }
}

/// Sparse leg: BM25 keyword search over the Tantivy index.
pub struct SparseRetriever {
    index: Arc<TantivyBm25Index>,
}

impl SparseRetriever {
    /// Creates a sparse retriever over `index`.
    #[must_use]
    pub const fn new(index: Arc<TantivyBm25Index>) -> Self {
        Self { index }
    }
}

#[async_trait]
impl Retriever for SparseRetriever {
    fn name(&self) -> &'static str {
        "sparse"
    }

    async fn retrieve(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>, OreoError> {
        let hits = self.index.search(query_text, top_k)?;
        Ok(hits
            .into_iter()
            .map(|hit| ScoredChunk {
                chunk: hit.chunk,
                score: hit.score,
            })
            .collect())
    }
}

/// Hybrid leg: dense + sparse in parallel, fused with RRF.
pub struct HybridRetriever {
    dense: Arc<dyn Retriever>,
    sparse: Arc<dyn Retriever>,
    rrf_k: usize,
    candidate_top: usize,
}

impl HybridRetriever {
    /// Creates a hybrid retriever.
    #[must_use]
    pub const fn new(
        dense: Arc<dyn Retriever>,
        sparse: Arc<dyn Retriever>,
        rrf_k: usize,
        candidate_top: usize,
    ) -> Self {
        Self {
            dense,
            sparse,
            rrf_k,
            candidate_top,
        }
    }
}

/// Stage timings captured by [`HybridRetriever::retrieve_with_timings`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridTimings {
    /// Dense leg duration in milliseconds.
    pub dense_ms: f64,
    /// Sparse leg duration in milliseconds.
    pub sparse_ms: f64,
    /// Fusion duration in milliseconds.
    pub fuse_ms: f64,
}

#[async_trait]
impl Retriever for HybridRetriever {
    fn name(&self) -> &'static str {
        "hybrid"
    }

    async fn retrieve(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>, OreoError> {
        self.retrieve_with_timings(query_text, top_k)
            .await
            .map(|(chunks, _timings)| chunks)
    }
}

impl HybridRetriever {
    /// Runs both legs concurrently, fuses them, and reports per-stage
    /// durations so the API layer can expose retrieval timings.
    ///
    /// # Errors
    /// Returns [`OreoError`] when either leg fails.
    pub async fn retrieve_with_timings(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Result<(Vec<ScoredChunk>, HybridTimings), OreoError> {
        let fetch = top_k.max(self.candidate_top);
        // Each leg carries its own stopwatch so per-stage timings stay
        // meaningful even though the legs execute concurrently.
        let dense_future = async {
            let started = std::time::Instant::now();
            let result = self.dense.retrieve(query_text, fetch).await;
            (result, elapsed_ms(started.elapsed()))
        };
        let sparse_future = async {
            let started = std::time::Instant::now();
            let result = self.sparse.retrieve(query_text, fetch).await;
            (result, elapsed_ms(started.elapsed()))
        };
        let ((dense_result, dense_ms), (sparse_result, sparse_ms)) =
            tokio::join!(dense_future, sparse_future);
        let dense = dense_result?;
        let sparse = sparse_result?;

        let fuse_started = std::time::Instant::now();
        let dense_records: Vec<ChunkRecord> =
            dense.into_iter().map(|scored| scored.chunk).collect();
        let sparse_records: Vec<ChunkRecord> =
            sparse.into_iter().map(|scored| scored.chunk).collect();
        let fused = rrf_fuse(
            &[dense_records, sparse_records],
            self.rrf_k,
            self.candidate_top,
        );
        let fuse_elapsed = fuse_started.elapsed();

        let timings = HybridTimings {
            dense_ms,
            sparse_ms,
            fuse_ms: elapsed_ms(fuse_elapsed),
        };
        Ok((
            fused
                .into_iter()
                .map(|(chunk, score)| ScoredChunk { chunk, score })
                .collect(),
            timings,
        ))
    }
}

/// Duration → milliseconds.
#[must_use]
pub fn elapsed_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::MemoryVectorStore;
    use vox_types::Language;

    fn chunk(id: &str, text: &str) -> ChunkRecord {
        ChunkRecord {
            document_id: id.to_owned(),
            chunk_id: format!("{id}#0000"),
            chunk_index: 0,
            text: text.to_owned(),
            language: Language::En,
            chunking_strategy: "fixed".to_owned(),
            source: "test".to_owned(),
        }
    }

    async fn setup() -> (Arc<DenseRetriever>, Arc<SparseRetriever>, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let bm25 = Arc::new(TantivyBm25Index::open_or_create(dir.path()).expect("index"));
        bm25.add_chunks(&[
            chunk("gst", "GST is an indirect tax on goods and services"),
            chunk("water", "Chennai metro water connection apply online"),
            chunk("itr", "Income tax return filing ITR form salary"),
        ])
        .expect("seed");

        let embedder = Arc::new(crate::embed::HashedEmbedder::new(128));
        let store = Arc::new(MemoryVectorStore::new());
        let seed_texts = [
            ("gst", "GST is an indirect tax on goods and services"),
            ("water", "Chennai metro water connection apply online"),
            ("itr", "Income tax return filing ITR form salary"),
        ];
        let mut pairs: Vec<(ChunkRecord, Vec<f32>)> = Vec::with_capacity(seed_texts.len());
        for (id, text) in seed_texts {
            let vector = embedder
                .embed(&[text.to_owned()])
                .await
                .expect("embed")
                .remove(0);
            pairs.push((chunk(id, text), vector));
        }
        store.upsert(&pairs).await.expect("upsert");

        (
            Arc::new(DenseRetriever::new(embedder, store)),
            Arc::new(SparseRetriever::new(bm25)),
            dir,
        )
    }

    #[tokio::test]
    async fn dense_retriever_should_rank_relevant_chunks_first() {
        let (dense, _sparse, _guard) = setup().await;
        let hits = dense.retrieve("What is GST?", 3).await.expect("retrieve");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].chunk.document_id, "gst");
        assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
    }

    #[tokio::test]
    async fn sparse_retriever_should_rank_bm25_matches_first() {
        let (_dense, sparse, _guard) = setup().await;
        let hits = sparse
            .retrieve("income tax return filing", 3)
            .await
            .expect("retrieve");
        assert_eq!(hits[0].chunk.document_id, "itr");
    }

    #[tokio::test]
    async fn hybrid_should_fuse_and_order_descending() {
        let (dense, sparse, _guard) = setup().await;
        let hybrid = HybridRetriever::new(dense, sparse, 60, 20);
        let hits = hybrid
            .retrieve("GST indirect tax", 5)
            .await
            .expect("retrieve");
        assert_eq!(hits[0].chunk.document_id, "gst");
        assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
        assert!(hits.len() <= 20);
    }

    #[tokio::test]
    async fn hybrid_should_preserve_metadata_through_fusion() {
        let (dense, sparse, _guard) = setup().await;
        let hybrid = HybridRetriever::new(dense, sparse, 60, 20);
        let hits = hybrid.retrieve("metro water", 5).await.expect("retrieve");
        let water = hits
            .iter()
            .find(|h| h.chunk.document_id == "water")
            .expect("water hit");
        assert_eq!(water.chunk.chunk_id, "water#0000");
        assert_eq!(water.chunk.language, Language::En);
        assert_eq!(water.chunk.source, "test");
        assert_eq!(water.chunk.chunking_strategy, "fixed");
    }
}
