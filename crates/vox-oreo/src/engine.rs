//! The OREO engine: ingestion → indexing → hybrid retrieval → reranking,
//! with per-stage timings on every query.
//!
//! [`OreoEngine::index_corpus`] is the reproducible offline pipeline
//! (`raw → clean → normalize → language metadata → documents → chunks →
//! embeddings → indexes`); [`OreoEngine::retrieve`] is the online path behind
//! `POST /v1/retrieve`.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use vox_types::{Query, RetrievalResponse, RetrievalTimings, RetrievedDocument};

use crate::bm25::TantivyBm25Index;
use crate::config::{OreoConfig, VectorStoreKind};
use crate::embed::{Embedder, HashedEmbedder};
use crate::error::OreoError;
use crate::ingest::{load_corpus, RawDocument};
use crate::preprocess::{preprocess, PreprocessStats};
use crate::rerank::{build_reranker, Reranker};
use crate::retriever::{DenseRetriever, HybridRetriever, Retriever, SparseRetriever};
use crate::vector_store::{MemoryVectorStore, QdrantStore, VectorStore};

/// How many chunks to embed per batch during indexing.
const EMBED_BATCH: usize = 64;

/// Counts reported after an indexing run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexReport {
    /// Raw records read from disk.
    pub raw_documents: usize,
    /// Documents surviving preprocessing.
    pub documents: usize,
    /// Chunks produced by the chunker.
    pub chunks: usize,
    /// Preprocessing skip counters.
    pub stats: PreprocessStats,
}

/// The assembled retrieval engine.
pub struct OreoEngine {
    config: OreoConfig,
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn VectorStore>,
    bm25: Arc<TantivyBm25Index>,
    dense: Arc<DenseRetriever>,
    sparse: Arc<SparseRetriever>,
    reranker: Arc<dyn Reranker>,
}

impl OreoEngine {
    /// Assembles the engine from `config`.
    ///
    /// # Errors
    /// Returns [`OreoError`] when configuration is inconsistent or the BM25
    /// index directory cannot be opened/created.
    pub fn new(config: OreoConfig) -> Result<Self, OreoError> {
        config.validate()?;
        let embedder: Arc<dyn Embedder> = Arc::new(HashedEmbedder::new(config.embedding_dim));
        let store: Arc<dyn VectorStore> = match config.vector_store {
            VectorStoreKind::Memory => Arc::new(MemoryVectorStore::new()),
            VectorStoreKind::Qdrant => Arc::new(QdrantStore::new(
                &config.qdrant_url,
                &config.qdrant_collection,
                config.embedding_dim,
                config.qdrant_timeout,
            )?),
        };
        let bm25 = Arc::new(TantivyBm25Index::open_or_create(&config.tantivy_dir)?);
        let dense = Arc::new(DenseRetriever::new(
            Arc::clone(&embedder),
            Arc::clone(&store),
        ));
        let sparse = Arc::new(SparseRetriever::new(Arc::clone(&bm25)));
        let reranker = build_reranker(config.reranker);
        tracing::info!(
            vector_store = store.name(),
            embedder = embedder.name(),
            reranker = reranker.name(),
            languages = ?config.languages.iter().map(|language| language.code()).collect::<Vec<_>>(),
            chunking = config.chunking.name(),
            candidate_top = config.candidate_top,
            final_top = config.final_top,
            rrf_k = config.rrf_k,
            "oreo engine assembled"
        );
        Ok(Self {
            config,
            embedder,
            store,
            bm25,
            dense,
            sparse,
            reranker,
        })
    }

    /// The engine's configuration.
    #[must_use]
    pub fn config(&self) -> &OreoConfig {
        &self.config
    }

    /// The hybrid retriever (dense + sparse + RRF) used for queries.
    ///
    /// Exposed so the service layer and benchmarks reuse the exact online
    /// path rather than reimplementing it.
    #[must_use]
    pub fn hybrid_retriever(&self) -> HybridRetriever {
        HybridRetriever::new(
            Arc::clone(&self.dense) as Arc<dyn Retriever>,
            Arc::clone(&self.sparse) as Arc<dyn Retriever>,
            self.config.rrf_k,
            self.config.candidate_top,
        )
    }

    /// The dense leg alone (embed + vector search), for ablation benchmarks.
    #[must_use]
    pub fn dense_retriever(&self) -> Arc<DenseRetriever> {
        Arc::clone(&self.dense)
    }

    /// The sparse (BM25) leg alone, for ablation benchmarks.
    #[must_use]
    pub fn sparse_retriever(&self) -> Arc<SparseRetriever> {
        Arc::clone(&self.sparse)
    }

    /// Runs the full offline pipeline over a corpus file/directory.
    ///
    /// # Errors
    /// Returns [`OreoError`] for unreadable corpora, embedding failures, or
    /// index write failures.
    pub async fn index_corpus(&self, input: &Path) -> Result<IndexReport, OreoError> {
        let raw = load_corpus(input)?;
        self.index_documents(raw).await
    }

    /// Runs the offline pipeline over already-loaded raw documents.
    ///
    /// # Errors
    /// Returns [`OreoError`] for embedding or index write failures.
    pub async fn index_documents(&self, raw: Vec<RawDocument>) -> Result<IndexReport, OreoError> {
        let started = Instant::now();
        let raw_count = raw.len();
        let (documents, stats) = preprocess(&raw, &self.config.languages)?;
        tracing::info!(
            raw = raw_count,
            kept = stats.kept,
            skipped_too_short = stats.skipped_too_short,
            skipped_unsupported_language = stats.skipped_unsupported_language,
            duration_ms = elapsed_ms(started.elapsed()),
            "preprocessing complete"
        );

        let chunker = self.config.chunking.build();
        let chunks: Vec<_> = documents
            .iter()
            .flat_map(|document| chunker.chunk_document(document))
            .collect();
        let chunk_count = chunks.len();

        // Embed in batches so large corpora never materialize all vectors at
        // once; batches are deterministic and order-stable.
        let mut pairs = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(EMBED_BATCH) {
            let texts: Vec<String> = batch.iter().map(|chunk| chunk.text.clone()).collect();
            let vectors = self.embedder.embed(&texts).await?;
            pairs.extend(batch.iter().cloned().zip(vectors));
        }

        self.store.upsert(&pairs).await?;
        self.bm25.add_chunks(&chunks)?;

        tracing::info!(
            chunks = chunk_count,
            duration_ms = elapsed_ms(started.elapsed()),
            "indexes updated"
        );
        Ok(IndexReport {
            raw_documents: raw_count,
            documents: documents.len(),
            chunks: chunk_count,
            stats,
        })
    }

    /// Executes hybrid retrieval + RRF fusion + reranking for one query.
    ///
    /// The fused pool holds up to `candidate_top` (default 20) results; the
    /// reranker trims it to `min(final_top, request.top_k)` (default 5).
    ///
    /// # Errors
    /// Returns [`OreoError`] on invalid input or index failures.
    pub async fn retrieve(&self, query: Query) -> Result<RetrievalResponse, OreoError> {
        let overall_started = Instant::now();
        query.validate()?;

        let embed_started = Instant::now();
        let query_vector = self
            .embedder
            .embed(std::slice::from_ref(&query.query))
            .await?
            .into_iter()
            .next();
        let embed_ms = elapsed_ms(embed_started.elapsed());

        let Some(_query_vector) = query_vector else {
            return Ok(RetrievalResponse {
                timings_ms: Some(RetrievalTimings {
                    embed: Some(embed_ms),
                    total: Some(elapsed_ms(overall_started.elapsed())),
                    ..RetrievalTimings::default()
                }),
                ..RetrievalResponse::default()
            });
        };

        let hybrid = self.hybrid_retriever();
        let (fused, leg_timings) = hybrid
            .retrieve_with_timings(&query.query, self.config.candidate_top)
            .await?;

        let rerank_started = Instant::now();
        let final_count = usize::from(query.top_k).min(self.config.final_top);
        let mut ranked = self
            .reranker
            .rerank(&query.query, fused, final_count)
            .await?;
        let rerank_ms = elapsed_ms(rerank_started.elapsed());

        let documents = ranked
            .drain(..)
            .enumerate()
            .map(|(index, scored)| {
                let metadata =
                    serde_json::to_value(&scored.chunk).unwrap_or(serde_json::Value::Null);
                RetrievedDocument {
                    id: scored.chunk.chunk_id,
                    text: scored.chunk.text,
                    score: scored.score,
                    rank: u32::try_from(index + 1).unwrap_or(u32::MAX),
                    metadata,
                }
            })
            .collect();

        Ok(RetrievalResponse {
            documents,
            timings_ms: Some(RetrievalTimings {
                embed: Some(embed_ms),
                dense: Some(leg_timings.dense_ms),
                sparse: Some(leg_timings.sparse_ms),
                fuse: Some(leg_timings.fuse_ms),
                rerank: Some(rerank_ms),
                total: Some(elapsed_ms(overall_started.elapsed())),
            }),
        })
    }

    /// Number of vectors in the dense store.
    ///
    /// # Errors
    /// Propagates store failures.
    pub async fn dense_count(&self) -> Result<usize, OreoError> {
        self.store.count().await
    }

    /// Number of documents in the BM25 index.
    ///
    /// # Errors
    /// Propagates index failures.
    pub fn sparse_count(&self) -> Result<usize, OreoError> {
        self.bm25.len()
    }

    /// Connectivity probe used by `oreo verify`: reports dense/sparse store
    /// names with live document counts.
    ///
    /// # Errors
    /// Propagates store/index failures.
    pub async fn verify(&self) -> Result<(String, String), OreoError> {
        let dense = format!(
            "{}({} points)",
            self.store.name(),
            self.dense_count().await?
        );
        let sparse = format!("tantivy({} docs)", self.sparse_count()?);
        Ok((dense, sparse))
    }
}

/// Duration → milliseconds helper shared across the engine.
#[must_use]
pub fn elapsed_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
