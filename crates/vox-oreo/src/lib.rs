//! OREO — the independent VOX retrieval engine.
//!
//! OREO owns everything between a raw corpus and ranked evidence:
//!
//! 1. **Ingestion** ([`ingest`]): reads AI4Bharat MSMARCO-XI-style JSONL
//!    corpora, tolerating the field-name variants shipped by the dataset.
//! 2. **Preprocessing** ([`preprocess`]): `raw → clean → normalize → language
//!    metadata → documents`, with script-based language detection and a
//!    configurable supported-language set (default: English, Hindi, Tamil).
//! 3. **Chunking** ([`chunk`]): a [`chunk::Chunker`] abstraction with fixed,
//!    sentence-aware, and sliding-window implementations; every chunk carries
//!    `document_id`, `chunk_id`, `language`, `chunking_strategy`, `source`.
//! 4. **Embedding** ([`embed`]): an [`embed::Embedder`] boundary with a
//!    deterministic multilingual hashed embedder as the offline default; a
//!    neural encoder can be dropped in behind the same trait.
//! 5. **Indexing**: dense vectors in Qdrant ([`vector_store::QdrantStore`],
//!    REST) or an in-process cosine store ([`vector_store::MemoryVectorStore`]),
//!    plus BM25 via Tantivy ([`bm25::TantivyBm25Index`]).
//! 6. **Retrieval** ([`retriever`]): [`retriever::DenseRetriever`],
//!    [`retriever::SparseRetriever`], and [`retriever::HybridRetriever`]
//!    fusing both legs with Reciprocal Rank Fusion ([`fusion::rrf`]) into a
//!    top-20 candidate pool.
//! 7. **Reranking** ([`rerank`]): a [`rerank::Reranker`] abstraction that
//!    reduces the fused pool to the final top-5.
//!
//! The engine is exposed three ways: in-process through [`engine::OreoEngine`]
//! (used by `vox-api`'s embedded backend), over HTTP through [`service`]
//! (`POST /v1/retrieve`, same wire contract), and via the `oreo` CLI
//! (`index`, `serve`, `verify`, `bench`). Nothing here couples to IR0K or any
//! other VOX stage.

pub mod bm25;
pub mod chunk;
pub mod config;
pub mod embed;
pub mod engine;
pub mod error;
pub mod fusion;
pub mod ingest;
pub mod preprocess;
pub mod rerank;
pub mod retriever;
pub mod service;
pub mod vector_store;

pub use chunk::{Chunker, FixedChunker, SentenceChunker, SlidingChunker};
pub use config::OreoConfig;
pub use embed::{Embedder, HashedEmbedder};
pub use engine::OreoEngine;
pub use error::OreoError;
pub use rerank::{LexicalOverlapReranker, NoopReranker, Reranker};
pub use retriever::{DenseRetriever, HybridRetriever, Retriever, ScoredChunk, SparseRetriever};

use std::sync::Arc;

/// Shared handle to a retriever implementation.
pub type SharedRetriever = Arc<dyn Retriever>;

/// Shared handle to a reranker implementation.
pub type SharedReranker = Arc<dyn Reranker>;
