//! Reranking: reorders and trims the fused candidate pool down to the final
//! result count (top 5 by configuration).
//!
//! [`Reranker`] is the substitution boundary; a cross-encoder or an LLM-based
//! reranker can replace the default without touching retrieval. The offline
//! default, [`LexicalOverlapReranker`], scores query↔chunk token overlap into
//! a `[0, 1]` range — deliberately comparable to VOX's grounding threshold,
//! unlike raw RRF scores which live near `1/k`.

use async_trait::async_trait;
use std::sync::Arc;

use crate::embed::HashedEmbedder;
use crate::error::OreoError;
use crate::retriever::ScoredChunk;

/// Which reranker implementation to build from configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankerKind {
    /// [`LexicalOverlapReranker`] (default).
    LexicalOverlap,
    /// [`NoopReranker`]: keeps fusion order and RRF scores.
    Noop,
}

/// Reorders candidates for a query.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Backend name used in logs.
    fn name(&self) -> &'static str;

    /// Returns at most `top_k` candidates, best first.
    ///
    /// # Errors
    /// Returns [`OreoError`] if the reranking backend fails.
    async fn rerank(
        &self,
        query_text: &str,
        candidates: Vec<ScoredChunk>,
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>, OreoError>;
}

/// Keeps the fused order but trims to `top_k`.
///
/// Useful as a baseline and for A/B comparisons against real rerankers.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopReranker;

#[async_trait]
impl Reranker for NoopReranker {
    fn name(&self) -> &'static str {
        "noop"
    }

    async fn rerank(
        &self,
        _query_text: &str,
        mut candidates: Vec<ScoredChunk>,
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>, OreoError> {
        candidates.truncate(top_k);
        Ok(candidates)
    }
}

/// Token-overlap reranker producing scores in `[0, 1]`.
///
/// Score = weighted Jaccard-style overlap between query tokens and chunk
/// tokens (word unigrams plus character trigrams, matching the embedder's
/// feature space). Deterministic, multilingual, and cheap enough to run on
/// every request.
#[derive(Debug, Clone, Default)]
pub struct LexicalOverlapReranker {
    _private: (),
}

impl LexicalOverlapReranker {
    /// Creates the lexical reranker.
    #[must_use]
    pub const fn new() -> Self {
        Self { _private: () }
    }

    fn score(query_text: &str, chunk_text: &str) -> f32 {
        let query_features = features(query_text);
        if query_features.is_empty() {
            return 0.0;
        }
        let chunk_features = features(chunk_text);
        let mut intersection = 0.0f32;
        for (feature, weight) in &query_features {
            if let Some(other) = chunk_features.get(feature) {
                intersection += weight.min(*other);
            }
        }
        let union: f32 = query_features.values().sum::<f32>().max(f32::EPSILON);
        (intersection / union).clamp(0.0, 1.0)
    }
}

fn features(text: &str) -> std::collections::HashMap<String, f32> {
    let mut map = std::collections::HashMap::new();
    for token in HashedEmbedder::tokenize(text) {
        *map.entry(token).or_insert(0.0) += 1.0;
    }
    for trigram in HashedEmbedder::char_trigrams(text) {
        *map.entry(trigram).or_insert(0.0) += 0.25;
    }
    map
}

#[async_trait]
impl Reranker for LexicalOverlapReranker {
    fn name(&self) -> &'static str {
        "lexical_overlap"
    }

    async fn rerank(
        &self,
        query_text: &str,
        candidates: Vec<ScoredChunk>,
        top_k: usize,
    ) -> Result<Vec<ScoredChunk>, OreoError> {
        let mut scored: Vec<ScoredChunk> = candidates
            .into_iter()
            .map(|candidate| ScoredChunk {
                score: Self::score(query_text, &candidate.chunk.text),
                chunk: candidate.chunk,
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk.chunk_id.cmp(&b.chunk.chunk_id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

/// Builds the configured reranker.
#[must_use]
pub fn build_reranker(kind: RerankerKind) -> Arc<dyn Reranker> {
    match kind {
        RerankerKind::LexicalOverlap => Arc::new(LexicalOverlapReranker::new()),
        RerankerKind::Noop => Arc::new(NoopReranker),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkRecord;
    use vox_types::Language;

    fn candidate(id: &str, text: &str, rrf_score: f32) -> ScoredChunk {
        ScoredChunk {
            chunk: ChunkRecord {
                document_id: id.to_owned(),
                chunk_id: format!("{id}#0000"),
                chunk_index: 0,
                text: text.to_owned(),
                language: Language::En,
                chunking_strategy: "fixed".to_owned(),
                source: "test".to_owned(),
            },
            score: rrf_score,
        }
    }

    #[tokio::test]
    async fn lexical_reranker_should_promote_relevant_candidates() {
        let reranker = LexicalOverlapReranker::new();
        let candidates = vec![
            candidate("water", "Chennai metro water connection apply online", 0.9),
            candidate("gst", "GST is an indirect tax on goods and services", 0.8),
        ];
        let ranked = reranker
            .rerank("What is GST indirect tax?", candidates, 2)
            .await
            .expect("rerank");
        assert_eq!(ranked[0].chunk.document_id, "gst");
        assert!(ranked[0].score > ranked[1].score);
        assert!((0.0..=1.0).contains(&ranked[0].score));
    }

    #[tokio::test]
    async fn reranker_should_trim_to_top_k() {
        let reranker = LexicalOverlapReranker::new();
        let candidates: Vec<ScoredChunk> = (0..20)
            .map(|i| candidate(&format!("d{i}"), "unrelated filler words here", 0.5))
            .collect();
        let ranked = reranker
            .rerank("query", candidates, 5)
            .await
            .expect("rerank");
        assert_eq!(ranked.len(), 5);
    }

    #[tokio::test]
    async fn noop_reranker_should_preserve_order_and_scores() {
        let reranker = NoopReranker;
        let candidates = vec![
            candidate("a", "alpha", 0.033),
            candidate("b", "beta", 0.032),
        ];
        let ranked = reranker
            .rerank("anything", candidates, 5)
            .await
            .expect("rerank");
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].chunk.document_id, "a");
        assert_eq!(ranked[0].score, 0.033);
    }

    #[tokio::test]
    async fn scores_should_stay_ordered_descending() {
        let reranker = LexicalOverlapReranker::new();
        let candidates: Vec<ScoredChunk> = (0..10)
            .map(|i| candidate(&format!("d{i}"), "gst tax goods services india", 0.5))
            .collect();
        let ranked = reranker
            .rerank("gst tax", candidates, 10)
            .await
            .expect("rerank");
        assert!(ranked.windows(2).all(|w| w[0].score >= w[1].score));
    }

    #[test]
    fn factory_should_build_each_kind() {
        assert_eq!(build_reranker(RerankerKind::Noop).name(), "noop");
        assert_eq!(
            build_reranker(RerankerKind::LexicalOverlap).name(),
            "lexical_overlap"
        );
    }
}
