//! The retrieval evaluation runner.
//!
//! Three experiments, all against the bundled sample corpus and the bundled
//! judged query set:
//!
//! * **Baseline** — the default production configuration (sentence chunking,
//!   hybrid + RRF + lexical rerank) with full per-stage latency attribution
//!   → `retrieval.json`.
//! * **Chunking comparison** — fixed vs sentence vs sliding chunking under
//!   the full pipeline → `chunking.json`.
//! * **Component ablation** — dense only / BM25 only / score-sum fusion /
//!   RRF fusion / RRF + reranker under sentence chunking → `ablation.json`.
//!
//! Every number in the outputs is measured at runtime; nothing is hardcoded.

use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use vox_oreo::chunk::ChunkingStrategy;
use vox_oreo::config::OreoConfig;
use vox_oreo::engine::{elapsed_ms, IndexReport, OreoEngine};
use vox_oreo::fusion::rrf_fuse;
use vox_oreo::retriever::{Retriever, ScoredChunk};
use vox_types::{Query, RetrievalResponse};

use crate::eval::EvalQuery;
use crate::metrics::{mrr, recall_at_k};

/// How many timing sweeps run over the query set (quality is computed once).
pub const TIMING_SWEEPS: usize = 3;

/// Candidate pool fetched per leg before fusion (matches production config).
const CANDIDATE_TOP: usize = 20;

/// Final result count evaluated (`Recall@5`, `MRR` over top 5).
const EVAL_TOP_K: usize = 5;

/// Which retrieval configuration a [`ModeEvaluation`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    /// Dense leg only (embed + vector search).
    DenseOnly,
    /// BM25 leg only.
    Bm25Only,
    /// Dense + BM25 fused by normalized score summation (no RRF).
    ScoreSum,
    /// Dense + BM25 fused with Reciprocal Rank Fusion.
    Rrf,
    /// The full production path: RRF fusion + lexical-overlap reranking.
    RrfRerank,
}

impl RetrievalMode {
    /// Stable machine-readable name used in JSON and Markdown output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::DenseOnly => "dense_only",
            Self::Bm25Only => "bm25_only",
            Self::ScoreSum => "dense+bm25_score_sum",
            Self::Rrf => "dense+bm25_rrf",
            Self::RrfRerank => "dense+bm25_rrf+rerank",
        }
    }

    /// All ablation modes in evaluation order.
    #[must_use]
    pub fn all() -> [Self; 5] {
        [
            Self::DenseOnly,
            Self::Bm25Only,
            Self::ScoreSum,
            Self::Rrf,
            Self::RrfRerank,
        ]
    }
}

/// Mean and P50/P70/P100 latency statistics for one stage, milliseconds.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct StageStats {
    /// Arithmetic mean over all samples.
    pub mean: f64,
    /// 50th percentile.
    pub p50: f64,
    /// 70th percentile.
    pub p70: f64,
    /// Maximum observed value (100th percentile).
    pub p100: f64,
}

impl StageStats {
    fn from_samples(samples: &mut [f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        Some(Self {
            mean,
            p50: crate::percentile(samples, 0.50)?,
            p70: crate::percentile(samples, 0.70)?,
            p100: *samples.last()?,
        })
    }
}

/// Per-stage latency statistics; stages that do not apply to a mode are
/// `null` in JSON.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StageTimings {
    /// Query embedding (production path only).
    pub embed: Option<StageStats>,
    /// Dense leg including its internal query embedding.
    pub dense: Option<StageStats>,
    /// Sparse/BM25 leg.
    pub sparse: Option<StageStats>,
    /// Fusion (score-sum or RRF).
    pub fuse: Option<StageStats>,
    /// Reranking.
    pub rerank: Option<StageStats>,
}

/// Quality and latency results for one retrieval mode or pipeline variant.
#[derive(Debug, Clone, Serialize)]
pub struct ModeEvaluation {
    /// Mode name (`dense_only`, `dense+bm25_rrf+rerank`, ...).
    pub mode: String,
    /// Queries evaluated.
    pub queries: usize,
    /// Mean Recall@5 over the evaluation set.
    pub recall_at_5: f64,
    /// Mean Reciprocal Rank over the evaluation set.
    pub mrr: f64,
    /// Per-stage latency statistics.
    pub stages: StageTimings,
    /// End-to-end retrieval latency statistics.
    pub total: StageStats,
}

/// Corpus/dataset facts recorded alongside every result file.
#[derive(Debug, Clone, Serialize)]
pub struct DatasetInfo {
    /// Raw documents in the corpus.
    pub corpus_documents: usize,
    /// Chunks produced by the active chunking strategy.
    pub chunks_indexed: usize,
    /// Judged queries in the evaluation set.
    pub queries: usize,
    /// Query/corpus languages covered.
    pub languages: Vec<String>,
    /// How relevance judgments were produced.
    pub judgment_method: String,
}

/// The engine settings a benchmark run exercised, echoed into the outputs.
#[derive(Debug, Clone, Serialize)]
pub struct EngineConfigSummary {
    /// Chunking strategy name.
    pub chunking: String,
    /// Embedding dimensionality.
    pub embedding_dim: usize,
    /// Dense store backend (`memory` or `qdrant`).
    pub vector_store: String,
    /// Fused candidate pool size before reranking.
    pub candidate_top: usize,
    /// Final results returned after reranking.
    pub final_top: usize,
    /// Reciprocal Rank Fusion constant.
    pub rrf_k: usize,
    /// Reranker backend name.
    pub reranker: String,
    /// Languages kept by preprocessing.
    pub languages: Vec<String>,
}

impl EngineConfigSummary {
    /// Summarizes `config` for reporting.
    #[must_use]
    pub fn of(config: &OreoConfig) -> Self {
        let reranker_name = vox_oreo::rerank::build_reranker(config.reranker).name();
        Self {
            chunking: config.chunking.name().to_owned(),
            embedding_dim: config.embedding_dim,
            vector_store: match config.vector_store {
                vox_oreo::config::VectorStoreKind::Memory => "memory".to_owned(),
                vox_oreo::config::VectorStoreKind::Qdrant => "qdrant".to_owned(),
            },
            candidate_top: config.candidate_top,
            final_top: config.final_top,
            rrf_k: config.rrf_k,
            reranker: reranker_name.to_owned(),
            languages: config
                .languages
                .iter()
                .map(|language| language.code().to_owned())
                .collect(),
        }
    }
}

/// The headline baseline run written to `retrieval.json`.
#[derive(Debug, Clone, Serialize)]
pub struct BaselineRun {
    /// Engine configuration exercised.
    pub configuration: EngineConfigSummary,
    /// Dataset facts.
    pub dataset: DatasetInfo,
    /// Measured quality and latency.
    pub results: ModeEvaluation,
}

/// One chunking-strategy comparison row for `chunking.json`.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkingEvaluation {
    /// Strategy name (`fixed:400:80`, ...).
    pub strategy: String,
    /// Chunks the strategy produced for the sample corpus.
    pub chunks_indexed: usize,
    /// Queries evaluated.
    pub queries: usize,
    /// Mean Recall@5 under the full pipeline.
    pub recall_at_5: f64,
    /// Mean MRR under the full pipeline.
    pub mrr: f64,
    /// End-to-end retrieval latency statistics.
    pub total: StageStats,
}

#[derive(Default)]
struct StageAccumulator {
    embed: Vec<f64>,
    dense: Vec<f64>,
    sparse: Vec<f64>,
    fuse: Vec<f64>,
    rerank: Vec<f64>,
    total: Vec<f64>,
}

impl StageAccumulator {
    fn finish(mut self) -> (StageTimings, Option<StageStats>) {
        let timings = StageTimings {
            embed: StageStats::from_samples(&mut self.embed),
            dense: StageStats::from_samples(&mut self.dense),
            sparse: StageStats::from_samples(&mut self.sparse),
            fuse: StageStats::from_samples(&mut self.fuse),
            rerank: StageStats::from_samples(&mut self.rerank),
        };
        (timings, StageStats::from_samples(&mut self.total))
    }
}

/// Builds an engine for `config` and indexes the bundled sample corpus.
///
/// # Errors
/// Returns [`vox_oreo::OreoError`] when engine assembly or indexing fails.
pub async fn build_indexed_engine(
    config: OreoConfig,
) -> Result<(OreoEngine, IndexReport), vox_oreo::OreoError> {
    let engine = OreoEngine::new(config)?;
    let report = engine
        .index_documents(vox_oreo::ingest::sample_corpus())
        .await?;
    Ok((engine, report))
}

/// A default-config engine over a private Tantivy directory.
pub fn bench_config(tantivy_dir: PathBuf, chunking: ChunkingStrategy) -> OreoConfig {
    OreoConfig {
        tantivy_dir,
        chunking,
        ..OreoConfig::default()
    }
}

fn top_document_ids(hits: &[ScoredChunk]) -> Vec<String> {
    hits.iter()
        .take(EVAL_TOP_K)
        .map(|hit| hit.chunk.document_id.clone())
        .collect()
}

fn response_document_ids(response: &RetrievalResponse) -> Vec<String> {
    response
        .documents
        .iter()
        .take(EVAL_TOP_K)
        .map(|doc| {
            doc.metadata
                .get("document_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| doc.id.split('#').next().unwrap_or_default())
                .to_owned()
        })
        .collect()
}

fn eval_query(query: &EvalQuery) -> Query {
    Query {
        text: query.query.clone(),
        normalized_text: String::new(),
        language: Some(query.language),
        top_k: u8::try_from(EVAL_TOP_K).unwrap_or(u8::MAX),
        intent: vox_types::QueryIntent::Unknown,
    }
}

/// Min-max normalizes scores into `[0, 1]` in place (degenerate ranges map
/// to `0.5` so single-hit lists contribute neutrally to the sum).
fn normalize_scores(hits: &mut [ScoredChunk]) {
    let Some((min, max)) =
        hits.iter()
            .map(|hit| hit.score)
            .fold(None::<(f32, f32)>, |acc, score| match acc {
                None => Some((score, score)),
                Some((lo, hi)) => Some((lo.min(score), hi.max(score))),
            })
    else {
        return;
    };
    if (max - min).abs() < f32::EPSILON {
        for hit in hits {
            hit.score = 0.5;
        }
        return;
    }
    for hit in hits {
        hit.score = (hit.score - min) / (max - min);
    }
}

/// Fuses two scored lists by normalized score summation — the non-RRF
/// baseline in the ablation.
fn score_sum_fuse(
    mut dense: Vec<ScoredChunk>,
    mut sparse: Vec<ScoredChunk>,
    limit: usize,
) -> Vec<ScoredChunk> {
    normalize_scores(&mut dense);
    normalize_scores(&mut sparse);
    let mut combined: std::collections::BTreeMap<String, (f32, ScoredChunk)> =
        std::collections::BTreeMap::new();
    for hit in dense.into_iter().chain(sparse) {
        let entry = combined
            .entry(hit.chunk.chunk_id.clone())
            .or_insert_with(|| (0.0, hit.clone()));
        entry.0 += hit.score;
    }
    let mut fused: Vec<(f32, ScoredChunk)> = combined.into_values().collect();
    fused.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.chunk.chunk_id.cmp(&b.1.chunk.chunk_id))
    });
    fused.truncate(limit);
    fused.into_iter().map(|(_score, hit)| hit).collect()
}

async fn fetch_legs(
    engine: &OreoEngine,
    query_text: &str,
    stages: &mut StageAccumulator,
) -> Result<(Vec<ScoredChunk>, Vec<ScoredChunk>), vox_oreo::OreoError> {
    let dense_started = Instant::now();
    let dense = engine
        .dense_retriever()
        .retrieve(query_text, CANDIDATE_TOP)
        .await?;
    stages.dense.push(elapsed_ms(dense_started.elapsed()));

    let sparse_started = Instant::now();
    let sparse = engine
        .sparse_retriever()
        .retrieve(query_text, CANDIDATE_TOP)
        .await?;
    stages.sparse.push(elapsed_ms(sparse_started.elapsed()));

    Ok((dense, sparse))
}

async fn fuse_legs(
    mode: RetrievalMode,
    rrf_k: usize,
    dense: Vec<ScoredChunk>,
    sparse: Vec<ScoredChunk>,
    stages: &mut StageAccumulator,
) -> Vec<ScoredChunk> {
    let fuse_started = Instant::now();
    let fused = if mode == RetrievalMode::ScoreSum {
        score_sum_fuse(dense, sparse, CANDIDATE_TOP)
    } else {
        let dense_records: Vec<_> = dense.into_iter().map(|hit| hit.chunk).collect();
        let sparse_records: Vec<_> = sparse.into_iter().map(|hit| hit.chunk).collect();
        rrf_fuse(&[dense_records, sparse_records], rrf_k, CANDIDATE_TOP)
            .into_iter()
            .map(|(chunk, score)| ScoredChunk { chunk, score })
            .collect()
    };
    stages.fuse.push(elapsed_ms(fuse_started.elapsed()));
    fused
}

/// Runs one query through `mode`, recording per-stage durations into
/// `stages` and returning the top-5 retrieved document ids.
async fn measure_once(
    engine: &OreoEngine,
    mode: RetrievalMode,
    query: &EvalQuery,
    stages: &mut StageAccumulator,
) -> Result<Vec<String>, vox_oreo::OreoError> {
    match mode {
        RetrievalMode::DenseOnly => {
            let started = Instant::now();
            let hits = engine
                .dense_retriever()
                .retrieve(&query.query, CANDIDATE_TOP)
                .await?;
            stages.dense.push(elapsed_ms(started.elapsed()));
            Ok(top_document_ids(&hits))
        }
        RetrievalMode::Bm25Only => {
            let started = Instant::now();
            let hits = engine
                .sparse_retriever()
                .retrieve(&query.query, CANDIDATE_TOP)
                .await?;
            stages.sparse.push(elapsed_ms(started.elapsed()));
            Ok(top_document_ids(&hits))
        }
        RetrievalMode::ScoreSum | RetrievalMode::Rrf => {
            let (dense, sparse) = fetch_legs(engine, &query.query, stages).await?;
            let fused = fuse_legs(mode, engine.config().rrf_k, dense, sparse, stages).await;
            Ok(top_document_ids(&fused))
        }
        RetrievalMode::RrfRerank => {
            let response = engine.retrieve(eval_query(query)).await?;
            if let Some(timings) = response.timings_ms {
                if let Some(value) = timings.embed {
                    stages.embed.push(value);
                }
                if let Some(value) = timings.dense {
                    stages.dense.push(value);
                }
                if let Some(value) = timings.sparse {
                    stages.sparse.push(value);
                }
                if let Some(value) = timings.fuse {
                    stages.fuse.push(value);
                }
                if let Some(value) = timings.rerank {
                    stages.rerank.push(value);
                }
            }
            Ok(response_document_ids(&response))
        }
    }
}

/// Evaluates one retrieval mode over the query set using `engine`'s indexes.
///
/// Runs one untimed-effectively warmup sweep, then [`TIMING_SWEEPS`] measured
/// sweeps; quality metrics are computed once during the first sweep.
///
/// # Errors
/// Returns [`vox_oreo::OreoError`] when any retrieval call fails.
pub async fn evaluate_mode(
    engine: &OreoEngine,
    mode: RetrievalMode,
    queries: &[EvalQuery],
) -> Result<ModeEvaluation, vox_oreo::OreoError> {
    // Warmup: pay one-time costs (Tantivy searcher reload, allocator pools)
    // outside the measured region.
    let mut warmup = StageAccumulator::default();
    for query in queries {
        measure_once(engine, mode, query, &mut warmup).await?;
    }

    let mut stages = StageAccumulator::default();
    let mut recall_sum = 0.0;
    let mut mrr_sum = 0.0;

    for sweep in 0..TIMING_SWEEPS {
        for query in queries {
            let started = Instant::now();
            let retrieved = measure_once(engine, mode, query, &mut stages).await?;
            stages.total.push(elapsed_ms(started.elapsed()));
            if sweep == 0 {
                recall_sum += recall_at_k(&retrieved, &query.relevant, EVAL_TOP_K);
                mrr_sum += mrr(&retrieved, &query.relevant);
            }
        }
    }

    let count = queries.len();
    let (stage_timings, total) = stages.finish();
    let zero_total = StageStats {
        mean: 0.0,
        p50: 0.0,
        p70: 0.0,
        p100: 0.0,
    };
    Ok(ModeEvaluation {
        mode: mode.name().to_owned(),
        queries: count,
        recall_at_5: if count == 0 {
            0.0
        } else {
            recall_sum / count as f64
        },
        mrr: if count == 0 {
            0.0
        } else {
            mrr_sum / count as f64
        },
        stages: stage_timings,
        total: total.unwrap_or(zero_total),
    })
}

/// The production-default sentence chunking parameters.
pub const DEFAULT_SENTENCE_CHUNKING: ChunkingStrategy = ChunkingStrategy::Sentence {
    max_chars: 700,
    min_chars: 80,
};

/// Runs the headline baseline: default production configuration, full
/// pipeline, full latency attribution.
///
/// # Errors
/// Returns [`vox_oreo::OreoError`] when assembly/indexing/retrieval fails.
pub async fn run_baseline(queries: &[EvalQuery]) -> Result<BaselineRun, vox_oreo::OreoError> {
    let dir = tempfile::tempdir().expect("baseline tantivy tempdir");
    let config = bench_config(dir.path().to_path_buf(), DEFAULT_SENTENCE_CHUNKING);
    let summary = EngineConfigSummary::of(&config);
    let (engine, report) = build_indexed_engine(config).await?;
    let results = evaluate_mode(&engine, RetrievalMode::RrfRerank, queries).await?;
    Ok(BaselineRun {
        configuration: summary,
        dataset: dataset_info(&report, queries),
        results,
    })
}

/// Evaluates one chunking strategy under the full pipeline.
///
/// # Errors
/// Returns [`vox_oreo::OreoError`] when assembly/indexing/retrieval fails.
pub async fn evaluate_chunking(
    strategy: ChunkingStrategy,
    label: &str,
    queries: &[EvalQuery],
) -> Result<ChunkingEvaluation, vox_oreo::OreoError> {
    let dir = tempfile::tempdir().expect("chunking tantivy tempdir");
    let config = bench_config(dir.path().to_path_buf(), strategy);
    let (engine, report) = build_indexed_engine(config).await?;
    let evaluation = evaluate_mode(&engine, RetrievalMode::RrfRerank, queries).await?;
    Ok(ChunkingEvaluation {
        strategy: label.to_owned(),
        chunks_indexed: report.chunks,
        queries: evaluation.queries,
        recall_at_5: evaluation.recall_at_5,
        mrr: evaluation.mrr,
        total: evaluation.total,
    })
}

fn dataset_info(report: &IndexReport, queries: &[EvalQuery]) -> DatasetInfo {
    DatasetInfo {
        corpus_documents: report.raw_documents,
        chunks_indexed: report.chunks,
        queries: queries.len(),
        languages: vec!["en".to_owned(), "hi".to_owned(), "ta".to_owned()],
        judgment_method: "hand-authored binary relevance judgments against the \
            bundled sample corpus (crates/vox-bench/data/eval-queries.jsonl)"
            .to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_sum_should_prefer_documents_in_both_lists() {
        let record = |id: &str| vox_oreo::chunk::ChunkRecord {
            document_id: id.to_owned(),
            chunk_id: format!("{id}#0000"),
            chunk_index: 0,
            text: format!("text {id}"),
            language: vox_types::Language::En,
            chunking_strategy: "test".to_owned(),
            source: "test".to_owned(),
        };
        let scored = |id: &str, score: f32| ScoredChunk {
            chunk: record(id),
            score,
        };
        // 'both' appears high in both lists; 'only' tops one list.
        let dense = vec![scored("both", 0.9), scored("other", 0.5)];
        let sparse = vec![scored("both", 12.0), scored("only", 9.0)];
        let fused = score_sum_fuse(dense, sparse, 3);
        assert_eq!(fused[0].chunk.document_id, "both");
    }

    #[tokio::test]
    async fn ablation_modes_should_all_evaluate_without_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = bench_config(dir.path().to_path_buf(), DEFAULT_SENTENCE_CHUNKING);
        let (engine, _report) = build_indexed_engine(config).await.expect("engine");
        let queries = crate::eval::eval_queries();
        for mode in RetrievalMode::all() {
            let evaluation = evaluate_mode(&engine, mode, &queries[..2])
                .await
                .expect("evaluate");
            assert_eq!(evaluation.queries, 2);
            assert!(evaluation.recall_at_5 >= 0.0 && evaluation.recall_at_5 <= 1.0);
            assert!(evaluation.mrr >= 0.0 && evaluation.mrr <= 1.0);
            assert!(evaluation.total.p100 >= evaluation.total.p50);
        }
    }
}
