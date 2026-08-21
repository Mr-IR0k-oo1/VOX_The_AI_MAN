//! Human-readable Markdown report generation from measured results.
//!
//! Every number rendered here comes from the measured result structs; the
//! conclusions section is derived programmatically from those same values.

use crate::environment::EnvironmentInfo;
use crate::runner::{BaselineRun, ChunkingEvaluation, ModeEvaluation};

/// Renders the full `retrieval_report.md` document.
#[must_use]
pub fn render_report(
    environment: &EnvironmentInfo,
    baseline: &BaselineRun,
    ablation: &[ModeEvaluation],
    chunkings: &[ChunkingEvaluation],
) -> String {
    let mut out = String::new();
    push_header(&mut out, environment, baseline);
    push_methodology(&mut out, baseline);
    push_chunking_results(&mut out, chunkings);
    push_ablation_results(&mut out, ablation);
    push_baseline_latency(&mut out, baseline);
    push_conclusions(&mut out, ablation, chunkings);
    push_limitations(&mut out);
    out
}

fn push_header(out: &mut String, environment: &EnvironmentInfo, baseline: &BaselineRun) {
    out.push_str("# Phase 3 — Retrieval Evaluation Report\n\n");
    out.push_str(&format!(
        "Measured: {} (Unix epoch {})\n\n",
        environment.measured_at_date, environment.measured_at_unix_secs
    ));
    out.push_str("## Environment\n\n");
    out.push_str("| Property | Value |\n|---|---|\n");
    row(out, "OS", &environment.os);
    row(out, "Architecture", &environment.arch);
    row(out, "CPU", &environment.cpu_model);
    row(
        out,
        "Logical parallelism",
        &environment.parallelism.to_string(),
    );
    row(
        out,
        "Build profile",
        &format!(
            "{} (cargo, opt-level of `{}`)",
            environment.profile,
            if environment.profile == "debug" {
                "dev"
            } else {
                "release"
            }
        ),
    );
    row(
        out,
        "Corpus",
        &format!(
            "{} documents → {} chunks (bundled sample corpus)",
            baseline.dataset.corpus_documents, baseline.dataset.chunks_indexed
        ),
    );
    row(
        out,
        "Queries",
        &format!(
            "{} judged queries across {:?}",
            baseline.dataset.queries, baseline.dataset.languages
        ),
    );
    out.push('\n');
}

fn push_methodology(out: &mut String, baseline: &BaselineRun) {
    out.push_str("## Methodology\n\n");
    out.push_str("- **Relevance judgments**: ");
    out.push_str(&baseline.dataset.judgment_method);
    out.push_str(". Judgments are binary at the document level; every query has 1–2 relevant documents in a 20-document corpus.\n");
    out.push_str("- **Quality metrics**: Recall@5 = |top-5 ∩ relevant| / |relevant| averaged over queries; MRR = mean over queries of 1/rank of the first relevant document in the top 5.\n");
    out.push_str("- **Latency measurement**: `std::time::Instant` wall-clock around each stage. One warmup sweep runs before measurement; each mode then executes 3 measured sweeps over all queries and percentiles are computed over the pooled samples.\n");
    out.push_str("- **Stage attribution**: the dense leg's timing includes its internal query embedding; BM25 has no embedding step. The full-pipeline mode (`dense+bm25_rrf+rerank`) reports the production engine's own stage timings, where query embedding is timed once up front and again inside the dense leg.\n");
    out.push_str("- **Determinism**: hashed embeddings, memory vector store, Tantivy BM25, RRF, and the lexical reranker are all deterministic; quality metrics are exact for this build.\n\n");

    let config = &baseline.configuration;
    out.push_str("## Configuration under test\n\n");
    out.push_str("| Setting | Value |\n|---|---|\n");
    row(out, "Chunking", &config.chunking);
    row(out, "Embedding dim", &config.embedding_dim.to_string());
    row(out, "Dense store", &config.vector_store);
    row(out, "Candidate pool", &config.candidate_top.to_string());
    row(out, "Final top-k", &config.final_top.to_string());
    row(out, "RRF k", &config.rrf_k.to_string());
    row(out, "Reranker", &config.reranker);
    row(out, "Languages", &config.languages.join(","));
    out.push('\n');
}

fn push_chunking_results(out: &mut String, chunkings: &[ChunkingEvaluation]) {
    out.push_str("## Chunking strategy comparison\n\n");
    out.push_str("Full pipeline (hybrid + RRF + rerank) per chunking strategy:\n\n");
    out.push_str("| Strategy | Chunks indexed | Recall@5 | MRR | P50 ms | P70 ms | P100 ms |\n");
    out.push_str("|---|---|---|---|---|---|---|\n");
    for entry in chunkings {
        out.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} | {:.2} | {:.2} | {:.2} |\n",
            entry.strategy,
            entry.chunks_indexed,
            entry.recall_at_5,
            entry.mrr,
            entry.total.p50,
            entry.total.p70,
            entry.total.p100,
        ));
    }
    out.push('\n');
}

fn push_ablation_results(out: &mut String, ablation: &[ModeEvaluation]) {
    out.push_str("## Retrieval component ablation\n\n");
    out.push_str("Sentence chunking held fixed; components added left to right:\n\n");
    out.push_str("| Mode | Recall@5 | MRR | P50 ms | P70 ms | P100 ms |\n");
    out.push_str("|---|---|---|---|---|---|\n");
    for mode in ablation {
        out.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.2} | {:.2} | {:.2} |\n",
            mode.mode, mode.recall_at_5, mode.mrr, mode.total.p50, mode.total.p70, mode.total.p100,
        ));
    }
    out.push('\n');

    out.push_str("### Per-stage latency (mean ms)\n\n");
    out.push_str("| Mode | embed | dense | sparse/bm25 | fuse | rerank |\n");
    out.push_str("|---|---|---|---|---|---|\n");
    for mode in ablation {
        let cell = |value: Option<crate::runner::StageStats>| match value {
            Some(stats) => format!("{:.3}", stats.mean),
            None => "—".to_owned(),
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            mode.mode,
            cell(mode.stages.embed),
            cell(mode.stages.dense),
            cell(mode.stages.sparse),
            cell(mode.stages.fuse),
            cell(mode.stages.rerank),
        ));
    }
    out.push('\n');
}

fn push_baseline_latency(out: &mut String, baseline: &BaselineRun) {
    let total = &baseline.results.total;
    out.push_str("## Baseline latency (default configuration)\n\n");
    out.push_str("| Metric | P50 ms | P70 ms | P100 ms | Mean ms |\n");
    out.push_str("|---|---|---|---|---|\n");
    out.push_str(&format!(
        "| Total retrieval | {:.2} | {:.2} | {:.2} | {:.2} |\n\n",
        total.p50, total.p70, total.p100, total.mean
    ));

    out.push_str("### Baseline stage latencies\n\n");
    out.push_str("| Stage | Mean ms | P50 ms | P70 ms | P100 ms |\n");
    out.push_str("|---|---|---|---|---|\n");
    let stages = [
        ("embed", baseline.results.stages.embed),
        ("dense", baseline.results.stages.dense),
        ("sparse/bm25", baseline.results.stages.sparse),
        ("fuse/rrf", baseline.results.stages.fuse),
        ("rerank", baseline.results.stages.rerank),
    ];
    for (name, stats) in stages {
        if let Some(stats) = stats {
            out.push_str(&format!(
                "| {name} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
                stats.mean, stats.p50, stats.p70, stats.p100
            ));
        }
    }
    out.push('\n');
}

fn push_conclusions(
    out: &mut String,
    ablation: &[ModeEvaluation],
    chunkings: &[ChunkingEvaluation],
) {
    out.push_str("## Conclusions (derived from the measurements above)\n\n");

    if let Some(best) = best_chunking(chunkings) {
        let tied = chunkings
            .iter()
            .filter(|entry| entry.partial_cmp_quality(best) == std::cmp::Ordering::Equal)
            .count();
        if tied > 1 {
            out.push_str(&format!(
                "- **Best chunking strategy**: {tied} strategies tie on quality (Recall@5 {:.3}, MRR {:.3}); `{}` is fastest at P50 {:.2} ms.\n",
                best.recall_at_5, best.mrr, best.strategy, best.total.p50
            ));
        } else {
            out.push_str(&format!(
                "- **Best chunking strategy**: `{}` (Recall@5 {:.3}, MRR {:.3}).\n",
                best.strategy, best.recall_at_5, best.mrr
            ));
        }
    }

    let by_name = |name: &str| {
        ablation
            .iter()
            .find(|mode| mode.mode == name)
            .unwrap_or_else(|| panic!("ablation missing mode {name}"))
    };
    let dense = by_name("dense_only");
    let bm25 = by_name("bm25_only");
    let score_sum = by_name("dense+bm25_score_sum");
    let rrf = by_name("dense+bm25_rrf");
    let rerank = by_name("dense+bm25_rrf+rerank");

    let best_single = if dense.recall_at_5 >= bm25.recall_at_5 {
        dense
    } else {
        bm25
    };
    out.push_str(&format!(
        "- **Hybrid vs single-leg**: best single leg is `{}` at Recall@5 {:.3}; score-sum fusion reaches {:.3} ({:+.3}) and RRF fusion reaches {:.3} ({:+.3}).\n",
        best_single.mode,
        best_single.recall_at_5,
        score_sum.recall_at_5,
        score_sum.recall_at_5 - best_single.recall_at_5,
        rrf.recall_at_5,
        rrf.recall_at_5 - best_single.recall_at_5,
    ));

    out.push_str(&format!(
        "- **Does reranking improve quality?** RRF alone: Recall@5 {:.3}, MRR {:.3}. RRF + lexical rerank: Recall@5 {:.3} ({:+.3}), MRR {:.3} ({:+.3}).\n",
        rrf.recall_at_5,
        rrf.mrr,
        rerank.recall_at_5,
        rerank.recall_at_5 - rrf.recall_at_5,
        rerank.mrr,
        rerank.mrr - rrf.mrr,
    ));

    if let Some((stage, stats)) = dominant_stage(rerank) {
        let share = if rerank.total.mean > 0.0 {
            100.0 * stats.mean / rerank.total.mean
        } else {
            0.0
        };
        out.push_str(&format!(
            "- **Dominant latency stage**: `{stage}` at mean {:.2} ms (~{share:.0}% of total retrieval latency).\n",
            stats.mean
        ));
    }

    let total = &rerank.total;
    out.push_str(&format!(
        "- **Retrieval latency** (full pipeline): P50 {:.2} ms, P70 {:.2} ms, P100 {:.2} ms.\n",
        total.p50, total.p70, total.p100
    ));

    let best_mode = ablation
        .iter()
        .max_by(|a, b| a.partial_cmp_quality(b))
        .expect("ablation non-empty");
    out.push_str(&format!(
        "- **Best retrieval configuration overall**: `{}` (Recall@5 {:.3}, MRR {:.3}, P50 {:.2} ms).\n",
        best_mode.mode, best_mode.recall_at_5, best_mode.mrr, best_mode.total.p50
    ));
    out.push('\n');
}

trait QualityOrder {
    fn partial_cmp_quality(&self, other: &Self) -> std::cmp::Ordering;
}

impl QualityOrder for ModeEvaluation {
    fn partial_cmp_quality(&self, other: &Self) -> std::cmp::Ordering {
        self.recall_at_5
            .partial_cmp(&other.recall_at_5)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                self.mrr
                    .partial_cmp(&other.mrr)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    }
}

impl QualityOrder for ChunkingEvaluation {
    fn partial_cmp_quality(&self, other: &Self) -> std::cmp::Ordering {
        self.recall_at_5
            .partial_cmp(&other.recall_at_5)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                self.mrr
                    .partial_cmp(&other.mrr)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    }
}

/// Highest-quality chunking; ties broken by fastest P50.
fn best_chunking(chunkings: &[ChunkingEvaluation]) -> Option<&ChunkingEvaluation> {
    let best = chunkings.iter().max_by(|a, b| a.partial_cmp_quality(b))?;
    chunkings
        .iter()
        .filter(|entry| entry.partial_cmp_quality(best) == std::cmp::Ordering::Equal)
        .min_by(|a, b| {
            a.total
                .p50
                .partial_cmp(&b.total.p50)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// The slowest measured stage of a full-pipeline evaluation.
fn dominant_stage(
    evaluation: &ModeEvaluation,
) -> Option<(&'static str, crate::runner::StageStats)> {
    let stages = [
        ("embed", evaluation.stages.embed),
        ("dense", evaluation.stages.dense),
        ("sparse/bm25", evaluation.stages.sparse),
        ("fuse/rrf", evaluation.stages.fuse),
        ("rerank", evaluation.stages.rerank),
    ];
    stages
        .into_iter()
        .filter_map(|(name, stats)| stats.map(|value| (name, value)))
        .max_by(|a, b| {
            a.1.mean
                .partial_cmp(&b.1.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn push_limitations(out: &mut String) {
    out.push_str("## Limitations\n\n");
    out.push_str("- The corpus is the bundled 20-document sample; absolute numbers will not transfer to production-scale corpora.\n");
    out.push_str("- Judgments are hand-authored binary labels (1–2 relevant documents per query), so Recall@5 saturates quickly; treat cross-config deltas as the signal, not absolute values.\n");
    out.push_str("- Queries are within-language (an English query judges English documents, etc.); cross-lingual retrieval is not evaluated here.\n");
    out.push_str("- The embedder is the deterministic offline hashed-embedding placeholder, not a neural multilingual encoder; dense-leg quality reflects that.\n");
    out.push_str("- Latencies are from a developer machine in debug profile unless stated otherwise; they are useful for relative comparisons between configurations, not as production SLOs.\n");
}

fn row(out: &mut String, key: &str, value: &str) {
    out.push_str(&format!("| {key} | {value} |\n"));
}
