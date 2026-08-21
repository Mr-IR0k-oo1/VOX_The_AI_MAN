//! `vox-bench` CLI: runs the Phase 3 retrieval evaluation suite and writes
//! machine- plus human-readable reports.
//!
//! Usage: `cargo run -p vox-bench -- [output-dir]` (default: `benchmarks`).

use std::path::PathBuf;

use serde::Serialize;
use vox_bench::environment::detect_environment;
use vox_bench::eval::eval_queries;
use vox_bench::report::render_report;
use vox_bench::runner::{evaluate_chunking, evaluate_mode, run_baseline, RetrievalMode};
use vox_oreo::chunk::ChunkingStrategy;

/// Everything written to `retrieval.json`.
#[derive(Serialize)]
struct RetrievalJson<'a> {
    environment: &'a vox_bench::EnvironmentInfo,
    configuration: &'a vox_bench::runner::EngineConfigSummary,
    dataset: &'a vox_bench::runner::DatasetInfo,
    results: &'a vox_bench::runner::ModeEvaluation,
}

/// Everything written to `ablation.json`.
#[derive(Serialize)]
struct AblationJson<'a> {
    environment: &'a vox_bench::EnvironmentInfo,
    dataset_queries: usize,
    modes: &'a [vox_bench::runner::ModeEvaluation],
}

/// Everything written to `chunking.json`.
#[derive(Serialize)]
struct ChunkingJson<'a> {
    environment: &'a vox_bench::EnvironmentInfo,
    strategies: &'a [vox_bench::runner::ChunkingEvaluation],
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "benchmarks".to_owned()),
    );
    std::fs::create_dir_all(&out_dir)?;

    let queries = eval_queries();
    println!(
        "evaluation set: {} queries (en/hi/ta) against the bundled sample corpus",
        queries.len()
    );

    // Baseline: default production configuration, full pipeline.
    let baseline = run_baseline(&queries).await?;
    println!(
        "baseline: recall@5={:.3} mrr={:.3} p50={:.2}ms",
        baseline.results.recall_at_5, baseline.results.mrr, baseline.results.total.p50
    );

    // Component ablation on the same indexed engine as the baseline would be
    // ideal, but the baseline engine is consumed by the baseline run; build a
    // fresh identical engine for the ablation instead.
    let ablation_dir = tempfile::tempdir()?;
    let config = vox_bench::runner::bench_config(
        ablation_dir.path().to_path_buf(),
        vox_bench::runner::DEFAULT_SENTENCE_CHUNKING,
    );
    let (engine, _report) = vox_bench::runner::build_indexed_engine(config).await?;
    let mut ablation = Vec::new();
    for mode in RetrievalMode::all() {
        let evaluation = evaluate_mode(&engine, mode, &queries).await?;
        println!(
            "ablation {}: recall@5={:.3} mrr={:.3} p50={:.2}ms",
            evaluation.mode, evaluation.recall_at_5, evaluation.mrr, evaluation.total.p50
        );
        ablation.push(evaluation);
    }
    drop(engine);
    drop(ablation_dir);

    // Chunking strategy comparison under the full pipeline.
    let strategies = [
        (
            "fixed:400:80",
            ChunkingStrategy::Fixed {
                size: 400,
                overlap: 80,
            },
        ),
        (
            "sentence:700:80",
            vox_bench::runner::DEFAULT_SENTENCE_CHUNKING,
        ),
        (
            "sliding:600:300",
            ChunkingStrategy::Sliding {
                window: 600,
                stride: 300,
            },
        ),
    ];
    let mut chunkings = Vec::new();
    for (label, strategy) in strategies {
        let evaluation = evaluate_chunking(strategy, label, &queries).await?;
        println!(
            "chunking {}: chunks={} recall@5={:.3} mrr={:.3} p50={:.2}ms",
            evaluation.strategy,
            evaluation.chunks_indexed,
            evaluation.recall_at_5,
            evaluation.mrr,
            evaluation.total.p50
        );
        chunkings.push(evaluation);
    }

    let environment = detect_environment();

    write_json(
        &out_dir.join("retrieval.json"),
        &RetrievalJson {
            environment: &environment,
            configuration: &baseline.configuration,
            dataset: &baseline.dataset,
            results: &baseline.results,
        },
    )?;
    write_json(
        &out_dir.join("ablation.json"),
        &AblationJson {
            environment: &environment,
            dataset_queries: queries.len(),
            modes: &ablation,
        },
    )?;
    write_json(
        &out_dir.join("chunking.json"),
        &ChunkingJson {
            environment: &environment,
            strategies: &chunkings,
        },
    )?;

    let markdown = render_report(&environment, &baseline, &ablation, &chunkings);
    std::fs::write(out_dir.join("retrieval_report.md"), markdown)?;

    println!("reports written to {}", out_dir.display());
    Ok(())
}

fn write_json<T: Serialize>(
    path: &std::path::Path,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    let pretty = serde_json::to_string_pretty(value)?;
    std::fs::write(path, pretty + "\n")?;
    Ok(())
}
