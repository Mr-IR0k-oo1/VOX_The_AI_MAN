//! OREO CLI: corpus indexing, standalone serving, connectivity verification,
//! and local latency benchmarking.
//!
//! ```text
//! oreo index  --input data/msmarco-xi/     # build dense + BM25 indexes
//! oreo serve                              # POST /v1/retrieve on :8090
//! oreo verify                             # check Qdrant + Tantivy health
//! oreo bench --queries 50                 # p50/p95 retrieval latency
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};
use vox_oreo::config::OreoConfig;
use vox_oreo::engine::OreoEngine;
use vox_types::Query;

#[derive(Debug, Parser)]
#[command(name = "oreo", about = "OREO multilingual retrieval engine", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Build the dense (Qdrant/memory) and sparse (Tantivy) indexes.
    Index {
        /// Corpus file or directory (.jsonl/.json). Defaults to the bundled
        /// sample corpus.
        #[arg(long)]
        input: Option<PathBuf>,
        /// Override the Tantivy index directory.
        #[arg(long)]
        tantivy_dir: Option<PathBuf>,
        /// Override the vector store kind (`memory` or `qdrant`).
        #[arg(long)]
        vector_store: Option<String>,
        /// Override the Qdrant base URL.
        #[arg(long)]
        qdrant_url: Option<String>,
    },
    /// Serve `POST /v1/retrieve` and `GET /health`.
    Serve {
        /// Bind host.
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Bind port.
        #[arg(long, default_value_t = 8090)]
        port: u16,
    },
    /// Verify Qdrant connectivity and Tantivy index health.
    Verify,
    /// Benchmark end-to-end retrieval latency with fixed queries.
    Bench {
        /// Number of timed query executions.
        #[arg(long, default_value_t = 30)]
        queries: usize,
    },
}

fn apply_overrides(mut config: OreoConfig, cli: &Command) -> OreoConfig {
    if let Command::Index {
        tantivy_dir,
        vector_store,
        qdrant_url,
        ..
    } = cli
    {
        if let Some(dir) = tantivy_dir {
            config.tantivy_dir = dir.clone();
        }
        match vector_store.as_deref() {
            Some("memory") => config.vector_store = vox_oreo::config::VectorStoreKind::Memory,
            Some("qdrant") => config.vector_store = vox_oreo::config::VectorStoreKind::Qdrant,
            _ => {}
        }
        if let Some(url) = qdrant_url {
            config.qdrant_url = url.trim().trim_end_matches('/').to_owned();
        }
    }
    config
}

async fn build_engine(cli: &Command) -> Result<Arc<OreoEngine>, Box<dyn std::error::Error>> {
    let config = apply_overrides(OreoConfig::from_env()?, cli);
    Ok(Arc::new(OreoEngine::new(config)?))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Index { input, .. } => {
            let engine = build_engine(&cli.command).await?;
            let report = match input {
                Some(path) => engine.index_corpus(path).await?,
                None => {
                    engine
                        .index_documents(vox_oreo::ingest::sample_corpus())
                        .await?
                }
            };
            println!(
                "indexed {} raw → {} documents → {} chunks (skipped: {} too short, {} unsupported language)",
                report.raw_documents,
                report.documents,
                report.chunks,
                report.stats.skipped_too_short,
                report.stats.skipped_unsupported_language,
            );
            let (dense, sparse) = engine.verify().await?;
            println!("dense: {dense}; sparse: {sparse}");
        }
        Command::Serve { host, port } => {
            let engine = build_engine(&cli.command).await?;
            let (dense, sparse) = engine.verify().await?;
            tracing::info!(%dense, %sparse, "indexes ready");
            let app = vox_oreo::service::build_router(engine);
            let addr = format!("{host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            tracing::info!(%addr, "oreo listening");
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
        Command::Verify => {
            let engine = build_engine(&cli.command).await?;
            let (dense, sparse) = engine.verify().await?;
            println!("OK dense={dense} sparse={sparse}");
        }
        Command::Bench { queries } => {
            let engine = build_engine(&cli.command).await?;
            run_bench(&engine, *queries).await;
        }
    }
    Ok(())
}

/// Fixed multilingual probe queries mirroring expected production traffic.
const BENCH_QUERIES: [&str; 6] = [
    "What is GST and who administers it?",
    "How do I update my Aadhaar address online?",
    "जीएसटी पंजीकरण कब अनिवार्य है?",
    "आधार पता अपडेट कैसे करें?",
    "ஜிஎஸ்டி என்றால் என்ன?",
    "சென்னை மெட்ரோ நீர் இணைப்பு விண்ணப்பிக்க",
];

async fn run_bench(engine: &Arc<OreoEngine>, iterations: usize) {
    let mut totals_ms: Vec<f64> = Vec::with_capacity(iterations * BENCH_QUERIES.len());
    let mut stage_sums = [0.0f64; 5];
    for round in 0..iterations.max(1) {
        for text in BENCH_QUERIES {
            let query = Query {
                query: text.to_owned(),
                language: vox_types::Language::En,
                top_k: 5,
                intent: vox_types::QueryIntent::Unknown,
            };
            let started = Instant::now();
            let response = match engine.retrieve(query).await {
                Ok(response) => response,
                Err(err) => {
                    eprintln!("bench query failed: {err}");
                    continue;
                }
            };
            let _ = round;
            if let Some(timings) = response.timings_ms {
                stage_sums[0] += timings.embed.unwrap_or_default();
                stage_sums[1] += timings.dense.unwrap_or_default();
                stage_sums[2] += timings.sparse.unwrap_or_default();
                stage_sums[3] += timings.fuse.unwrap_or_default();
                stage_sums[4] += timings.rerank.unwrap_or_default();
            }
            totals_ms.push(vox_oreo::engine::elapsed_ms(started.elapsed()));
        }
    }
    totals_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let count = totals_ms.len().max(1);
    let percentile = |p: f64| -> f64 {
        let index = ((p / 100.0) * (totals_ms.len() as f64 - 1.0)).round() as usize;
        totals_ms[index.min(totals_ms.len() - 1)]
    };
    println!("queries executed: {}", totals_ms.len());
    println!(
        "latency ms: mean={:.2} p50={:.2} p95={:.2} max={:.2}",
        totals_ms.iter().sum::<f64>() / count as f64,
        percentile(50.0),
        percentile(95.0),
        totals_ms.last().copied().unwrap_or_default(),
    );
    let labels = ["embed", "dense", "sparse", "fuse", "rerank"];
    for (label, sum) in labels.iter().zip(stage_sums) {
        println!("stage {label}: mean {:.3} ms", sum / count as f64);
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}
