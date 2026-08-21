//! `vox-bench` CLI: two harnesses over one binary.
//!
//! Default — VOX performance harness (Phase 7). Measures the actual
//! pipeline instead of guessing:
//!
//! - `L0` — query → retrieval (the OREO boundary alone)
//! - `L1` — query → analysis → retrieval → grounding → LLM → guardrails
//! - `L2` — audio → STT → (full L1 tail)
//!
//! Every execution of the scenario set records one sample with the
//! pipeline's own per-stage timings plus an externally measured wall-clock
//! total. Raw samples and computed summaries (mean, median, P50, P70, P100)
//! are written as JSON so nothing is cherry-picked or transcribed by hand.
//!
//! Run in release mode for defensible numbers:
//!
//! ```sh
//! cargo run -p vox-bench --release -- --rounds 100 --warmup 10 --out benchmarks
//! ```
//!
//! `eval` subcommand — Phase 3 retrieval evaluation suite: hand-judged
//! multilingual query set, chunking-strategy comparison, component ablation,
//! and quality metrics written as machine- plus human-readable reports:
//!
//! ```sh
//! cargo run -p vox-bench -- eval [output-dir]
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use vox_bench::{summarize, SampleSummary};
use vox_core::VoxPipeline;
use vox_grounding::GroundingConfig;
use vox_guard::GuardService;
use vox_llm::ExtractiveProvider;
use vox_retrieval::{MockRetrievalClient, OREORetrievalClient, RetrievalClient, RetryPolicy};
use vox_stt::MockRecognizer;
use vox_types::{ms, AudioFormat, Language, Query, VoiceRequest};

/// One measured execution: wall-clock total plus the pipeline's own
/// per-stage timings (present stages only).
#[derive(Debug, Serialize)]
struct RunSample {
    scenario: &'static str,
    total_ms: f64,
    stages: BTreeMap<String, f64>,
}

/// Everything recorded for one benchmark level.
#[derive(Serialize)]
struct LevelReport {
    level: &'static str,
    description: &'static str,
    generated_at_unix_secs: u64,
    profile: &'static str,
    rounds: usize,
    warmup_rounds: usize,
    scenarios: Vec<&'static str>,
    backends: BTreeMap<&'static str, String>,
    runs: Vec<RunSample>,
    total_summary: SampleSummary,
    stage_summaries: BTreeMap<String, SampleSummary>,
}

struct Scenario {
    id: &'static str,
    text: &'static str,
    language: Option<Language>,
}

/// Meaningful test set: three languages across the supported corpus, one
/// legitimate in-domain question without corpus coverage, one off-topic
/// question. Exercises answer generation and both refusal paths.
const SCENARIOS: [Scenario; 5] = [
    Scenario {
        id: "supported_en",
        text: "What is artificial intelligence?",
        language: Some(Language::En),
    },
    Scenario {
        id: "supported_hi",
        text: "जीएसटी क्या है?",
        language: Some(Language::Hi),
    },
    Scenario {
        id: "supported_ta",
        text: "குங்குமப்பூ என்றால் என்ன?",
        language: None,
    },
    Scenario {
        id: "unsupported_itr",
        text: "how to file itr online",
        language: Some(Language::En),
    },
    Scenario {
        id: "offtopic_cricket",
        text: "who will win the cricket world cup",
        language: Some(Language::En),
    },
];

struct Args {
    levels: Vec<String>,
    rounds: usize,
    warmup: usize,
    out: PathBuf,
    retrieval_url: Option<String>,
}

fn print_usage() {
    println!(
        "usage: vox-bench [--level all|L0|L1|L2] [--rounds N] [--warmup N] \
         [--out DIR] [--retrieval-url URL]\n       \
         vox-bench eval [output-dir]\n       \
         vox-bench multilingual [output-dir] [--rounds N]"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        levels: vec!["L0".to_owned(), "L1".to_owned(), "L2".to_owned()],
        rounds: 100,
        warmup: 10,
        out: PathBuf::from("benchmarks"),
        retrieval_url: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--level" => {
                let value = it.next().ok_or("--level needs a value")?;
                args.levels = match value.as_str() {
                    "all" => vec!["L0".to_owned(), "L1".to_owned(), "L2".to_owned()],
                    other => vec![other.to_owned()],
                };
            }
            "--rounds" => {
                args.rounds = it
                    .next()
                    .ok_or("--rounds needs a value")?
                    .parse()
                    .map_err(|_| "--rounds must be a number")?;
            }
            "--warmup" => {
                args.warmup = it
                    .next()
                    .ok_or("--warmup needs a value")?
                    .parse()
                    .map_err(|_| "--warmup must be a number")?;
            }
            "--out" => {
                args.out = PathBuf::from(it.next().ok_or("--out needs a value")?);
            }
            "--retrieval-url" => {
                args.retrieval_url = Some(it.next().ok_or("--retrieval-url needs a value")?);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    Ok(args)
}

struct Backends {
    retrieval_name: String,
    llm_name: &'static str,
    stt_name: &'static str,
    retrieval: Arc<dyn RetrievalClient>,
    topic_vocabulary: Vec<String>,
}

fn build_backends(retrieval_url: Option<&str>) -> Backends {
    match retrieval_url {
        None => Backends {
            retrieval_name: "mock".to_owned(),
            llm_name: "extractive",
            stt_name: "mock",
            retrieval: Arc::new(MockRetrievalClient::default()),
            topic_vocabulary: MockRetrievalClient::topic_vocabulary(),
        },
        Some(url) => {
            let client = OREORetrievalClient::new(
                url,
                std::time::Duration::from_millis(800),
                RetryPolicy::default(),
            )
            .expect("OREO client builds");
            Backends {
                retrieval_name: format!("http ({url})"),
                llm_name: "extractive",
                stt_name: "mock",
                // The domain of a remote corpus is unknown up front, matching
                // production wiring: off-topic input checks stay disabled.
                topic_vocabulary: Vec::new(),
                retrieval: Arc::new(client),
            }
        }
    }
}

fn build_pipeline(backends: &Backends, transcript: (&str, Option<Language>)) -> VoxPipeline {
    VoxPipeline::new(
        Arc::new(MockRecognizer::with_transcript(transcript.0, transcript.1)),
        Arc::clone(&backends.retrieval),
        Arc::new(ExtractiveProvider),
        GroundingConfig::default(),
        Arc::new(GuardService::new(backends.topic_vocabulary.clone())),
    )
}

fn stage_map(metrics: &vox_types::LatencyMetrics) -> BTreeMap<String, f64> {
    let mut stages = BTreeMap::new();
    for (name, value) in [
        ("stt", metrics.stt),
        ("language", metrics.language),
        ("query_analysis", metrics.query_analysis),
        ("retrieval", metrics.retrieval),
        ("grounding", metrics.grounding),
        ("guardrail", metrics.guardrail),
        ("llm", metrics.llm),
    ] {
        if let Some(v) = value {
            stages.insert(name.to_owned(), v);
        }
    }
    stages
}

async fn run_l0(backends: &Backends, rounds: usize) -> Vec<RunSample> {
    let mut samples = Vec::with_capacity(rounds * SCENARIOS.len());
    for round in 0..rounds {
        for scenario in &SCENARIOS {
            let query = Query::new(scenario.text, scenario.language, 3);
            let started = Instant::now();
            let response = backends.retrieval.retrieve(query).await.expect("retrieve");
            let total = ms(started.elapsed());
            assert!(
                !response.documents.is_empty(),
                "scenario {} returned no documents in round {round}",
                scenario.id
            );
            samples.push(RunSample {
                scenario: scenario.id,
                total_ms: total,
                stages: BTreeMap::new(),
            });
        }
    }
    samples
}

async fn run_l1(backends: &Backends, rounds: usize) -> Vec<RunSample> {
    let pipeline = build_pipeline(backends, ("bench", None));
    let mut samples = Vec::with_capacity(rounds * SCENARIOS.len());
    for round in 0..rounds {
        for (idx, scenario) in SCENARIOS.iter().enumerate() {
            let query = Query::new(scenario.text, scenario.language, 3);
            let started = Instant::now();
            let response = pipeline
                .run_text(format!("bench-l1-{round}-{idx}"), query)
                .await
                .expect("pipeline run");
            let total = ms(started.elapsed());
            assert_eq!(
                response.refusal_reason.is_some(),
                response.answerability != vox_types::Answerability::Supported
            );
            samples.push(RunSample {
                scenario: scenario.id,
                total_ms: total,
                stages: stage_map(&response.metrics),
            });
        }
    }
    samples
}

async fn run_l2(backends: &Backends, rounds: usize) -> Vec<RunSample> {
    // One pipeline per scenario: the mock recognizer is fixed at
    // construction, mirroring one spoken utterance per scenario.
    let pipelines: Vec<_> = SCENARIOS
        .iter()
        .map(|scenario| build_pipeline(backends, (scenario.text, scenario.language)))
        .collect();
    let audio = "aGVsbG8=";
    let mut samples = Vec::with_capacity(rounds * SCENARIOS.len());
    for round in 0..rounds {
        for (idx, scenario) in SCENARIOS.iter().enumerate() {
            let request = VoiceRequest {
                audio_base64: audio.to_owned(),
                format: AudioFormat::Wav,
                language: None,
                top_k: 3,
            };
            let started = Instant::now();
            let response = pipelines[idx]
                .run_voice(format!("bench-l2-{round}-{idx}"), request)
                .await
                .expect("pipeline run");
            let total = ms(started.elapsed());
            assert_eq!(response.transcript.text, scenario.text);
            samples.push(RunSample {
                scenario: scenario.id,
                total_ms: total,
                stages: stage_map(&response.metrics),
            });
        }
    }
    samples
}

fn level_report(
    level: &'static str,
    description: &'static str,
    args: &Args,
    backends: &Backends,
    runs: Vec<RunSample>,
) -> LevelReport {
    let totals: Vec<f64> = runs.iter().map(|run| run.total_ms).collect();
    let total_summary = summarize(&totals).expect("non-empty totals");

    let mut stage_values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for run in &runs {
        for (stage, value) in &run.stages {
            stage_values.entry(stage.clone()).or_default().push(*value);
        }
    }
    let stage_summaries: BTreeMap<String, SampleSummary> = stage_values
        .into_iter()
        .filter_map(|(stage, values)| summarize(&values).map(|s| (stage, s)))
        .collect();

    let mut backends_map = BTreeMap::new();
    backends_map.insert("stt", backends.stt_name.to_owned());
    backends_map.insert("retrieval", backends.retrieval_name.clone());
    backends_map.insert("llm", backends.llm_name.to_owned());

    LevelReport {
        level,
        description,
        generated_at_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_secs(),
        profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        rounds: args.rounds,
        warmup_rounds: args.warmup,
        scenarios: SCENARIOS.iter().map(|s| s.id).collect(),
        backends: backends_map,
        runs,
        total_summary,
        stage_summaries,
    }
}

fn print_report(report: &LevelReport) {
    println!("== {} ({}) ==", report.level, report.description);
    println!(
        "{:>10} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "series", "count", "mean", "p50", "p70", "p100"
    );
    let t = &report.total_summary;
    println!(
        "{:>10} {:>10} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
        "total", t.count, t.mean_ms, t.p50_ms, t.p70_ms, t.p100_ms
    );
    for (stage, s) in &report.stage_summaries {
        println!(
            "{:>10} {:>10} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            stage, s.count, s.mean_ms, s.p50_ms, s.p70_ms, s.p100_ms
        );
    }
    println!();
}

fn output_path(out: &Path, level: &str) -> PathBuf {
    match level {
        "L0" => out.join("retrieval.json"),
        "L1" => out.join("text_e2e.json"),
        "L2" => out.join("voice_e2e.json"),
        other => panic!("unknown level {other}"),
    }
}

async fn e2e_main() {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("{err}");
        print_usage();
        std::process::exit(2);
    });
    let backends = build_backends(args.retrieval_url.as_deref());

    std::fs::create_dir_all(&args.out).expect("output directory");

    for level in &args.levels {
        let (name, description, runs) = match level.as_str() {
            "L0" => (
                "L0",
                "query -> retrieval boundary",
                run_l0(&backends, args.rounds).await,
            ),
            "L1" => (
                "L1",
                "query -> analysis -> retrieval -> grounding -> LLM -> guardrails",
                run_l1(&backends, args.rounds).await,
            ),
            "L2" => (
                "L2",
                "audio -> STT -> full text tail",
                run_l2(&backends, args.rounds).await,
            ),
            other => {
                eprintln!("unknown level: {other}");
                print_usage();
                std::process::exit(2);
            }
        };

        // Warmup executes the same paths untimed so allocator/cache state
        // does not distort the first timed rounds.
        let _ = match level.as_str() {
            "L0" => run_l0(&backends, args.warmup).await,
            "L1" => run_l1(&backends, args.warmup).await,
            _ => run_l2(&backends, args.warmup).await,
        };

        let report = level_report(name, description, &args, &backends, runs);
        print_report(&report);

        let path = output_path(&args.out, name);
        let json = serde_json::to_string_pretty(&report).expect("serialize report");
        std::fs::write(&path, json).unwrap_or_else(|err| panic!("write {path:?}: {err}"));
        println!("raw output written to {}", path.display());
    }
}

// ---------------------------------------------------------------------------
// Phase 3 retrieval evaluation (`vox-bench eval`)
// ---------------------------------------------------------------------------

/// Everything written to `retrieval_eval.json`.
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

async fn eval_main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(
        std::env::args()
            .nth(2)
            .unwrap_or_else(|| "benchmarks".to_owned()),
    );
    std::fs::create_dir_all(&out_dir)?;

    let queries = vox_bench::eval_queries();
    println!(
        "evaluation set: {} queries (en/hi/ta) against the bundled sample corpus",
        queries.len()
    );

    // Baseline: default production configuration, full pipeline.
    let baseline = vox_bench::run_baseline(&queries).await?;
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
    for mode in vox_bench::runner::RetrievalMode::all() {
        let evaluation = vox_bench::evaluate_mode(&engine, mode, &queries).await?;
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
            vox_oreo::chunk::ChunkingStrategy::Fixed {
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
            vox_oreo::chunk::ChunkingStrategy::Sliding {
                window: 600,
                stride: 300,
            },
        ),
    ];
    let mut chunkings = Vec::new();
    for (label, strategy) in strategies {
        let evaluation = vox_bench::evaluate_chunking(strategy, label, &queries).await?;
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

    let environment = vox_bench::environment::detect_environment();

    write_json(
        &out_dir.join("retrieval_eval.json"),
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

    let markdown = vox_bench::report::render_report(&environment, &baseline, &ablation, &chunkings);
    std::fs::write(out_dir.join("retrieval_report.md"), markdown)?;

    println!("reports written to {}", out_dir.display());
    Ok(())
}

async fn multilingual_main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out_dir = PathBuf::from("benchmarks");
    let mut rounds = vox_bench::DEFAULT_BENCH_ROUNDS;

    let mut it = std::env::args().skip(2);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--rounds" => {
                if let Some(r) = it.next() {
                    rounds = r.parse().unwrap_or(vox_bench::DEFAULT_BENCH_ROUNDS);
                }
            }
            dir if !dir.starts_with('-') => {
                out_dir = PathBuf::from(dir);
            }
            _ => {}
        }
    }

    std::fs::create_dir_all(&out_dir)?;

    println!("Running Phase 8: Multilingual Validation across primary (en, hi, ta) and secondary (te, kn) languages...");
    let report = vox_bench::run_multilingual_validation(rounds).await?;

    vox_bench::print_multilingual_summary(&report);

    let path = out_dir.join("multilingual.json");
    write_json(&path, &report)?;
    println!("multilingual benchmark JSON written to {}", path.display());

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

#[tokio::main]
async fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("eval") => {
            eval_main().await.unwrap_or_else(|err| {
                eprintln!("eval failed: {err}");
                std::process::exit(1);
            });
        }
        Some("multilingual") => {
            multilingual_main().await.unwrap_or_else(|err| {
                eprintln!("multilingual evaluation failed: {err}");
                std::process::exit(1);
            });
        }
        _ => {
            e2e_main().await;
        }
    }
}
