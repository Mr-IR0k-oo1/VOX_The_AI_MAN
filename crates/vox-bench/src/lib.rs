//! Benchmarking utilities for VOX.
//!
//! Phase 3 adds the retrieval evaluation runner: a hand-judged multilingual
//! query set, quality metrics (Recall@5, MRR), latency percentiles
//! (P50/P70/P100) with per-stage attribution, and machine- plus
//! human-readable reports. Percentile math lives here so API and pipeline
//! reporting can share one implementation.

pub mod environment;
pub mod eval;
pub mod metrics;
pub mod report;
pub mod runner;

pub use environment::EnvironmentInfo;
pub use eval::{eval_queries, EvalQuery};
pub use metrics::{mrr, recall_at_k};
pub use runner::{
    bench_config, evaluate_chunking, evaluate_mode, run_baseline, BaselineRun, ChunkingEvaluation,
    DatasetInfo, EngineConfigSummary, ModeEvaluation, RetrievalMode, StageStats, StageTimings,
    DEFAULT_SENTENCE_CHUNKING, TIMING_SWEEPS,
};

/// Computes the `quantile` (in `0.0..=1.0`) of an ascending-sorted sample.
///
/// Uses linear interpolation between closest ranks. Returns `None` for an
/// empty sample; quantiles outside `0.0..=1.0` are clamped into range.
#[must_use]
pub fn percentile_sorted(sorted: &[f64], quantile: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let q = quantile.clamp(0.0, 1.0);
    let n = sorted.len();
    if n == 1 {
        return Some(sorted[0]);
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    Some(sorted[lo] + (sorted[hi] - sorted[lo]) * frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_should_return_none_for_empty_samples() {
        assert_eq!(percentile_sorted(&[], 0.5), None);
    }

    #[test]
    fn percentile_should_return_the_single_sample() {
        assert_eq!(percentile_sorted(&[42.0], 0.99), Some(42.0));
    }

    #[test]
    fn percentile_should_interpolate_between_ranks() {
        let sample = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(percentile_sorted(&sample, 0.0), Some(10.0));
        assert_eq!(percentile_sorted(&sample, 0.5), Some(25.0));
        assert_eq!(percentile_sorted(&sample, 1.0), Some(40.0));
    }

    #[test]
    fn percentile_should_clamp_out_of_range_quantiles() {
        let sample = [1.0, 2.0];
        assert_eq!(percentile_sorted(&sample, -1.0), Some(1.0));
        assert_eq!(percentile_sorted(&sample, 2.0), Some(2.0));
    }

    #[test]
    fn percentile_p50_and_p95_should_match_expected_values() {
        let sample: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile_sorted(&sample, 0.50), Some(50.5));
        assert_eq!(percentile_sorted(&sample, 0.95), Some(95.05));
    }
}
