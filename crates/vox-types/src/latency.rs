//! Per-stage latency metrics for the pipeline.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Converts a [`Duration`] to milliseconds as `f64`.
///
/// Sub-millisecond precision matters: several stages are expected to run in
/// tens of microseconds once retrieval is local.
#[must_use]
pub fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

/// Latency of each pipeline stage in milliseconds.
///
/// Stages that did not run (e.g. `stt` on a text query, or `llm` after a
/// refusal) are omitted from serialization rather than reported as zero, so
/// consumers never mistake a skipped stage for a fast one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Speech-to-text duration (voice flow only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stt: Option<f64>,
    /// Language identification/validation duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<f64>,
    /// Query analysis (normalization) duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_analysis: Option<f64>,
    /// Total time inside the retrieval boundary (dense + BM25 + RRF + rerank).
    ///
    /// Stage-level splits arrive via the retrieval service's own metrics; this
    /// measures the full round trip across the boundary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval: Option<f64>,
    /// Grounding/evidence-sufficiency check duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<f64>,
    /// Answer generation duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<f64>,
    /// Final guardrail pass duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guardrail: Option<f64>,
    /// End-to-end pipeline duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ms_should_convert_to_milliseconds() {
        assert!((ms(Duration::from_millis(250)) - 250.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unset_stages_should_be_omitted_from_json() {
        let timings = LatencyMetrics {
            total: Some(1.5),
            ..LatencyMetrics::default()
        };
        let json = serde_json::to_value(&timings).expect("serialize timings");
        assert_eq!(json, serde_json::json!({ "total": 1.5 }));
    }

    #[test]
    fn empty_json_should_deserialize_to_all_unset() {
        let timings: LatencyMetrics = serde_json::from_str("{}").expect("deserialize");
        assert_eq!(timings, LatencyMetrics::default());
    }
}
