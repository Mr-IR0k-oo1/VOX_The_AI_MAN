//! Grounding/safety gate applied to evidence and generated answers.

use vox_core::RetrievedChunk;

/// Outcome of a guardrail check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailVerdict {
    /// Proceed: the content passes the check.
    Allow,
    /// Refuse: answer must not be served; carries a machine-readable reason.
    Refuse {
        /// Stable reason code surfaced to API consumers.
        reason: &'static str,
    },
}

/// Decides whether retrieved evidence is strong enough to ground an answer.
///
/// The request is refused when there is no evidence or when the best score is
/// below `min_score`. A non-finite best score also refuses: it means the
/// upstream violated the contract and no comparison against the threshold is
/// meaningful.
#[must_use]
pub fn evaluate_grounding(evidence: &[RetrievedChunk], min_score: f32) -> GuardrailVerdict {
    let best = evidence
        .iter()
        .map(|chunk| chunk.score)
        .fold(f32::NEG_INFINITY, f32::max);

    if !best.is_finite() || best < min_score {
        GuardrailVerdict::Refuse {
            reason: "insufficient_evidence",
        }
    } else {
        GuardrailVerdict::Allow
    }
}

/// Final pass over the generated answer before it is served.
///
/// Phase 1 scope: refuse empty generations. Off-topic and unsafe-content
/// detection require a classifier and land in the guardrails phase.
#[must_use]
pub fn evaluate_answer(answer: &str) -> GuardrailVerdict {
    if answer.trim().is_empty() {
        GuardrailVerdict::Refuse {
            reason: "empty_generation",
        }
    } else {
        GuardrailVerdict::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn chunk(score: f32) -> RetrievedChunk {
        RetrievedChunk {
            id: "c1".to_owned(),
            text: "text".to_owned(),
            score,
            rank: 1,
            metadata: Value::Null,
        }
    }

    #[test]
    fn grounding_should_allow_when_best_score_meets_threshold() {
        let verdict = evaluate_grounding(&[chunk(0.4), chunk(0.9)], 0.30);
        assert_eq!(verdict, GuardrailVerdict::Allow);
    }

    #[test]
    fn grounding_should_refuse_when_all_scores_are_below_threshold() {
        let verdict = evaluate_grounding(&[chunk(0.1), chunk(0.2)], 0.30);
        assert_eq!(
            verdict,
            GuardrailVerdict::Refuse {
                reason: "insufficient_evidence"
            }
        );
    }

    #[test]
    fn grounding_should_refuse_on_empty_evidence() {
        assert_eq!(
            evaluate_grounding(&[], 0.30),
            GuardrailVerdict::Refuse {
                reason: "insufficient_evidence"
            }
        );
    }

    #[test]
    fn grounding_should_refuse_non_finite_scores() {
        assert_eq!(
            evaluate_grounding(&[chunk(f32::NAN)], 0.0),
            GuardrailVerdict::Refuse {
                reason: "insufficient_evidence"
            }
        );
    }

    #[test]
    fn answer_guard_should_refuse_blank_answers() {
        assert_eq!(
            evaluate_answer("   "),
            GuardrailVerdict::Refuse {
                reason: "empty_generation"
            }
        );
    }

    #[test]
    fn answer_guard_should_allow_non_empty_answers() {
        assert_eq!(evaluate_answer("GST is a tax."), GuardrailVerdict::Allow);
    }
}
