//! Evidence-sufficiency assessment for retrieved documents.
//!
//! Phase 0 scope: a score-threshold check over the evidence set. Richer
//! signals (entailment, citation coverage) land with the grounding phase.

use vox_types::{Answerability, RetrievedDocument};

/// Decides whether retrieved evidence is strong enough to ground an answer.
///
/// The evidence is deemed insufficient when there is none or when the best
/// score is below `min_score`. A non-finite best score is also treated as
/// insufficient: it means the upstream violated the contract and no
/// comparison against the threshold is meaningful.
#[must_use]
pub fn assess(evidence: &[RetrievedDocument], min_score: f32) -> Answerability {
    let best = evidence
        .iter()
        .map(|doc| doc.score)
        .fold(f32::NEG_INFINITY, f32::max);

    if !best.is_finite() || best < min_score {
        Answerability::InsufficientEvidence
    } else {
        Answerability::Answerable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn doc(score: f32) -> RetrievedDocument {
        RetrievedDocument {
            id: "d1".to_owned(),
            text: "text".to_owned(),
            score,
            rank: 1,
            metadata: Value::Null,
        }
    }

    #[test]
    fn should_answer_when_best_score_meets_threshold() {
        assert_eq!(
            assess(&[doc(0.4), doc(0.9)], 0.30),
            Answerability::Answerable
        );
    }

    #[test]
    fn should_refuse_when_all_scores_are_below_threshold() {
        assert_eq!(
            assess(&[doc(0.1), doc(0.2)], 0.30),
            Answerability::InsufficientEvidence
        );
    }

    #[test]
    fn should_refuse_empty_evidence() {
        assert_eq!(assess(&[], 0.30), Answerability::InsufficientEvidence);
    }

    #[test]
    fn should_refuse_non_finite_scores() {
        assert_eq!(
            assess(&[doc(f32::NAN)], 0.0),
            Answerability::InsufficientEvidence
        );
    }
}
