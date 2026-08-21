//! Safety gates applied to evidence verdicts and generated answers.
//!
//! Phase 0 scope: verdict mapping and empty-generation refusal. Off-topic and
//! unsafe-content detection require a classifier and land in the guardrails
//! phase.

use vox_types::{Answerability, GuardrailDecision};

/// Maps an evidence-sufficiency verdict onto a serve/refuse decision.
#[must_use]
pub fn decide(answerability: Answerability) -> GuardrailDecision {
    match answerability {
        Answerability::Answerable => GuardrailDecision::Allow,
        Answerability::InsufficientEvidence => GuardrailDecision::Refuse {
            reason: "insufficient_evidence".to_owned(),
        },
        Answerability::OffTopic => GuardrailDecision::Refuse {
            reason: "off_topic".to_owned(),
        },
    }
}

/// Final pass over the generated answer before it is served.
///
/// Phase 0 scope: refuse empty generations.
#[must_use]
pub fn evaluate_answer(answer: &str) -> GuardrailDecision {
    if answer.trim().is_empty() {
        GuardrailDecision::Refuse {
            reason: "empty_generation".to_owned(),
        }
    } else {
        GuardrailDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decide_should_allow_answerable_evidence() {
        assert_eq!(decide(Answerability::Answerable), GuardrailDecision::Allow);
    }

    #[test]
    fn decide_should_refuse_insufficient_evidence() {
        assert_eq!(
            decide(Answerability::InsufficientEvidence),
            GuardrailDecision::Refuse {
                reason: "insufficient_evidence".to_owned()
            }
        );
    }

    #[test]
    fn decide_should_refuse_off_topic_queries() {
        assert_eq!(
            decide(Answerability::OffTopic),
            GuardrailDecision::Refuse {
                reason: "off_topic".to_owned()
            }
        );
    }

    #[test]
    fn answer_guard_should_refuse_blank_answers() {
        assert_eq!(
            evaluate_answer("   "),
            GuardrailDecision::Refuse {
                reason: "empty_generation".to_owned()
            }
        );
    }

    #[test]
    fn answer_guard_should_allow_non_empty_answers() {
        assert_eq!(evaluate_answer("GST is a tax."), GuardrailDecision::Allow);
    }
}
