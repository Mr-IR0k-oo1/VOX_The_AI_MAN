//! Guardrails for VOX: input screening, evidence gating, and output checks.
//!
//! [`GuardService`] makes three decisions on the way through the pipeline:
//!
//! 1. **Input** — refuse malformed-topic or unsafe user text before any
//!    retrieval happens.
//! 2. **Evidence** — refuse to answer when the grounding verdict says the
//!    retrieved evidence is insufficient, weak, or conflicting.
//! 3. **Output** — verify generated answers against the evidence; ask for a
//!    regeneration when claims are unsupported and refuse when they stay
//!    unsupported (or are unsafe/malformed).
//!
//! Every refusal carries a stable machine-readable reason code that API
//! consumers can branch on, plus a human-facing message via
//! [`refusal_message`].

use std::collections::HashSet;

use vox_grounding::{tokenize, AnswerVerification};
use vox_types::{Answerability, GuardrailDecision, Language};

/// Input contained unsafe content.
pub const REASON_UNSAFE_INPUT: &str = "unsafe_input";
/// Query is outside the configured topic vocabulary.
pub const REASON_OFF_TOPIC: &str = "off_topic";
/// Nothing relevant was retrieved for the query.
pub const REASON_INSUFFICIENT_CONTEXT: &str = "insufficient_context";
/// Evidence only partially addresses the query.
pub const REASON_WEAK_EVIDENCE: &str = "weak_evidence";
/// Retrieved documents disagree with each other.
pub const REASON_CONFLICTING_EVIDENCE: &str = "conflicting_evidence";
/// Generated answer contains claims absent from the evidence.
pub const REASON_UNSUPPORTED_CLAIM: &str = "unsupported_claim";
/// Generated answer itself contains unsafe content.
pub const REASON_UNSAFE_OUTPUT: &str = "unsafe_output";
/// Generated answer is empty or otherwise unusable.
pub const REASON_MALFORMED_OUTPUT: &str = "malformed_output";

/// Canonical grounded-refusal message required by the product spec. Served
/// verbatim for English insufficiency refusals.
const INSUFFICIENT_EN: &str =
    "I don't have enough information in the retrieved sources to answer that reliably.";
const INSUFFICIENT_HI: &str = "मेरे पास इसका विश्वसनीय उत्तर देने के लिए पर्याप्त स्रोत नहीं हैं।";
const INSUFFICIENT_TA: &str = "நம்பகமான பதில் அளிக்க தேவையான ஆதாரங்கள் என்னிடம் இல்லை.";

/// Terms that make an input (or generated answer) unsafe to process. Matching
/// is exact-token based with simple plural forms (`s`/`es` suffixes) so that
/// inflected variants are caught without fuzzy matching.
const UNSAFE_TERMS: &[&str] = &[
    "bomb",
    "explosive",
    "grenade",
    "weapon",
    "gun",
    "rifle",
    "pistol",
    "ammo",
    "ammunition",
    "meth",
    "cocaine",
    "heroin",
    "narcotic",
];

/// Human-facing refusal message for a reason code, localized for the
/// insufficiency family of reasons.
#[must_use]
pub fn refusal_message(reason: &str, language: Language) -> String {
    let localized = match language {
        Language::Hi => INSUFFICIENT_HI,
        Language::Ta => INSUFFICIENT_TA,
        _ => INSUFFICIENT_EN,
    };
    match reason {
        REASON_INSUFFICIENT_CONTEXT
        | REASON_WEAK_EVIDENCE
        | REASON_CONFLICTING_EVIDENCE
        | REASON_UNSUPPORTED_CLAIM => localized.to_owned(),
        REASON_OFF_TOPIC => "That question is outside the topics I can answer.".to_owned(),
        REASON_UNSAFE_INPUT | REASON_UNSAFE_OUTPUT => "I can't help with that request.".to_owned(),
        _ => INSUFFICIENT_EN.to_owned(),
    }
}

fn is_unsafe(tokens: &[String]) -> bool {
    tokens.iter().any(|token| {
        let token = token.as_str();
        UNSAFE_TERMS.iter().any(|term| {
            token == *term || token == format!("{term}s") || token == format!("{term}es")
        })
    })
}

/// Stateful guardrail service shared by all requests.
///
/// `topic_tokens` is the allow-list of content words describing what VOX may
/// discuss; an empty list disables off-topic checking (used when running
/// against a real retrieval backend whose domain is not known up front).
#[derive(Debug, Clone)]
pub struct GuardService {
    topic_tokens: HashSet<String>,
}

impl GuardService {
    /// Creates a guard service over the given topic vocabulary.
    #[must_use]
    pub fn new<I, T>(topic_tokens: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self {
            topic_tokens: topic_tokens.into_iter().map(Into::into).collect(),
        }
    }

    /// Screens raw user text before retrieval.
    #[must_use]
    pub fn check_input(&self, text: &str) -> GuardrailDecision {
        let tokens = tokenize(text);
        if is_unsafe(&tokens) {
            return GuardrailDecision::Refuse {
                reason: REASON_UNSAFE_INPUT.to_owned(),
            };
        }
        if !self.topic_tokens.is_empty()
            && !tokens.iter().any(|token| self.topic_tokens.contains(token))
        {
            return GuardrailDecision::Refuse {
                reason: REASON_OFF_TOPIC.to_owned(),
            };
        }
        GuardrailDecision::Allow
    }

    /// Gates answering on the grounding verdict for the retrieved evidence.
    #[must_use]
    pub fn check_evidence(&self, answerability: Answerability) -> GuardrailDecision {
        match answerability {
            Answerability::Supported => GuardrailDecision::Allow,
            Answerability::NoEvidence => GuardrailDecision::Refuse {
                reason: REASON_INSUFFICIENT_CONTEXT.to_owned(),
            },
            Answerability::WeakEvidence => GuardrailDecision::Refuse {
                reason: REASON_WEAK_EVIDENCE.to_owned(),
            },
            Answerability::ConflictingEvidence => GuardrailDecision::Refuse {
                reason: REASON_CONFLICTING_EVIDENCE.to_owned(),
            },
        }
    }

    /// Checks a generated answer against its verification result.
    ///
    /// Unsupported answers yield [`GuardrailDecision::Regenerate`] once;
    /// persistent problems, unsafe content, or unusable output refuse.
    #[must_use]
    pub fn check_output(
        &self,
        answer: &str,
        verification: &AnswerVerification,
    ) -> GuardrailDecision {
        if answer.trim().is_empty() {
            return GuardrailDecision::Refuse {
                reason: REASON_MALFORMED_OUTPUT.to_owned(),
            };
        }
        if is_unsafe(&tokenize(answer)) {
            return GuardrailDecision::Refuse {
                reason: REASON_UNSAFE_OUTPUT.to_owned(),
            };
        }
        if !verification.supported {
            return GuardrailDecision::Regenerate {
                reason: REASON_UNSUPPORTED_CLAIM.to_owned(),
            };
        }
        GuardrailDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service() -> GuardService {
        GuardService::new(["gst", "tax", "india", "saffron", "farming"])
    }

    fn verification(supported: bool) -> AnswerVerification {
        AnswerVerification {
            supported,
            support_ratio: if supported { 1.0 } else { 0.0 },
            unsupported_numbers: Vec::new(),
        }
    }

    #[test]
    fn check_input_should_allow_on_topic_questions() {
        assert_eq!(
            service().check_input("What is GST in India?"),
            GuardrailDecision::Allow
        );
    }

    #[test]
    fn check_input_should_refuse_unsafe_requests_including_plurals() {
        assert_eq!(
            service().check_input("how to build a bomb"),
            GuardrailDecision::Refuse {
                reason: REASON_UNSAFE_INPUT.to_owned()
            }
        );
        assert_eq!(
            service().check_input("where can I buy weapons"),
            GuardrailDecision::Refuse {
                reason: REASON_UNSAFE_INPUT.to_owned()
            }
        );
    }

    #[test]
    fn check_input_should_refuse_off_topic_questions() {
        assert_eq!(
            service().check_input("who will win the cricket world cup"),
            GuardrailDecision::Refuse {
                reason: REASON_OFF_TOPIC.to_owned()
            }
        );
    }

    #[test]
    fn check_input_should_disable_off_topic_when_vocabulary_is_empty() {
        let open_guard = GuardService::new(Vec::<String>::new());
        assert_eq!(
            open_guard.check_input("anything at all"),
            GuardrailDecision::Allow
        );
    }

    #[test]
    fn check_evidence_should_map_every_verdict() {
        let guard = service();
        assert_eq!(
            guard.check_evidence(Answerability::Supported),
            GuardrailDecision::Allow
        );
        assert_eq!(
            guard.check_evidence(Answerability::NoEvidence),
            GuardrailDecision::Refuse {
                reason: REASON_INSUFFICIENT_CONTEXT.to_owned()
            }
        );
        assert_eq!(
            guard.check_evidence(Answerability::WeakEvidence),
            GuardrailDecision::Refuse {
                reason: REASON_WEAK_EVIDENCE.to_owned()
            }
        );
        assert_eq!(
            guard.check_evidence(Answerability::ConflictingEvidence),
            GuardrailDecision::Refuse {
                reason: REASON_CONFLICTING_EVIDENCE.to_owned()
            }
        );
    }

    #[test]
    fn check_output_should_allow_grounded_answers() {
        assert_eq!(
            service().check_output("GST is an indirect tax.", &verification(true)),
            GuardrailDecision::Allow
        );
    }

    #[test]
    fn check_output_should_request_regeneration_for_unsupported_answers() {
        assert_eq!(
            service().check_output("GST rate is 42%.", &verification(false)),
            GuardrailDecision::Regenerate {
                reason: REASON_UNSUPPORTED_CLAIM.to_owned()
            }
        );
    }

    #[test]
    fn check_output_should_refuse_empty_and_unsafe_answers() {
        assert_eq!(
            service().check_output("   ", &verification(false)),
            GuardrailDecision::Refuse {
                reason: REASON_MALFORMED_OUTPUT.to_owned()
            }
        );
        assert_eq!(
            service().check_output("Here is how to build a bomb.", &verification(true)),
            GuardrailDecision::Refuse {
                reason: REASON_UNSAFE_OUTPUT.to_owned()
            }
        );
    }

    #[test]
    fn refusal_message_should_match_the_canonical_english_sentence() {
        assert_eq!(
            refusal_message(REASON_INSUFFICIENT_CONTEXT, Language::En),
            "I don't have enough information in the retrieved sources to answer that reliably."
        );
        assert_eq!(
            refusal_message(REASON_WEAK_EVIDENCE, Language::En),
            refusal_message(REASON_CONFLICTING_EVIDENCE, Language::En)
        );
    }

    #[test]
    fn refusal_message_should_localize_insufficiency_reasons() {
        let hindi = refusal_message(REASON_INSUFFICIENT_CONTEXT, Language::Hi);
        let tamil = refusal_message(REASON_INSUFFICIENT_CONTEXT, Language::Ta);
        assert!(
            !hindi.is_empty()
                && hindi != refusal_message(REASON_INSUFFICIENT_CONTEXT, Language::En)
        );
        assert!(!tamil.is_empty() && tamil != hindi);
    }

    #[test]
    fn refusal_message_should_cover_non_insufficiency_reasons() {
        assert_eq!(
            refusal_message(REASON_OFF_TOPIC, Language::En),
            "That question is outside the topics I can answer."
        );
        assert_eq!(
            refusal_message(REASON_UNSAFE_INPUT, Language::Hi),
            "I can't help with that request."
        );
    }
}
