//! Verdict types shared by the grounding and guardrail stages.

use serde::{Deserialize, Serialize};

/// Evidence-sufficiency verdict produced by the grounding stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Answerability {
    /// Evidence is relevant, sufficiently covering, and internally consistent.
    Supported,
    /// Evidence only partially addresses the query or scores too low to trust.
    WeakEvidence,
    /// Nothing relevant was retrieved for the query.
    NoEvidence,
    /// Retrieved documents that address the query disagree with each other.
    ConflictingEvidence,
}

/// Outcome of a guardrail check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailDecision {
    /// Proceed: the content passes the check.
    Allow,
    /// Retry generation (e.g. the first answer contained unsupported claims).
    Regenerate {
        /// Machine-readable reason code surfaced to API consumers.
        reason: String,
    },
    /// Refuse: the answer must not be served; carries a stable reason code
    /// surfaced to API consumers.
    Refuse {
        /// Machine-readable refusal reason (e.g. `insufficient_context`).
        reason: String,
    },
}

impl GuardrailDecision {
    /// The machine-readable reason for non-allow decisions.
    #[must_use]
    pub fn reason(&self) -> Option<&str> {
        match self {
            GuardrailDecision::Allow => None,
            GuardrailDecision::Regenerate { reason } | GuardrailDecision::Refuse { reason } => {
                Some(reason.as_str())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answerability_should_serialize_snake_case() {
        assert_eq!(
            serde_json::to_value(Answerability::WeakEvidence).expect("serialize"),
            serde_json::json!("weak_evidence")
        );
        assert_eq!(
            serde_json::to_value(Answerability::NoEvidence).expect("serialize"),
            serde_json::json!("no_evidence")
        );
        assert_eq!(
            serde_json::to_value(Answerability::ConflictingEvidence).expect("serialize"),
            serde_json::json!("conflicting_evidence")
        );
        assert_eq!(
            serde_json::to_value(Answerability::Supported).expect("serialize"),
            serde_json::json!("supported")
        );
    }

    #[test]
    fn guardrail_decision_should_roundtrip_all_variants() {
        for decision in [
            GuardrailDecision::Allow,
            GuardrailDecision::Regenerate {
                reason: "unsupported_claim".to_owned(),
            },
            GuardrailDecision::Refuse {
                reason: "off_topic".to_owned(),
            },
        ] {
            let json = serde_json::to_string(&decision).expect("serialize");
            let parsed: GuardrailDecision = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, decision);
        }
    }

    #[test]
    fn reason_should_be_present_only_for_non_allow_decisions() {
        assert_eq!(GuardrailDecision::Allow.reason(), None);
        assert_eq!(
            GuardrailDecision::Refuse {
                reason: "unsafe_input".to_owned()
            }
            .reason(),
            Some("unsafe_input")
        );
    }
}
