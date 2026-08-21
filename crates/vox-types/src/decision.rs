//! Verdict types shared by the grounding and guardrail stages.

use serde::{Deserialize, Serialize};

/// Whether retrieved evidence is sufficient to ground an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Answerability {
    /// Evidence meets the configured sufficiency threshold.
    Answerable,
    /// No evidence was returned, or the best score is below threshold.
    InsufficientEvidence,
    /// The query falls outside the served domain (reserved; detection lands
    /// with the guardrails phase).
    OffTopic,
}

/// Outcome of a guardrail check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailDecision {
    /// Proceed: the content passes the check.
    Allow,
    /// Refuse: the answer must not be served; carries a stable reason code
    /// surfaced to API consumers.
    Refuse {
        /// Machine-readable refusal reason (e.g. `insufficient_evidence`).
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answerability_should_serialize_snake_case() {
        assert_eq!(
            serde_json::to_value(Answerability::InsufficientEvidence).expect("serialize"),
            serde_json::json!("insufficient_evidence")
        );
    }

    #[test]
    fn guardrail_decision_should_roundtrip() {
        let decision = GuardrailDecision::Refuse {
            reason: "off_topic".to_owned(),
        };
        let json = serde_json::to_string(&decision).expect("serialize");
        let parsed: GuardrailDecision = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, decision);
    }
}
