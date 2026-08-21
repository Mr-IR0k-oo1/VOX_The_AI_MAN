//! Answer-generation substitution boundary for VOX.
//!
//! The pipeline depends only on [`LlmProvider`]; one implementation lives
//! behind this trait per provider, and the pipeline never talks to an LLM SDK
//! directly. Phase 0 ships a deterministic extractive stub so latency and
//! refusal behavior can be validated before provider integration.

use async_trait::async_trait;
use thiserror::Error;
use vox_types::{Language, RetrievedDocument};

/// Failures raised by an answer-generation backend.
#[derive(Debug, Error)]
pub enum LlmError {
    /// The backend failed to produce any output.
    #[error("llm backend failed: {0}")]
    Backend(String),
}

/// Output of the generation stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmOutput {
    /// Raw generated answer text (may be empty; the guardrail decides).
    pub answer: String,
}

/// Substitution boundary for answer generation.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Backend name used in logs and metrics.
    fn name(&self) -> &'static str;

    /// Generates an answer for `query` grounded in `evidence`.
    ///
    /// Implementations receive already-validated evidence and must not invent
    /// facts beyond it; grounding is enforced by the caller's guardrail pass.
    ///
    /// # Errors
    /// Returns [`LlmError`] when the backend fails to produce any output. An
    /// empty output is reported as success and handled by the guardrail as a
    /// refusal.
    async fn generate(
        &self,
        query: &str,
        language: Language,
        evidence: &[RetrievedDocument],
    ) -> Result<LlmOutput, LlmError>;
}

/// Deterministic extractive baseline used until a real provider is wired in.
///
/// Returns the highest-scoring evidence snippet (first two sentences). This
/// is intentionally naive — it exists so latency measurement and refusal
/// behavior can be validated before provider integration.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtractiveProvider;

const MAX_SNIPPET_CHARS: usize = 280;
const SENTENCE_TERMINATORS: [char; 4] = ['.', '!', '?', '।'];

#[async_trait]
impl LlmProvider for ExtractiveProvider {
    fn name(&self) -> &'static str {
        "extractive-stub"
    }

    async fn generate(
        &self,
        _query: &str,
        _language: Language,
        evidence: &[RetrievedDocument],
    ) -> Result<LlmOutput, LlmError> {
        let best = evidence.iter().max_by(|a, b| a.score.total_cmp(&b.score));
        let answer = best.map_or_else(String::new, |doc| first_sentences(&doc.text, 2));
        Ok(LlmOutput { answer })
    }
}

/// Extracts up to `count` sentences, capped at [`MAX_SNIPPET_CHARS`] chars on
/// a char boundary.
fn first_sentences(text: &str, count: usize) -> String {
    let mut end = 0usize;
    let mut sentences = 0usize;
    for (idx, ch) in text.char_indices() {
        if SENTENCE_TERMINATORS.contains(&ch) {
            end = idx + ch.len_utf8();
            sentences += 1;
            if sentences == count {
                break;
            }
        }
    }

    if end == 0 || sentences < count {
        end = text.len();
    }
    let mut snippet = &text[..end];
    if snippet.len() > MAX_SNIPPET_CHARS {
        let mut cut = MAX_SNIPPET_CHARS;
        while !snippet.is_char_boundary(cut) {
            cut -= 1;
        }
        snippet = &snippet[..cut];
    }
    snippet.trim_end().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn doc(score: f32, text: &str) -> RetrievedDocument {
        RetrievedDocument {
            id: "d1".to_owned(),
            text: text.to_owned(),
            score,
            rank: 1,
            metadata: Value::Null,
        }
    }

    #[tokio::test]
    async fn extractive_should_answer_from_the_best_scoring_document() {
        let evidence = [doc(0.4, "weaker"), doc(0.9, "GST is a tax. It has slabs!")];
        let out = ExtractiveProvider
            .generate("what is gst", Language::En, &evidence)
            .await
            .expect("generate");
        assert_eq!(out.answer, "GST is a tax. It has slabs!");
    }

    #[tokio::test]
    async fn extractive_should_return_empty_output_without_evidence() {
        let out = ExtractiveProvider
            .generate("q", Language::En, &[])
            .await
            .expect("generate");
        assert_eq!(out.answer, "");
    }

    #[test]
    fn first_sentences_should_keep_two_terminated_sentences() {
        let text = "GST is a tax. It has slabs! More detail here";
        assert_eq!(first_sentences(text, 2), "GST is a tax. It has slabs!");
    }

    #[test]
    fn first_sentences_should_fall_back_to_full_text_without_terminators() {
        assert_eq!(
            first_sentences("no terminators at all", 2),
            "no terminators at all"
        );
    }

    #[test]
    fn first_sentences_should_cut_on_char_boundaries() {
        // 'अ' is 3 bytes; a naive byte-cut would panic or split the glyph.
        let text = "अ".repeat(200);
        let out = first_sentences(&text, 1);
        assert!(out.len() <= MAX_SNIPPET_CHARS);
        assert!(out.chars().all(|c| c == 'अ'));
    }
}
