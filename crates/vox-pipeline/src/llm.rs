//! Answer generation backends.

use async_trait::async_trait;
use vox_core::{Language, RetrievedChunk};

use crate::error::PipelineError;

/// Output of the generation stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmOutput {
    /// Raw generated answer text (may be empty; the guardrail decides).
    pub answer: String,
}

/// Substitution boundary for answer generation.
///
/// One implementation per provider behind this trait; the pipeline never
/// talks to an LLM SDK directly.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Backend name used in logs and metrics.
    fn name(&self) -> &'static str;

    /// Generates an answer for `query` grounded in `evidence`.
    ///
    /// Implementations receive already-validated evidence and must not
    /// invent facts beyond it; grounding is enforced by the caller's
    /// guardrail pass.
    ///
    /// # Errors
    /// Returns [`PipelineError::Llm`] when the backend fails to produce any
    /// output. An empty output is reported as success and handled by the
    /// guardrail as a refusal.
    async fn generate(
        &self,
        query: &str,
        language: Language,
        evidence: &[RetrievedChunk],
    ) -> Result<LlmOutput, PipelineError>;
}

/// Deterministic extractive baseline used until a real provider is wired in.
///
/// Returns the highest-scoring evidence snippet (first two sentences). This
/// is intentionally naive — it exists so latency measurement and refusal
/// behavior can be validated before provider integration.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtractiveLlmClient;

const MAX_SNIPPET_CHARS: usize = 280;
const SENTENCE_TERMINATORS: [char; 4] = ['.', '!', '?', '।'];

#[async_trait]
impl LlmClient for ExtractiveLlmClient {
    fn name(&self) -> &'static str {
        "extractive-stub"
    }

    async fn generate(
        &self,
        _query: &str,
        _language: Language,
        evidence: &[RetrievedChunk],
    ) -> Result<LlmOutput, PipelineError> {
        let best = evidence
            .iter()
            .max_by(|a, b| a.score.total_cmp(&b.score));
        let answer = best.map_or_else(String::new, |chunk| first_sentences(&chunk.text, 2));
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

    #[test]
    fn first_sentences_should_keep_two_terminated_sentences() {
        let text = "GST is a tax. It has slabs! More detail here";
        assert_eq!(first_sentences(text, 2), "GST is a tax. It has slabs!");
    }

    #[test]
    fn first_sentences_should_fall_back_to_full_text_without_terminators() {
        assert_eq!(first_sentences("no terminators at all", 2), "no terminators at all");
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
