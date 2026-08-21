//! Deterministic extractive baseline provider.
//!
//! Answers with the highest-scoring evidence snippet (first two sentences).
//! Intentionally naive: it exists so the full pipeline, latency measurement,
//! and refusal behavior can be validated without any external dependency,
//! and it doubles as the default `mock` LLM backend.

use async_trait::async_trait;

use crate::{LlmError, LlmProvider, LlmRequest, LlmResponse};

const MAX_SNIPPET_CHARS: usize = 280;
const SENTENCE_TERMINATORS: [char; 4] = ['.', '!', '?', '।'];

/// Deterministic extractive provider used as the mock/default backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtractiveProvider;

#[async_trait]
impl LlmProvider for ExtractiveProvider {
    fn name(&self) -> &'static str {
        "extractive"
    }

    async fn generate(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let best = request
            .evidence
            .iter()
            .max_by(|a, b| a.score.total_cmp(&b.score));
        let answer = best.map_or_else(String::new, |doc| first_sentences(&doc.text, 2));
        Ok(LlmResponse {
            answer,
            model: None,
        })
    }
}

/// Extracts up to `count` sentences, capped at [`MAX_SNIPPET_CHARS`] chars on
/// a char boundary.
#[must_use]
pub fn first_sentences(text: &str, count: usize) -> String {
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
    use serde_json::Value;
    use vox_types::{Language, RetrievedDocument};

    use super::*;

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
        let request = LlmRequest::new(
            "what is gst",
            Language::En,
            vec![doc(0.4, "weaker"), doc(0.9, "GST is a tax. It has slabs!")],
        );
        let out = ExtractiveProvider
            .generate(&request)
            .await
            .expect("generate");
        assert_eq!(out.answer, "GST is a tax. It has slabs!");
        assert_eq!(out.model, None);
    }

    #[tokio::test]
    async fn extractive_should_return_empty_output_without_evidence() {
        let request = LlmRequest::new("q", Language::En, vec![]);
        let out = ExtractiveProvider
            .generate(&request)
            .await
            .expect("generate");
        assert_eq!(out.answer, "");
    }

    #[test]
    fn first_sentences_should_keep_two_terminated_sentences() {
        assert_eq!(
            first_sentences("GST is a tax. It has slabs! More detail here", 2),
            "GST is a tax. It has slabs!"
        );
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
