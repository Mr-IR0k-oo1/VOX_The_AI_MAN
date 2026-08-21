//! Deterministic prompt construction for grounded answer generation.
//!
//! Every provider receives the same system instructions: use only the
//! supplied evidence, never invent unsupported claims, say so explicitly
//! when the evidence is insufficient, and answer in the request language.

use crate::LlmRequest;

/// Maximum characters of evidence text included per document.
const MAX_DOC_CHARS: usize = 1_200;

/// A fully rendered prompt ready for a chat-style provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// System message: grounding rules.
    pub system: String,
    /// User message: evidence blocks + the question.
    pub user: String,
}

/// Renders the grounding prompt for `request`.
///
/// The output is deterministic: same request, same prompt bytes.
#[must_use]
pub fn build_prompt(request: &LlmRequest) -> Prompt {
    let system = format!(
        "You are a question-answering assistant for VOX. Follow these rules exactly:\n\
         1. Use ONLY the supplied evidence to answer. Do not use outside knowledge.\n\
         2. Never invent claims that are not supported by the evidence.\n\
         3. If the evidence is insufficient or unrelated to the question, state \
         explicitly that you do not have enough information instead of guessing.\n\
         4. Write your entire answer in {name} ({code}).",
        name = request.language.display_name(),
        code = request.language.code(),
    );

    let mut user = String::from("Evidence:\n");
    if request.evidence.is_empty() {
        user.push_str("(no evidence was retrieved)\n");
    } else {
        for (i, doc) in request.evidence.iter().enumerate() {
            user.push_str(&format!(
                "[{}] (id: {}, score: {:.2})\n",
                i + 1,
                doc.id,
                doc.score
            ));
            user.push_str(truncate_chars(&doc.text, MAX_DOC_CHARS));
            user.push_str("\n\n");
        }
    }
    user.push_str("Question: ");
    user.push_str(request.question.trim());
    user.push('\n');
    user.push_str("Answer using only the evidence above. If it is not enough, say so.");

    Prompt { system, user }
}

/// Truncates on a char boundary to at most `max` characters.
fn truncate_chars(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use vox_types::{Language, RetrievedDocument};

    use super::*;

    fn doc(id: &str, score: f32, text: &str) -> RetrievedDocument {
        RetrievedDocument {
            id: id.to_owned(),
            text: text.to_owned(),
            score,
            rank: 1,
            metadata: Value::Null,
        }
    }

    #[test]
    fn prompt_should_carry_all_four_grounding_instructions() {
        let request = LlmRequest::new(
            "What is GST?",
            Language::En,
            vec![doc("d1", 0.9, "GST is a tax.")],
        );
        let prompt = build_prompt(&request);

        assert!(prompt.system.contains("ONLY the supplied evidence"));
        assert!(prompt.system.contains("Never invent"));
        assert!(prompt.system.contains("not have enough information"));
        assert!(prompt.system.contains("English (en)"));
        assert!(prompt.user.contains("[1] (id: d1, score: 0.90)"));
        assert!(prompt.user.contains("GST is a tax."));
        assert!(prompt.user.contains("Question: What is GST?"));
    }

    #[test]
    fn prompt_should_instruct_the_requested_language() {
        for (lang, expected) in [
            (Language::Hi, "Hindi (hi)"),
            (Language::Ta, "Tamil (ta)"),
            (Language::En, "English (en)"),
        ] {
            let request = LlmRequest::new("q", lang, vec![]);
            let prompt = build_prompt(&request);
            assert!(
                prompt.system.contains(expected),
                "missing language instruction for {expected}"
            );
        }
    }

    #[test]
    fn prompt_should_handle_empty_evidence_explicitly() {
        let request = LlmRequest::new("q", Language::En, vec![]);
        let prompt = build_prompt(&request);
        assert!(prompt.user.contains("(no evidence was retrieved)"));
    }

    #[test]
    fn prompt_should_be_deterministic() {
        let request = LlmRequest::new("q", Language::Ta, vec![doc("d1", 0.5, "text")]);
        assert_eq!(build_prompt(&request), build_prompt(&request));
    }

    #[test]
    fn truncate_should_cut_on_char_boundaries() {
        // 'अ' is 3 bytes; a naive byte-cut would split the glyph.
        let text = "अ".repeat(50);
        let cut = truncate_chars(&text, 10);
        assert_eq!(cut.chars().count(), 10);
        assert!(cut.chars().all(|c| c == 'अ'));
    }

    #[test]
    fn truncate_should_leave_short_text_untouched() {
        assert_eq!(truncate_chars("short", 10), "short");
    }
}
