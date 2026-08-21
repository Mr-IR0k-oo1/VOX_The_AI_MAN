//! Conservative query analysis: normalization and intent detection.
//!
//! Normalization is deliberately conservative — whitespace and punctuation
//! cleanup only. Text is never translated or transliterated; the original
//! language is preserved end to end.

use vox_types::QueryIntent;

/// Normalizes a query without altering its language.
///
/// - trims the ends
/// - collapses all whitespace runs to single spaces
/// - maps typographic variants (curly quotes, en/em dashes, NBSP) onto their
///   ASCII/plain equivalents
/// - collapses repeated punctuation runs (`??!` → `?`)
#[must_use]
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_was_space = true; // suppresses leading whitespace
    let mut prev_punct: Option<char> = None;

    for ch in text.chars() {
        let ch = map_char(ch);

        if ch.is_whitespace() {
            if !prev_was_space {
                out.push(' ');
                prev_was_space = true;
            }
            continue;
        }

        if let Some(punct) = punct_of(ch) {
            // Collapse runs of the same punctuation character.
            if prev_punct == Some(punct) {
                continue;
            }
            prev_punct = Some(punct);
        } else {
            prev_punct = None;
        }

        out.push(ch);
        prev_was_space = false;
    }

    while out.ends_with(' ') {
        out.pop();
    }
    out
}

fn map_char(ch: char) -> char {
    match ch {
        '\u{2018}' | '\u{2019}' | '\u{201B}' => '\'', // single quotes
        '\u{201C}' | '\u{201D}' => '"',               // double quotes
        '\u{2013}' | '\u{2014}' => '-',               // en/em dashes
        '\u{00A0}' | '\u{2007}' | '\u{202F}' => ' ',  // non-breaking spaces
        _ => ch,
    }
}

fn punct_of(ch: char) -> Option<char> {
    match ch {
        '!' | '?' | ',' | '.' | ';' | ':' => Some(ch),
        _ => None,
    }
}

/// Classifies a normalized query with deterministic keyword heuristics.
///
/// English markers are checked first, then a small set of high-precision
/// Hindi and Tamil markers. Anything unmatched is [`QueryIntent::Unknown`].
#[must_use]
pub fn detect_intent(normalized: &str) -> QueryIntent {
    let lower = normalized.to_lowercase();

    const COMPARISON: [&str; 5] = ["difference between", "compare ", " vs ", "versus ", "अंतर"];
    const PROCEDURAL: [&str; 7] = [
        "how to ",
        "how do ",
        "how can ",
        "how should ",
        "steps to ",
        "कैसे",
        "எப்படி",
    ];
    const DEFINITION: [&str; 8] = [
        "what is ",
        "what are ",
        "define ",
        "definition of ",
        "meaning of ",
        "क्या है",
        "क्या होता है",
        "என்றால் என்ன",
    ];
    const FACTUAL: [&str; 9] = [
        "who is ",
        "who was ",
        "when did ",
        "when was ",
        "where is ",
        "why is ",
        "why do ",
        "कौन",
        "कब",
    ];

    if COMPARISON.iter().any(|m| lower.contains(m)) {
        return QueryIntent::Comparison;
    }
    if PROCEDURAL.iter().any(|m| lower.contains(m)) {
        return QueryIntent::Procedural;
    }
    if DEFINITION.iter().any(|m| lower.contains(m)) {
        return QueryIntent::Definition;
    }
    if FACTUAL.iter().any(|m| lower.contains(m)) {
        return QueryIntent::Factual;
    }
    QueryIntent::Unknown
}

/// Runs the full query-analysis stage: normalization + intent detection.
///
/// Returns `(normalized_text, intent)`.
#[must_use]
pub fn analyze(text: &str) -> (String, QueryIntent) {
    let normalized = normalize(text);
    let intent = detect_intent(&normalized);
    (normalized, intent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_should_collapse_whitespace_and_trim() {
        assert_eq!(normalize("  What   is \n GST?\t"), "What is GST?");
    }

    #[test]
    fn normalize_should_collapse_repeated_punctuation() {
        // Runs of the same character collapse; alternating marks are kept.
        assert_eq!(normalize("wait...."), "wait.");
        assert_eq!(normalize("really?!"), "really?!");
        assert_eq!(normalize("stop!!!  ok??"), "stop! ok?");
    }

    #[test]
    fn normalize_should_map_typographic_variants() {
        assert_eq!(
            normalize("\u{201C}hi\u{201D} \u{2014} ok\u{00A0}now"),
            "\"hi\" - ok now"
        );
    }

    #[test]
    fn normalize_should_preserve_tamil_text() {
        assert_eq!(normalize("  தமிழ்   வணக்கம்  "), "தமிழ் வணக்கம்");
    }

    #[test]
    fn normalize_should_preserve_hindi_text() {
        assert_eq!(normalize(" नमस्ते   दुनिया "), "नमस्ते दुनिया");
    }

    #[test]
    fn normalize_should_handle_empty_input() {
        assert_eq!(normalize("   "), "");
    }

    #[test]
    fn intent_should_detect_definitions() {
        assert_eq!(
            detect_intent("what is artificial intelligence?"),
            QueryIntent::Definition
        );
        assert_eq!(detect_intent("gst क्या है"), QueryIntent::Definition);
    }

    #[test]
    fn intent_should_detect_comparisons_before_definitions() {
        assert_eq!(
            detect_intent("what is the difference between tcp and udp"),
            QueryIntent::Comparison
        );
        assert_eq!(
            detect_intent("compare petrol vs diesel"),
            QueryIntent::Comparison
        );
    }

    #[test]
    fn intent_should_detect_procedures() {
        assert_eq!(
            detect_intent("how to file itr online"),
            QueryIntent::Procedural
        );
        assert_eq!(detect_intent("அதை எப்படி செய்வது"), QueryIntent::Procedural);
    }

    #[test]
    fn intent_should_detect_facts() {
        assert_eq!(
            detect_intent("who is the finance minister"),
            QueryIntent::Factual
        );
        assert_eq!(detect_intent("why is the sky blue"), QueryIntent::Factual);
    }

    #[test]
    fn intent_should_default_to_unknown() {
        assert_eq!(detect_intent("tell me about cricket"), QueryIntent::Unknown);
    }

    #[test]
    fn analyze_should_return_both_outputs() {
        let (normalized, intent) = analyze("  WHAT   IS  AI?? ");
        assert_eq!(normalized, "WHAT IS AI?");
        assert_eq!(intent, QueryIntent::Definition);
    }
}
