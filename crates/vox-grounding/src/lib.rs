//! Evidence-sufficiency assessment and answer verification for VOX.
//!
//! The grounding stage answers two questions with deterministic lexical
//! heuristics:
//!
//! 1. **Is the retrieved evidence good enough to answer at all?**
//!    [`assess`] scores *relevance* (best single document's coverage of the
//!    query terms), *coverage* (union coverage across all documents), and
//!    *consistency* (do the relevant documents agree?) and maps the scores
//!    onto [`Answerability`] using configurable thresholds.
//! 2. **Is a generated answer actually supported by that evidence?**
//!    [`verify_answer`] checks how much of the answer's content appears in
//!    the evidence and flags numbers that appear nowhere in the evidence —
//!    the classic hallucination signal.
//!
//! These heuristics are deliberately simple and auditable; richer signals
//! (entailment models, citation span checks) can replace them behind the
//! same contract later.

use std::collections::HashSet;

use vox_types::{Answerability, RetrievedDocument};

/// Thresholds controlling the grounding verdicts. All ratios live in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundingConfig {
    /// Minimum best hybrid-retrieval score for evidence to count as strong.
    pub min_score: f32,
    /// Required best-single-document query-term coverage for `Supported`.
    pub relevance_min: f32,
    /// Required union query-term coverage for `Supported`.
    pub coverage_min: f32,
    /// Required agreement ratio among relevant documents for `Supported`.
    pub consistency_min: f32,
    /// Pairwise overlap coefficient above which two documents count as
    /// agreeing rather than conflicting.
    pub agreement_min: f32,
    /// Required share of an answer's content tokens present in the evidence.
    pub answer_support_min: f32,
}

impl Default for GroundingConfig {
    fn default() -> Self {
        Self {
            min_score: 0.30,
            relevance_min: 0.50,
            coverage_min: 0.50,
            consistency_min: 0.50,
            agreement_min: 0.10,
            answer_support_min: 0.60,
        }
    }
}

/// Per-document containment below which a document is not considered part of
/// the evidence set when judging consistency.
const RELEVANT_FLOOR: f32 = 0.10;

/// Lexical sufficiency scores for one retrieval result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SufficiencyScores {
    /// Best single-document coverage of the query's content tokens, in `[0,1]`.
    pub relevance: f32,
    /// Union coverage of the query's content tokens across all documents, in
    /// `[0,1]`.
    pub coverage: f32,
    /// Share of relevant-document pairs that agree, in `[0,1]`. `1.0` when
    /// fewer than two relevant documents exist (nothing can conflict).
    pub consistency: f32,
}

/// Outcome of the evidence-sufficiency assessment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Assessment {
    /// The verdict consumed by the evidence guard.
    pub answerability: Answerability,
    /// The underlying scores, for logging and debugging.
    pub scores: SufficiencyScores,
}

/// Outcome of grounding verification for one generated answer.
#[derive(Debug, Clone, PartialEq)]
pub struct AnswerVerification {
    /// Whether the answer may be served as grounded.
    pub supported: bool,
    /// Share of the answer's content tokens found in the evidence.
    pub support_ratio: f32,
    /// Digit-bearing answer tokens that appear nowhere in the evidence.
    pub unsupported_numbers: Vec<String>,
}

/// Tokenizes text for lexical matching: lowercased word runs with stopwords
/// dropped (a small English/Hindi/Tamil list) and single-character tokens
/// ignored.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if is_word_char(ch) {
            current.push(ch);
        } else if !current.is_empty() {
            push_token(&mut tokens, &current);
            current.clear();
        }
    }
    if !current.is_empty() {
        push_token(&mut tokens, &current);
    }
    tokens
}

/// Whether `ch` belongs to a word. Besides alphanumerics this accepts the
/// combining vowel signs, virama, and other marks of the Devanagari and
/// Tamil blocks — without them, Indic words split into meaningless
/// fragments (`குங்குமப்பூ` would become four pieces).
fn is_word_char(ch: char) -> bool {
    if ch.is_alphanumeric() {
        return true;
    }
    matches!(ch as u32,
        0x0900..=0x0903                       // Devanagari signs
        | 0x093A..=0x094F                     // Devanagari vowel signs, virama
        | 0x0951..=0x0957                     // Devanagari stress marks
        | 0x0962..=0x0963                     // Devanagari vocalic signs
        | 0x0B82 | 0x0B83                     // Tamil sign combinations
        | 0x0BBE..=0x0BC2 | 0x0BC6..=0x0BC8   // Tamil vowel signs
        | 0x0BCA..=0x0BCD | 0x0BD7            // Tamil virama, length mark
    )
}

fn push_token(tokens: &mut Vec<String>, raw: &str) {
    let token = raw.to_lowercase();
    if token.chars().count() >= 2 && !STOPWORDS.contains(&token.as_str()) {
        tokens.push(token);
    }
}

/// Small multilingual stopword list covering English plus the most common
/// Hindi and Tamil function words. Question words carry intent, not topic,
/// so they are excluded from lexical matching.
const STOPWORDS: &[&str] = &[
    // English
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "am",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "of",
    "in",
    "on",
    "for",
    "to",
    "from",
    "by",
    "with",
    "and",
    "or",
    "not",
    "can",
    "could",
    "should",
    "would",
    "will",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "at",
    "about",
    "into",
    "vs",
    "versus",
    "define",
    "meaning",
    // Hindi
    "क्या",
    "है",
    "हैं",
    "का",
    "की",
    "के",
    "में",
    "से",
    "पर",
    "और",
    "या",
    "एक",
    "को",
    "यह",
    "वह",
    "कैसे",
    "कब",
    "कौन",
    "क्यों",
    // Tamil
    "என்றால்",
    "என்ன",
    "எப்படி",
    "ஒரு",
    "மற்றும்",
    "இல்",
    "யார்",
    "எப்போது",
    "ஏன்",
];

/// Assesses whether the retrieved evidence suffices to ground an answer.
///
/// Deterministic: identical inputs produce identical verdicts.
#[must_use]
pub fn assess(
    query_text: &str,
    evidence: &[RetrievedDocument],
    config: &GroundingConfig,
) -> Assessment {
    let query_tokens = tokenize(query_text);
    if evidence.is_empty() || query_tokens.is_empty() {
        return no_evidence();
    }

    let query_set: HashSet<&String> = query_tokens.iter().collect();
    let doc_sets: Vec<HashSet<String>> = evidence
        .iter()
        .map(|doc| tokenize(&doc.text).into_iter().collect::<HashSet<_>>())
        .collect();

    let containments: Vec<f32> = doc_sets
        .iter()
        .map(|doc_set| {
            let hits = doc_set.iter().filter(|t| query_set.contains(*t)).count();
            hits as f32 / query_set.len() as f32
        })
        .collect();

    let relevance = containments.iter().copied().fold(0.0_f32, f32::max);

    let mut covered = 0usize;
    for token in &query_set {
        if doc_sets
            .iter()
            .any(|doc_set| doc_set.contains(token.as_str()))
        {
            covered += 1;
        }
    }
    let coverage = covered as f32 / query_set.len() as f32;

    let relevant: Vec<&HashSet<String>> = doc_sets
        .iter()
        .zip(&containments)
        .filter(|(_, containment)| **containment >= RELEVANT_FLOOR)
        .map(|(doc_set, _)| doc_set)
        .collect();
    let consistency = pairwise_agreement(&relevant, config.agreement_min);

    let scores = SufficiencyScores {
        relevance,
        coverage,
        consistency,
    };
    let answerability = verdict(&scores, evidence, config);
    Assessment {
        answerability,
        scores,
    }
}

fn no_evidence() -> Assessment {
    Assessment {
        answerability: Answerability::NoEvidence,
        scores: SufficiencyScores {
            relevance: 0.0,
            coverage: 0.0,
            consistency: 1.0,
        },
    }
}

fn verdict(
    scores: &SufficiencyScores,
    evidence: &[RetrievedDocument],
    config: &GroundingConfig,
) -> Answerability {
    if scores.relevance <= 0.0 {
        return Answerability::NoEvidence;
    }

    let best_score = evidence
        .iter()
        .map(|doc| doc.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let score_ok = best_score.is_finite() && best_score >= config.min_score;
    let coverage_ok = scores.coverage >= config.coverage_min;
    let relevance_ok = scores.relevance >= config.relevance_min;
    let consistency_ok = scores.consistency >= config.consistency_min;

    if relevance_ok && coverage_ok && score_ok && consistency_ok {
        return Answerability::Supported;
    }
    if !consistency_ok {
        return Answerability::ConflictingEvidence;
    }
    Answerability::WeakEvidence
}

/// Fraction of relevant-document pairs whose overlap coefficient reaches
/// `agreement_min`. One or zero relevant documents cannot conflict, so the
/// result is `1.0`.
fn pairwise_agreement(relevant: &[&HashSet<String>], agreement_min: f32) -> f32 {
    if relevant.len() < 2 {
        return 1.0;
    }
    let mut pairs = 0usize;
    let mut agreeing = 0usize;
    for i in 0..relevant.len() {
        for j in (i + 1)..relevant.len() {
            pairs += 1;
            let shared = relevant[i].intersection(relevant[j]).count();
            let smaller = relevant[i].len().min(relevant[j].len());
            let coefficient = if smaller == 0 {
                0.0
            } else {
                shared as f32 / smaller as f32
            };
            if coefficient >= agreement_min {
                agreeing += 1;
            }
        }
    }
    agreeing as f32 / pairs as f32
}

/// Verifies that a generated answer is grounded in the supplied evidence.
///
/// An answer counts as supported when enough of its content tokens appear in
/// the evidence (`answer_support_min`) and none of its digit-bearing tokens
/// are absent from the evidence — invented figures are the clearest
/// unsupported-claim signal.
#[must_use]
pub fn verify_answer(
    answer: &str,
    evidence: &[RetrievedDocument],
    config: &GroundingConfig,
) -> AnswerVerification {
    let answer_tokens = tokenize(answer);
    let mut evidence_union: HashSet<String> = HashSet::new();
    for doc in evidence {
        evidence_union.extend(tokenize(&doc.text));
    }

    let mut found = 0usize;
    let mut checked = 0usize;
    let mut unsupported_numbers = Vec::new();
    for token in &answer_tokens {
        checked += 1;
        if evidence_union.contains(token) {
            found += 1;
        }
        if token.chars().any(|ch| ch.is_ascii_digit()) && !evidence_union.contains(token) {
            unsupported_numbers.push(token.clone());
        }
    }

    let support_ratio = if checked == 0 {
        0.0
    } else {
        found as f32 / checked as f32
    };
    let supported =
        checked > 0 && support_ratio >= config.answer_support_min && unsupported_numbers.is_empty();

    AnswerVerification {
        supported,
        support_ratio,
        unsupported_numbers,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

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
    fn tokenize_should_lowercase_split_and_drop_stopwords() {
        let tokens = tokenize("What is the GST slabs in India?");
        assert_eq!(tokens, vec!["gst", "slabs", "india"]);
    }

    #[test]
    fn tokenize_should_handle_hindi_and_tamil_queries() {
        assert_eq!(tokenize("जीएसटी क्या है?"), vec!["जीएसटी"]);
        assert_eq!(tokenize("குங்குமப்பூ என்றால் என்ன?"), vec!["குங்குமப்பூ"]);
    }

    #[test]
    fn tokenize_should_drop_single_character_tokens() {
        assert_eq!(tokenize("a b cd"), vec!["cd"]);
    }

    #[test]
    fn assess_should_mark_corpus_evidence_as_supported() {
        let config = GroundingConfig::default();
        let evidence = [
            doc(
                "d1",
                0.95,
                "GST is an indirect tax used in India on the supply of goods and services.",
            ),
            doc(
                "d2",
                0.90,
                "GST in India is a destination-based tax with slabs of 5%, 12%, 18% and 28%.",
            ),
        ];
        let assessment = assess("what is gst tax in india", &evidence, &config);
        assert_eq!(assessment.answerability, Answerability::Supported);
        assert!(assessment.scores.relevance >= 0.5);
        assert!(assessment.scores.consistency >= 0.5);
    }

    #[test]
    fn assess_should_report_no_evidence_when_nothing_was_retrieved() {
        let assessment = assess("what is gst", &[], &GroundingConfig::default());
        assert_eq!(assessment.answerability, Answerability::NoEvidence);
    }

    #[test]
    fn assess_should_report_no_evidence_when_documents_share_no_query_terms() {
        let evidence = [doc(
            "d1",
            0.95,
            "Sample reference material from the offline corpus.",
        )];
        let assessment = assess(
            "quantum chromodynamics explained",
            &[evidence[0].clone()],
            &GroundingConfig::default(),
        );
        assert_eq!(assessment.answerability, Answerability::NoEvidence);
    }

    #[test]
    fn assess_should_report_weak_evidence_for_partial_matches() {
        let config = GroundingConfig::default();
        // Only 2 of 8 query terms appear; scores are fine but coverage is low.
        let evidence = [doc("d1", 0.95, "GST is a tax.")];
        let assessment = assess(
            "gst tax slabs india returns filing process deadline",
            &evidence,
            &config,
        );
        assert_eq!(assessment.answerability, Answerability::WeakEvidence);
    }

    #[test]
    fn assess_should_report_weak_evidence_when_scores_are_below_threshold() {
        let config = GroundingConfig::default();
        let evidence = [doc(
            "d1",
            0.10,
            "GST is an indirect tax on goods and services in India.",
        )];
        let assessment = assess("gst tax india", &evidence, &config);
        assert_eq!(assessment.answerability, Answerability::WeakEvidence);
    }

    #[test]
    fn assess_should_report_conflicting_evidence_when_relevant_docs_disagree() {
        let config = GroundingConfig::default();
        let evidence = [
            doc(
                "d1",
                0.95,
                "GST is a tax applied to goods and services everywhere.",
            ),
            doc(
                "d2",
                0.90,
                "Bats navigate at night using echolocation sounds.",
            ),
        ];
        // Both docs are partially relevant to the mixed query but share no
        // vocabulary with each other.
        let assessment = assess("gst tax bats echolocation night", &evidence, &config);
        assert_eq!(assessment.answerability, Answerability::ConflictingEvidence);
    }

    #[test]
    fn assess_should_be_deterministic() {
        let config = GroundingConfig::default();
        let evidence = [doc("d1", 0.9, "GST is a tax.")];
        let first = assess("gst tax", &evidence, &config);
        let second = assess("gst tax", &evidence, &config);
        assert_eq!(first, second);
    }

    #[test]
    fn verify_answer_should_accept_an_extractive_copy() {
        let config = GroundingConfig::default();
        let evidence = [doc(
            "d1",
            0.95,
            "GST is an indirect tax used in India on goods and services.",
        )];
        let verification =
            verify_answer("GST is an indirect tax used in India.", &evidence, &config);
        assert!(verification.supported);
        assert!((verification.support_ratio - 1.0).abs() < 1e-6);
        assert!(verification.unsupported_numbers.is_empty());
    }

    #[test]
    fn verify_answer_should_flag_invented_numbers() {
        let config = GroundingConfig::default();
        let evidence = [doc(
            "d1",
            0.95,
            "GST is an indirect tax used in India on goods and services.",
        )];
        let verification = verify_answer("GST rate is 42% for luxury cars.", &evidence, &config);
        assert!(!verification.supported);
        assert_eq!(verification.unsupported_numbers, vec!["42"]);
    }

    #[test]
    fn verify_answer_should_reject_answers_outside_the_evidence() {
        let config = GroundingConfig::default();
        let evidence = [doc("d1", 0.95, "GST is an indirect tax.")];
        let verification = verify_answer(
            "Platypuses are monotremes native to Australia.",
            &evidence,
            &config,
        );
        assert!(!verification.supported);
        assert!(verification.support_ratio < config.answer_support_min);
    }

    #[test]
    fn verify_answer_should_reject_empty_answers() {
        let verification = verify_answer(
            "   ",
            &[doc("d1", 0.9, "text")],
            &GroundingConfig::default(),
        );
        assert!(!verification.supported);
    }
}
