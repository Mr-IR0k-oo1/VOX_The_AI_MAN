//! The evaluation query set: hand-judged multilingual queries with binary
//! relevance judgments against the bundled OREO sample corpus.
//!
//! Judgments were authored by inspecting each corpus document's content and
//! marking every document that directly answers the query. They are exact
//! `docid`s from `crates/vox-oreo/data/sample-docs.jsonl`, so the evaluation
//! is fully reproducible offline.

use serde::Deserialize;
use vox_types::Language;

/// One judged evaluation query.
#[derive(Debug, Clone, Deserialize)]
pub struct EvalQuery {
    /// Stable query identifier (`en-01`, `hi-07`, ...).
    pub qid: String,
    /// Natural-language query text in the query language.
    pub query: String,
    /// Query language; matches the language of the relevant documents.
    pub language: Language,
    /// Document ids (corpus `docid`s) judged relevant to this query.
    pub relevant: Vec<String>,
}

/// The bundled evaluation set, compiled into the binary for reproducibility.
pub const EVAL_QUERIES_JSONL: &str = include_str!("../data/eval-queries.jsonl");

/// Parses the bundled evaluation set.
///
/// # Panics
/// Never in practice: the bundled file is compile-time validated by tests.
#[must_use]
pub fn eval_queries() -> Vec<EvalQuery> {
    EVAL_QUERIES_JSONL
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|err| panic!("evaluation query line failed to parse: {err}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_set_should_parse_and_cover_three_languages() {
        let queries = eval_queries();
        assert_eq!(queries.len(), 36);
        assert!(queries.iter().all(|q| !q.relevant.is_empty()));
        for language in [Language::En, Language::Hi, Language::Ta] {
            let count = queries.iter().filter(|q| q.language == language).count();
            assert_eq!(count, 12, "expected 12 queries per language");
        }
    }

    #[test]
    fn eval_judgments_should_reference_sample_corpus_documents() {
        let corpus = vox_oreo::ingest::sample_corpus();
        let ids: std::collections::HashSet<&str> = corpus.iter().map(|d| d.id.as_str()).collect();
        for query in eval_queries() {
            for relevant in &query.relevant {
                assert!(
                    ids.contains(relevant.as_str()),
                    "query {} judges unknown document {relevant}",
                    query.qid
                );
            }
        }
    }
}
