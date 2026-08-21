//! Deterministic in-memory retrieval backend.

use std::time::Duration;

use async_trait::async_trait;
use vox_types::{Query, RetrievalResponse, RetrievedDocument};

use crate::error::RetrievalError;
use crate::RetrievalClient;

/// Placeholder backend used until the real retrieval service is stable.
///
/// Returns deterministic sample evidence so the full pipeline can be
/// exercised end-to-end without Qdrant/Tantivy: a small curated corpus for
/// well-known topics, plus generic documents derived from the query for
/// everything else. An optional artificial delay supports timeout/retry
/// testing.
#[derive(Debug, Clone)]
pub struct MockRetrievalClient {
    delay: Duration,
}

impl MockRetrievalClient {
    /// Creates a mock backend that sleeps for `delay` before answering.
    #[must_use]
    pub const fn new(delay: Duration) -> Self {
        Self { delay }
    }

    /// Content vocabulary the sample corpus can speak about, used by callers
    /// to configure input-guard off-topic checks. Deliberately includes a few
    /// legitimate terms with no corpus coverage so that well-formed but
    /// unsupported questions still reach retrieval and exercise the
    /// insufficient-context refusal path.
    #[must_use]
    pub fn topic_vocabulary() -> Vec<String> {
        let mut words: Vec<String> = CORPUS
            .iter()
            .flat_map(|entry| entry.triggers.iter())
            .flat_map(|trigger| trigger.split_whitespace())
            .map(str::to_owned)
            .collect();
        words.extend(
            ["itr", "income", "return", "filing"]
                .into_iter()
                .map(str::to_owned),
        );
        words.sort();
        words.dedup();
        words
    }
}

impl Default for MockRetrievalClient {
    fn default() -> Self {
        Self::new(Duration::ZERO)
    }
}

/// One entry of the canned sample corpus.
struct CorpusEntry {
    /// Lowercase tokens that activate this entry.
    triggers: &'static [&'static str],
    /// Deterministic passages served when activated.
    passages: &'static [&'static str],
}

const CORPUS: [CorpusEntry; 4] = [
    CorpusEntry {
        triggers: &["artificial intelligence", "ai", "machine learning"],
        passages: &[
            "Artificial intelligence (AI) is the simulation of human intelligence \
             processes by computer systems. Machine learning is a subset of AI in \
             which models learn patterns from data.",
            "Modern AI systems are built on neural networks trained on large \
             datasets. Applications include speech recognition, translation, and \
             question answering.",
        ],
    },
    CorpusEntry {
        triggers: &["gst", "goods and services tax", "tax"],
        passages: &[
            "Goods and Services Tax (GST) is an indirect tax used in India on the \
             supply of goods and services. It replaced multiple cascading taxes \
             levied by the central and state governments.",
            "GST in India is a multi-stage, destination-based tax with slabs of \
             0%, 5%, 12%, 18%, and 28%. Returns are filed monthly or quarterly \
             depending on turnover.",
        ],
    },
    CorpusEntry {
        triggers: &["जीएसटी", "वस्तु एवं सेवा कर"],
        passages: &[
            "जीएसटी यानी वस्तु एवं सेवा कर भारत में लगाया जाने वाला अप्रत्यक्ष कर है। \
             इसने कई पुराने करों की जगह ली है।",
            "भारत में जीएसटी के स्लैब 5%, 12%, 18% और 28% हैं। टर्नओवर के आधार पर \
             रिटर्न मासिक या त्रैमासिक भरे जाते हैं।",
        ],
    },
    CorpusEntry {
        triggers: &["குங்குமப்பூ"],
        passages: &[
            "குங்குமப்பூ என்பது குரோக்கஸ் சாடிவஸ் மலரின் மகரந்தத்திலிருந்து \
             பெறப்படும் ஒரு மசாலா பொருள் ஆகும்.",
            "குங்குமப்பூ உலகின் மிக விலையுயர்ந்த மசாலா ஆகும். இது உணவு, \
             மருத்துவம் மற்றும் அழகுசாதனங்களில் பயன்படுகிறது.",
        ],
    },
];

fn corpus_documents(query_lower: &str) -> Vec<&'static str> {
    let mut passages = Vec::new();
    for entry in &CORPUS {
        if entry.triggers.iter().any(|t| query_lower.contains(t)) {
            passages.extend_from_slice(entry.passages);
        }
    }
    passages
}

#[async_trait]
impl RetrievalClient for MockRetrievalClient {
    fn name(&self) -> &'static str {
        "mock"
    }

    async fn retrieve(&self, request: Query) -> Result<RetrievalResponse, RetrievalError> {
        request.validate()?;
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }

        let normalized = if request.normalized_text.is_empty() {
            request.text.trim()
        } else {
            request.normalized_text.trim()
        };
        let lower = normalized.to_lowercase();
        let mut passages: Vec<String> = corpus_documents(&lower)
            .into_iter()
            .map(str::to_owned)
            .collect();

        // Fill remaining slots with deterministic neutral documents. The
        // filler deliberately shares no vocabulary with the query: queries
        // outside the corpus must look unsupported to the grounding stage
        // instead of being echoed back as fake evidence.
        let top_k = usize::from(request.top_k);
        let mut filler = 0usize;
        while passages.len() < top_k {
            passages.push(format!(
                "General reference note {filler} from the offline sample corpus. \
                 It holds background material unrelated to specific requests."
            ));
            filler += 1;
        }

        let language_code = request
            .language
            .map_or_else(String::new, |lang| lang.code().to_owned());
        let documents = passages
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(i, text)| RetrievedDocument {
                id: format!("mock-{i:04}"),
                text: text.to_owned(),
                score: (0.95 - i as f32 * 0.05).max(0.10),
                rank: u32::try_from(i).expect("index fits u32") + 1,
                metadata: serde_json::json!({
                    "source": "mock",
                    "language": language_code,
                }),
            })
            .collect();

        Ok(RetrievalResponse {
            documents,
            timings_ms: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use vox_types::{Language, Query, QueryIntent, ValidationError};

    use super::*;

    fn request(query: &str, top_k: u8) -> Query {
        Query::new(query, Some(Language::Ta), top_k)
    }

    #[tokio::test]
    async fn retrieve_should_return_deterministic_ranked_results() {
        let client = MockRetrievalClient::default();
        let first = client
            .retrieve(request("what is gst", 3))
            .await
            .expect("first");
        let second = client
            .retrieve(request("what is gst", 3))
            .await
            .expect("second");

        assert_eq!(first.documents.len(), 3);
        assert_eq!(first.documents[0].rank, 1);
        assert_eq!(first.documents[2].id, "mock-0002");
        assert_eq!(first.documents, second.documents);
    }

    #[tokio::test]
    async fn retrieve_should_serve_corpus_passages_for_known_topics() {
        let client = MockRetrievalClient::default();
        let response = client
            .retrieve(request("What is artificial intelligence?", 2))
            .await
            .expect("retrieve");

        assert_eq!(response.documents.len(), 2);
        assert!(
            response.documents[0]
                .text
                .contains("simulation of human intelligence"),
            "AI corpus passage expected"
        );
    }

    #[tokio::test]
    async fn retrieve_should_fall_back_to_neutral_documents() {
        let client = MockRetrievalClient::default();
        let response = client
            .retrieve(request("obscure topic xyz", 2))
            .await
            .expect("retrieve");

        assert!(response.documents[0]
            .text
            .contains("General reference note 0"));
        assert!(response.documents[1]
            .text
            .contains("General reference note 1"));
        // The filler must not echo query terms back as fake evidence.
        assert!(!response.documents[0].text.contains("obscure"));
    }

    #[tokio::test]
    async fn retrieve_should_serve_hindi_and_tamil_corpus_entries() {
        let client = MockRetrievalClient::default();

        let hindi = client
            .retrieve(request("जीएसटी क्या है?", 2))
            .await
            .expect("hindi");
        assert!(hindi.documents[0].text.contains("जीएसटी"));

        let tamil = client
            .retrieve(request("குங்குமப்பூ என்றால் என்ன?", 2))
            .await
            .expect("tamil");
        assert!(tamil.documents[0].text.contains("குங்குமப்பூ"));
    }

    #[test]
    fn topic_vocabulary_should_cover_corpus_words_and_unsupported_terms() {
        let vocab = MockRetrievalClient::topic_vocabulary();
        for word in ["gst", "tax", "ai", "जीएसटी", "குங்குமப்பூ", "itr"]
        {
            assert!(vocab.iter().any(|w| w == word), "missing {word}");
        }
    }

    #[tokio::test]
    async fn retrieve_should_use_normalized_text_when_present() {
        let client = MockRetrievalClient::default();
        let mut query = request("  what   is GST?  ", 1);
        query.normalized_text = "what is GST?".to_owned();

        let response = client.retrieve(query).await.expect("retrieve");
        assert!(response.documents[0].text.contains("GST"));
    }

    #[tokio::test]
    async fn retrieve_should_reject_invalid_input_without_delay() {
        let client = MockRetrievalClient::new(Duration::from_secs(60));
        let err = client
            .retrieve(request("   ", 3))
            .await
            .expect_err("should reject");
        assert!(matches!(
            err,
            RetrievalError::Validation(ValidationError::EmptyQuery)
        ));
    }

    #[test]
    fn intent_field_should_roundtrip_through_the_query_contract() {
        let mut query = request("q", 5);
        query.intent = QueryIntent::Procedural;
        let json = serde_json::to_string(&query).expect("serialize");
        assert!(json.contains("\"intent\":\"procedural\""));
    }
}
