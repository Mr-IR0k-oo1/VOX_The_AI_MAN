//! Chunking abstraction: strategies for splitting documents into indexable
//! chunks while preserving full provenance metadata.
//!
//! Every chunk records `document_id`, `chunk_id` (`{document_id}#{index}`),
//! `language`, `chunking_strategy`, and `source`; these keys survive all the
//! way into Qdrant payloads, Tantivy stored fields, and API responses.

use serde::{Deserialize, Serialize};
use vox_types::Language;

use crate::preprocess::PreprocessedDocument;

/// Selectable chunking strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// [`FixedChunker`]: hard character windows with optional overlap.
    Fixed {
        /// Window size in characters.
        size: usize,
        /// Overlap between consecutive windows in characters.
        overlap: usize,
    },
    /// [`SentenceChunker`]: sentence-aware packing up to a character budget.
    Sentence {
        /// Maximum characters per chunk.
        max_chars: usize,
        /// Soft minimum; short trailing chunks are merged back when possible.
        min_chars: usize,
    },
    /// [`SlidingChunker`]: overlapping windows advanced by a stride.
    Sliding {
        /// Window size in characters.
        window: usize,
        /// Characters advanced between window starts.
        stride: usize,
    },
}

impl ChunkingStrategy {
    /// Strategy name recorded in chunk metadata.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Fixed { .. } => "fixed",
            Self::Sentence { .. } => "sentence",
            Self::Sliding { .. } => "sliding",
        }
    }

    /// Builds the concrete [`Chunker`] described by this strategy.
    #[must_use]
    pub fn build(&self) -> Box<dyn Chunker> {
        match self.clone() {
            Self::Fixed { size, overlap } => Box::new(FixedChunker::new(size, overlap)),
            Self::Sentence {
                max_chars,
                min_chars,
            } => Box::new(SentenceChunker::new(max_chars, min_chars)),
            Self::Sliding { window, stride } => Box::new(SlidingChunker::new(window, stride)),
        }
    }
}

/// A single chunk with its full provenance metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkRecord {
    /// Document the chunk came from.
    pub document_id: String,
    /// Stable chunk identifier (`{document_id}#{index}`).
    pub chunk_id: String,
    /// Zero-based position of the chunk inside its document.
    pub chunk_index: usize,
    /// Chunk text.
    pub text: String,
    /// Detected document language.
    pub language: Language,
    /// Name of the chunking strategy that produced this chunk.
    pub chunking_strategy: String,
    /// Provenance label of the source file.
    pub source: String,
}

/// Splits documents into chunks.
pub trait Chunker: Send + Sync {
    /// Strategy name; recorded in every chunk's metadata.
    fn name(&self) -> &'static str;

    /// Splits raw text into ordered chunks. Never returns an empty vec for
    /// non-empty input.
    fn split(&self, text: &str) -> Vec<String>;

    /// Convenience wrapper producing fully-attributed [`ChunkRecord`]s.
    fn chunk_document(&self, document: &PreprocessedDocument) -> Vec<ChunkRecord> {
        self.split(&document.text)
            .into_iter()
            .enumerate()
            .map(|(index, text)| ChunkRecord {
                chunk_id: format!("{}#{index:04}", document.id),
                document_id: document.id.clone(),
                chunk_index: index,
                text,
                language: document.language,
                chunking_strategy: self.name().to_owned(),
                source: document.source.clone(),
            })
            .collect()
    }
}

/// Hard character-window chunker with optional overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedChunker {
    size: usize,
    overlap: usize,
}

impl FixedChunker {
    /// Creates a fixed chunker; zero/negative effective sizes are clamped to
    /// safe values so configuration mistakes cannot wedge ingestion.
    #[must_use]
    pub const fn new(size: usize, overlap: usize) -> Self {
        Self { size, overlap }
    }
}

impl Chunker for FixedChunker {
    fn name(&self) -> &'static str {
        "fixed"
    }

    fn split(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() || self.size == 0 {
            return if chars.is_empty() {
                Vec::new()
            } else {
                vec![text.to_owned()]
            };
        }
        let step = self.size.saturating_sub(self.overlap).max(1);
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let end = (start + self.size).min(chars.len());
            chunks.push(chars[start..end].iter().collect());
            if end == chars.len() {
                break;
            }
            start += step;
        }
        chunks
    }
}

/// Sentence-aware chunker: packs whole sentences (splitting on `.`, `!`, `?`,
/// the Devanagari danda `।`, and newlines) into chunks under `max_chars`,
/// merging tiny trailing chunks below `min_chars`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentenceChunker {
    max_chars: usize,
    min_chars: usize,
}

impl SentenceChunker {
    /// Creates a sentence chunker.
    #[must_use]
    pub const fn new(max_chars: usize, min_chars: usize) -> Self {
        Self {
            max_chars,
            min_chars,
        }
    }
}

/// Terminal punctuation recognized as sentence boundaries across the
/// supported languages.
const SENTENCE_TERMINATORS: [char; 6] = ['.', '!', '?', '।', '॥', '\n'];

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if SENTENCE_TERMINATORS.contains(&ch) {
            let trimmed = current.trim().to_owned();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_owned());
    }
    sentences
}

impl Chunker for SentenceChunker {
    fn name(&self) -> &'static str {
        "sentence"
    }

    fn split(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let max = self.max_chars.max(1);
        let mut chunks: Vec<String> = Vec::new();
        let mut current = String::new();
        for sentence in split_sentences(text) {
            // A single over-long sentence becomes its own oversized chunk
            // rather than being cut mid-sentence.
            if sentence.chars().count() > max {
                if !current.is_empty() {
                    chunks.push(current.trim().to_owned());
                    current.clear();
                }
                chunks.push(sentence);
                continue;
            }
            if current.chars().count() + sentence.chars().count() + 1 > max {
                chunks.push(current.trim().to_owned());
                current.clear();
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(&sentence);
        }
        if !current.trim().is_empty() {
            chunks.push(current.trim().to_owned());
        }

        // Merge trailing chunks that fall below the soft minimum.
        let mut merged: Vec<String> = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            if let Some(last) = merged.last_mut() {
                if last.chars().count() < self.min_chars
                    && last.chars().count() + chunk.chars().count() < max * 2
                {
                    last.push(' ');
                    last.push_str(&chunk);
                    continue;
                }
            }
            merged.push(chunk);
        }
        merged
    }
}

/// Overlapping sliding-window chunker advancing by a fixed stride.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlidingChunker {
    window: usize,
    stride: usize,
}

impl SlidingChunker {
    /// Creates a sliding chunker.
    ///
    /// # Panics
    /// Never; invalid parameters are clamped in [`SlidingChunker::split`].
    #[must_use]
    pub const fn new(window: usize, stride: usize) -> Self {
        Self { window, stride }
    }
}

impl Chunker for SlidingChunker {
    fn name(&self) -> &'static str {
        "sliding"
    }

    fn split(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }
        let window = self.window.max(1);
        let stride = self.stride.max(1);
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let end = (start + window).min(chars.len());
            chunks.push(chars[start..end].iter().collect());
            if end == chars.len() {
                break;
            }
            start += stride;
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::RawDocument;

    fn document(text: &str) -> PreprocessedDocument {
        PreprocessedDocument {
            id: "doc-1".to_owned(),
            text: text.to_owned(),
            language: Language::En,
            source: "sample".to_owned(),
        }
    }

    #[test]
    fn fixed_chunker_should_respect_size_and_overlap() {
        let chunker = FixedChunker::new(10, 3);
        let chunks = chunker.split("abcdefghij klmnopqrst uvwxyz");
        assert!(chunks.len() >= 3);
        for chunk in &chunks[..chunks.len() - 1] {
            assert!(chunk.chars().count() <= 10);
        }
        // Overlap: consecutive chunks share characters.
        let tail: String = chunks[0]
            .chars()
            .skip(chunks[0].chars().count() - 3)
            .collect();
        let head: String = chunks[1].chars().take(3).collect();
        assert_eq!(tail, head);
    }

    #[test]
    fn fixed_chunker_should_return_single_chunk_for_short_text() {
        let chunker = FixedChunker::new(100, 10);
        assert_eq!(chunker.split("short text"), vec!["short text".to_owned()]);
        assert!(chunker.split("").is_empty());
    }

    #[test]
    fn sentence_chunker_should_split_on_danda_and_periods() {
        let chunker = SentenceChunker::new(60, 10);
        let text = "GST is an indirect tax. It replaced VAT! क्या यह सही है? जीएसटी एक कर है। tail";
        let chunks = chunker.split(text);
        assert!(chunks.len() >= 2);
        let joined = chunks.join(" ");
        for word in ["indirect", "VAT", "जीएसटी", "tail"] {
            assert!(joined.contains(word), "missing {word}");
        }
    }

    #[test]
    fn sentence_chunker_should_pack_under_max_chars() {
        let chunker = SentenceChunker::new(50, 5);
        let text = "First sentence here. Second sentence follows it. Third one arrives now.";
        let chunks = chunker.split(text);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 120, "oversized chunk: {chunk}");
        }
        assert!(chunks.len() >= 2 && chunks.len() <= 3);
    }

    #[test]
    fn sliding_chunker_should_overlap_by_window_minus_stride() {
        let chunker = SlidingChunker::new(8, 4);
        let chunks = chunker.split("abcdefghijklmnop");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "abcdefgh");
        assert_eq!(chunks[1], "efghijkl");
        assert_eq!(chunks[2], "ijklmnop");
    }

    #[test]
    fn chunk_document_should_preserve_full_metadata() {
        let doc = document("Sentence one ends here. Sentence two ends there.");
        let records = SentenceChunker::new(40, 5).chunk_document(&doc);
        assert!(!records.is_empty());
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record.document_id, "doc-1");
            assert_eq!(record.chunk_id, format!("doc-1#{index:04}"));
            assert_eq!(record.chunk_index, index);
            assert_eq!(record.language, Language::En);
            assert_eq!(record.chunking_strategy, "sentence");
            assert_eq!(record.source, "sample");
        }
    }

    #[test]
    fn strategy_factory_should_build_matching_chunkers() {
        let strategies = [
            ChunkingStrategy::Fixed {
                size: 32,
                overlap: 8,
            },
            ChunkingStrategy::Sentence {
                max_chars: 64,
                min_chars: 8,
            },
            ChunkingStrategy::Sliding {
                window: 16,
                stride: 8,
            },
        ];
        for strategy in strategies {
            let built = strategy.build();
            assert_eq!(built.name(), strategy.name());
        }
    }

    #[test]
    fn chunk_ids_should_be_unique_per_document() {
        let docs = vec![
            PreprocessedDocument {
                id: "a".into(),
                text: "One two three four five.".into(),
                language: Language::En,
                source: "s".into(),
            },
            PreprocessedDocument {
                id: "b".into(),
                text: "Six seven eight nine ten.".into(),
                language: Language::En,
                source: "s".into(),
            },
        ];
        let chunker = SentenceChunker::new(12, 4);
        let mut ids = Vec::new();
        for doc in &docs {
            ids.extend(chunker.chunk_document(doc).into_iter().map(|c| c.chunk_id));
        }
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len());
    }

    #[test]
    fn raw_documents_flow_through_chunking_metadata() {
        let raw = RawDocument {
            id: "hi-1".into(),
            text: "जीएसटी एक अप्रत्यक्ष कर है भारत में पूरी तरह".into(),
            source: "msmarco-xi".into(),
        };
        let (docs, _) = crate::preprocess::preprocess(&[raw], &[Language::Hi]).expect("preprocess");
        let records = SentenceChunker::new(30, 5).chunk_document(&docs[0]);
        assert_eq!(records[0].language, Language::Hi);
        assert_eq!(records[0].source, "msmarco-xi");
        assert_eq!(records[0].document_id, "hi-1");
    }
}
