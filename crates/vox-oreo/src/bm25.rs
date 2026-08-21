//! Sparse retrieval: a Tantivy-backed BM25 index over chunks.
//!
//! The schema indexes only the chunk text for scoring; every metadata field
//! is stored so hits reconstruct into full [`ChunkRecord`]s without a second
//! lookup. The index persists in a directory shared by the `oreo index`,
//! `oreo serve`, and embedded-backend flows.

use std::path::{Path, PathBuf};

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

use crate::chunk::ChunkRecord;
use crate::error::OreoError;

/// Characters Tantivy's query parser treats as syntax; stripped from user
/// queries so arbitrary text never fails to parse.
const QUERY_SYNTAX_CHARS: [char; 18] = [
    '+', '-', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', '\\', ':', '/', '&',
];

/// A scored BM25 hit.
#[derive(Debug, Clone, PartialEq)]
pub struct Bm25Hit {
    /// Matched chunk with full metadata.
    pub chunk: ChunkRecord,
    /// BM25 score (higher is better).
    pub score: f32,
}

/// Persistent BM25 index over chunk texts.
///
/// Writes are sequential by design (ingestion is a batch job); reads are
/// served from a manual-reload searcher so freshly committed documents are
/// immediately visible.
pub struct TantivyBm25Index {
    dir: PathBuf,
    index: Index,
    reader: IndexReader,
    fields: Fields,
}

#[derive(Debug, Clone, Copy)]
struct Fields {
    text: tantivy::schema::Field,
    document_id: tantivy::schema::Field,
    chunk_id: tantivy::schema::Field,
    chunk_index: tantivy::schema::Field,
    language: tantivy::schema::Field,
    chunking_strategy: tantivy::schema::Field,
    source: tantivy::schema::Field,
}

impl TantivyBm25Index {
    fn build_schema() -> (Schema, Fields) {
        let mut builder = Schema::builder();
        let text_options = TextOptions::default().set_stored().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("default")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        );
        let text = builder.add_text_field("text", text_options);
        let document_id = builder.add_text_field("document_id", tantivy::schema::STORED);
        let chunk_id = builder.add_text_field("chunk_id", tantivy::schema::STORED);
        let chunk_index = builder.add_u64_field("chunk_index", tantivy::schema::STORED);
        let language = builder.add_text_field("language", tantivy::schema::STORED);
        let chunking_strategy =
            builder.add_text_field("chunking_strategy", tantivy::schema::STORED);
        let source = builder.add_text_field("source", tantivy::schema::STORED);
        (
            builder.build(),
            Fields {
                text,
                document_id,
                chunk_id,
                chunk_index,
                language,
                chunking_strategy,
                source,
            },
        )
    }

    /// Opens the index at `dir`, creating it when absent.
    ///
    /// # Errors
    /// Returns [`OreoError`] when the directory cannot be created or the
    /// existing index cannot be opened.
    pub fn open_or_create(dir: &Path) -> Result<Self, OreoError> {
        std::fs::create_dir_all(dir)?;
        let (schema, fields) = Self::build_schema();
        let exists = dir.join("meta.json").exists();
        let index = if exists {
            Index::open_in_dir(dir)?
        } else {
            Index::create_in_dir(dir, schema.clone())?
        };
        let mut writer: IndexWriter<TantivyDocument> = index.writer(64_000_000)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        writer.commit()?;
        Ok(Self {
            dir: dir.to_path_buf(),
            index,
            reader,
            fields,
        })
    }

    /// Adds one chunk and commits.
    ///
    /// # Errors
    /// Returns [`OreoError`] when writing or committing fails.
    pub fn add_chunk(&self, chunk: &ChunkRecord) -> Result<(), OreoError> {
        self.add_chunks(std::slice::from_ref(chunk))
    }

    /// Adds many chunks in one committed batch.
    ///
    /// # Errors
    /// Returns [`OreoError`] when writing or committing fails.
    pub fn add_chunks(&self, chunks: &[ChunkRecord]) -> Result<(), OreoError> {
        let mut writer: IndexWriter<TantivyDocument> = self.index.writer(64_000_000)?;
        for chunk in chunks {
            writer.add_document(doc!(
                self.fields.text => chunk.text.clone(),
                self.fields.document_id => chunk.document_id.clone(),
                self.fields.chunk_id => chunk.chunk_id.clone(),
                self.fields.chunk_index => u64::try_from(chunk.chunk_index).unwrap_or(u64::MAX),
                self.fields.language => chunk.language.code(),
                self.fields.chunking_strategy => chunk.chunking_strategy.clone(),
                self.fields.source => chunk.source.clone(),
            ))?;
        }
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Number of documents in the index.
    ///
    /// # Errors
    /// Returns [`OreoError`] when the reader cannot be acquired.
    pub fn len(&self) -> Result<usize, OreoError> {
        Ok(usize::try_from(self.reader.searcher().num_docs()).unwrap_or(0))
    }

    /// Whether the index holds no documents.
    ///
    /// # Errors
    /// Returns [`OreoError`] when the reader cannot be acquired.
    pub fn is_empty(&self) -> Result<bool, OreoError> {
        Ok(self.len()? == 0)
    }

    /// Directory backing this index.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Runs a BM25 keyword search, returning up to `top_k` hits ordered by
    /// descending score. Unparseable queries yield an empty result rather
    /// than an error, mirroring search-engine behaviour.
    ///
    /// # Errors
    /// Returns [`OreoError`] only for genuine index failures.
    pub fn search(&self, query_text: &str, top_k: usize) -> Result<Vec<Bm25Hit>, OreoError> {
        let sanitized: String = query_text
            .chars()
            .map(|ch| {
                if QUERY_SYNTAX_CHARS.contains(&ch) {
                    ' '
                } else {
                    ch
                }
            })
            .collect();
        if sanitized.split_whitespace().next().is_none() {
            return Ok(Vec::new());
        }
        let parser = QueryParser::for_index(&self.index, vec![self.fields.text]);
        let query = match parser.parse_query(sanitized.trim()) {
            Ok(query) => query,
            Err(_) => return Ok(Vec::new()),
        };
        let searcher = self.reader.searcher();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k.max(1)))?;
        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, address) in top_docs {
            let retrieved: TantivyDocument = searcher.doc(address)?;
            let Some(chunk) = self.reconstruct(&retrieved) else {
                continue;
            };
            hits.push(Bm25Hit { chunk, score });
        }
        Ok(hits)
    }

    fn reconstruct(&self, document: &TantivyDocument) -> Option<ChunkRecord> {
        let text = owned_str(document.get_first(self.fields.text))?;
        let document_id = owned_str(document.get_first(self.fields.document_id))?;
        let chunk_id = owned_str(document.get_first(self.fields.chunk_id))?;
        let chunk_index = match document.get_first(self.fields.chunk_index) {
            Some(tantivy::schema::OwnedValue::U64(value)) => usize::try_from(*value).ok()?,
            _ => 0,
        };
        let language = owned_str(document.get_first(self.fields.language))?;
        let chunking_strategy =
            owned_str(document.get_first(self.fields.chunking_strategy))?.to_owned();
        let source = owned_str(document.get_first(self.fields.source))?.to_owned();
        Some(ChunkRecord {
            document_id: document_id.to_owned(),
            chunk_id: chunk_id.to_owned(),
            chunk_index,
            text: text.to_owned(),
            language: language.parse().ok()?,
            chunking_strategy,
            source,
        })
    }
}

fn owned_str(value: Option<&tantivy::schema::OwnedValue>) -> Option<&str> {
    match value {
        Some(tantivy::schema::OwnedValue::Str(text)) => Some(text.as_str()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_types::Language;

    fn chunk(id: &str, text: &str) -> ChunkRecord {
        ChunkRecord {
            document_id: id.to_owned(),
            chunk_id: format!("{id}#0000"),
            chunk_index: 0,
            text: text.to_owned(),
            language: Language::En,
            chunking_strategy: "sentence".to_owned(),
            source: "test-corpus".to_owned(),
        }
    }

    fn temp_index() -> (tempfile::TempDir, TantivyBm25Index) {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = TantivyBm25Index::open_or_create(dir.path()).expect("open");
        (dir, index)
    }

    #[test]
    fn should_create_persist_and_reopen() {
        let dir_guard = tempfile::tempdir().expect("tempdir");
        {
            let index = TantivyBm25Index::open_or_create(dir_guard.path()).expect("create");
            index
                .add_chunks(&[chunk("a", "Goods and Services Tax overview")])
                .expect("add");
        }
        let reopened = TantivyBm25Index::open_or_create(dir_guard.path()).expect("reopen");
        assert_eq!(reopened.len().expect("len"), 1);
    }

    #[test]
    fn bm25_should_rank_matching_documents_first() {
        let (_guard, index) = temp_index();
        index
            .add_chunks(&[
                chunk(
                    "gst",
                    "GST is an indirect tax on goods and services in India",
                ),
                chunk(
                    "water",
                    "Chennai metro water connection apply online portal",
                ),
                chunk("itr", "Income tax return filing ITR-1 Sahaj form salary"),
            ])
            .expect("add");

        let hits = index.search("GST indirect tax", 3).expect("search");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].chunk.document_id, "gst");
        assert!(hits[0].score >= hits.last().expect("hit").score);

        // Metadata must survive the round trip through stored fields.
        assert_eq!(hits[0].chunk.chunk_id, "gst#0000");
        assert_eq!(hits[0].chunk.language, Language::En);
        assert_eq!(hits[0].chunk.chunking_strategy, "sentence");
        assert_eq!(hits[0].chunk.source, "test-corpus");
    }

    #[test]
    fn indic_queries_should_match_indic_documents() {
        let (_guard, index) = temp_index();
        index
            .add_chunks(&[
                chunk("hi", "जीएसटी एक अप्रत्यक्ष कर है जो वस्तु एवं सेवा पर लागू होता है"),
                chunk("en", "GST is an indirect tax applied to goods and services"),
            ])
            .expect("add");
        let hits = index.search("जीएसटी अप्रत्यक्ष कर", 2).expect("search");
        assert_eq!(hits.first().expect("hit").chunk.document_id, "hi");
    }

    #[test]
    fn unparseable_queries_should_return_empty_not_error() {
        let (_guard, index) = temp_index();
        index.add_chunks(&[chunk("a", "some text")]).expect("add");
        assert!(index.search("", 5).expect("empty query").is_empty());
        assert!(index.search("+-*/()", 5).expect("syntax only").is_empty());
    }

    #[test]
    fn limit_should_cap_results() {
        let (_guard, index) = temp_index();
        index
            .add_chunks(
                &(0..10)
                    .map(|i| chunk(&format!("d{i}"), "common words repeated here"))
                    .collect::<Vec<_>>(),
            )
            .expect("add");
        let hits = index.search("common words", 3).expect("search");
        assert_eq!(hits.len(), 3);
    }
}
