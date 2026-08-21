//! Runtime configuration for the OREO retrieval engine.
//!
//! Every knob is environment-overridable so the same binary serves local
//! tests, the standalone OREO service, and the embedded backend inside
//! `vox-api`. Language support is a first-class setting: adding a language is
//! a configuration change, not a code change.

use std::path::PathBuf;
use std::time::Duration;

use vox_types::Language;

use crate::chunk::ChunkingStrategy;
use crate::error::OreoError;
use crate::rerank::RerankerKind;

/// Which storage backend backs dense retrieval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorStoreKind {
    /// In-process brute-force cosine store (tests, demos, small corpora).
    Memory,
    /// Qdrant server reached over REST.
    Qdrant,
}

/// Full engine configuration.
#[derive(Debug, Clone)]
pub struct OreoConfig {
    /// Languages kept by preprocessing; documents in other languages are
    /// skipped. Defaults to English, Hindi, Tamil.
    pub languages: Vec<Language>,
    /// Chunking strategy applied to every document.
    pub chunking: ChunkingStrategy,
    /// Embedding vector dimensionality.
    pub embedding_dim: usize,
    /// Where dense vectors live.
    pub vector_store: VectorStoreKind,
    /// Base URL of the Qdrant server (required for [`VectorStoreKind::Qdrant`]).
    pub qdrant_url: String,
    /// Qdrant request timeout.
    pub qdrant_timeout: Duration,
    /// Qdrant collection name.
    pub qdrant_collection: String,
    /// Directory holding the Tantivy BM25 index.
    pub tantivy_dir: PathBuf,
    /// BM25 candidate pool fetched per leg before fusion.
    ///
    /// The fused pool (task spec: top 20) is drawn from these candidates.
    pub candidate_top: usize,
    /// Final result count produced by the reranker (task spec: top 5).
    pub final_top: usize,
    /// Reciprocal Rank Fusion constant `k`.
    pub rrf_k: usize,
    /// Reranker used to order and trim fused candidates.
    pub reranker: RerankerKind,
}

impl Default for OreoConfig {
    fn default() -> Self {
        Self {
            languages: vec![Language::En, Language::Hi, Language::Ta],
            chunking: ChunkingStrategy::Sentence {
                max_chars: 700,
                min_chars: 80,
            },
            embedding_dim: 256,
            vector_store: VectorStoreKind::Memory,
            qdrant_url: "http://localhost:6333".to_owned(),
            qdrant_timeout: Duration::from_millis(500),
            qdrant_collection: "vox-chunks".to_owned(),
            tantivy_dir: PathBuf::from("data/oreo-index"),
            candidate_top: 20,
            final_top: 5,
            rrf_k: 60,
            reranker: RerankerKind::LexicalOverlap,
        }
    }
}

impl OreoConfig {
    /// Validates internal consistency.
    ///
    /// # Errors
    /// Returns [`OreoError::Config`] when dimensions/pool sizes are zero,
    /// when the candidate pool is smaller than the final result count, or
    /// when Qdrant is selected without a base URL.
    pub fn validate(&self) -> Result<(), OreoError> {
        if self.embedding_dim == 0 {
            return Err(OreoError::Config("embedding_dim must be > 0".into()));
        }
        if self.candidate_top == 0 || self.final_top == 0 {
            return Err(OreoError::Config(
                "candidate_top and final_top must be > 0".into(),
            ));
        }
        if self.final_top > self.candidate_top {
            return Err(OreoError::Config(
                "final_top cannot exceed candidate_top".into(),
            ));
        }
        if self.rrf_k == 0 {
            return Err(OreoError::Config("rrf_k must be > 0".into()));
        }
        if self.vector_store == VectorStoreKind::Qdrant && self.qdrant_url.trim().is_empty() {
            return Err(OreoError::Config(
                "qdrant_url is required when the vector store kind is qdrant".into(),
            ));
        }
        Ok(())
    }

    /// Loads configuration from the process environment over [`OreoConfig::default`].
    ///
    /// # Errors
    /// Returns [`OreoError::Config`] when a variable fails to parse or the
    /// resulting combination is inconsistent.
    pub fn from_env() -> Result<Self, OreoError> {
        Self::from_source(|name| std::env::var(name).ok())
    }

    /// Loads configuration from an arbitrary name→value source (testable).
    ///
    /// # Errors
    /// Same contract as [`OreoConfig::from_env`].
    pub fn from_source(source: impl Fn(&str) -> Option<String>) -> Result<Self, OreoError> {
        let mut config = Self::default();
        if let Some(raw) = source("VOX_OREO_LANGUAGES") {
            let mut languages = Vec::new();
            for token in raw.split(',') {
                let token = token.trim();
                if token.is_empty() {
                    continue;
                }
                let language = token
                    .parse::<Language>()
                    .map_err(|err| OreoError::Config(err.to_string()))?;
                languages.push(language);
            }
            if !languages.is_empty() {
                config.languages = languages;
            }
        }
        if let Some(raw) = source("VOX_OREO_CHUNKING") {
            config.chunking = parse_chunking(&raw)?;
        }
        if let Some(raw) = source("VOX_OREO_EMBEDDING_DIM") {
            config.embedding_dim = parse_usize("VOX_OREO_EMBEDDING_DIM", &raw)?;
        }
        match source("VOX_OREO_VECTOR_STORE").as_deref() {
            Some("memory") => config.vector_store = VectorStoreKind::Memory,
            Some("qdrant") => config.vector_store = VectorStoreKind::Qdrant,
            Some(other) => {
                return Err(OreoError::Config(format!(
                    "unknown VOX_OREO_VECTOR_STORE value: {other}"
                )))
            }
            None => {}
        }
        if let Some(raw) = source("VOX_OREO_QDRANT_URL") {
            config.qdrant_url = raw.trim().trim_end_matches('/').to_owned();
        }
        if let Some(raw) = source("VOX_OREO_COLLECTION") {
            config.qdrant_collection = raw.trim().to_owned();
        }
        if let Some(raw) = source("VOX_OREO_TANTIVY_DIR") {
            config.tantivy_dir = PathBuf::from(raw.trim());
        }
        if let Some(raw) = source("VOX_OREO_CANDIDATE_TOP") {
            config.candidate_top = parse_usize("VOX_OREO_CANDIDATE_TOP", &raw)?;
        }
        if let Some(raw) = source("VOX_OREO_FINAL_TOP") {
            config.final_top = parse_usize("VOX_OREO_FINAL_TOP", &raw)?;
        }
        if let Some(raw) = source("VOX_OREO_RRF_K") {
            config.rrf_k = parse_usize("VOX_OREO_RRF_K", &raw)?;
        }
        match source("VOX_OREO_RERANKER").as_deref() {
            Some("lexical_overlap") => config.reranker = RerankerKind::LexicalOverlap,
            Some("noop") => config.reranker = RerankerKind::Noop,
            Some(other) => {
                return Err(OreoError::Config(format!(
                    "unknown VOX_OREO_RERANKER value: {other}"
                )))
            }
            None => {}
        }
        config.validate()?;
        Ok(config)
    }
}

fn parse_usize(name: &str, raw: &str) -> Result<usize, OreoError> {
    raw.trim()
        .parse::<usize>()
        .map_err(|err| OreoError::Config(format!("{name}: {err}")))
}

fn parse_chunking(raw: &str) -> Result<ChunkingStrategy, OreoError> {
    let parse_usize_field = |raw: Option<&str>, default: usize| -> Result<usize, OreoError> {
        match raw {
            Some(value) => value
                .trim()
                .parse::<usize>()
                .map_err(|err| OreoError::Config(format!("chunking parameter: {err}"))),
            None => Ok(default),
        }
    };
    let mut parts = raw.trim().split(':');
    match parts.next() {
        Some("fixed") => {
            let size = parse_usize_field(parts.next(), 900)?;
            let overlap = parse_usize_field(parts.next(), 120)?;
            Ok(ChunkingStrategy::Fixed { size, overlap })
        }
        Some("sentence") => {
            let max_chars = parse_usize_field(parts.next(), 700)?;
            let min_chars = parse_usize_field(parts.next(), 80)?;
            Ok(ChunkingStrategy::Sentence {
                max_chars,
                min_chars,
            })
        }
        Some("sliding") => {
            let window = parse_usize_field(parts.next(), 600)?;
            let stride = parse_usize_field(parts.next(), 300)?;
            Ok(ChunkingStrategy::Sliding { window, stride })
        }
        Some(other) => Err(OreoError::Config(format!(
            "unknown chunking strategy: {other}"
        ))),
        None => Err(OreoError::Config("empty chunking strategy".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_from<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        let owned: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |name| {
            owned
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        }
    }

    #[test]
    fn defaults_should_validate() {
        assert!(OreoConfig::default().validate().is_ok());
    }

    #[test]
    fn defaults_should_cover_primary_languages() {
        let config = OreoConfig::default();
        assert_eq!(
            config.languages,
            vec![Language::En, Language::Hi, Language::Ta]
        );
        assert_eq!(config.candidate_top, 20);
        assert_eq!(config.final_top, 5);
    }

    #[test]
    fn from_source_should_parse_language_list_and_pools() {
        let config = OreoConfig::from_source(source_from(&[
            ("VOX_OREO_LANGUAGES", "en, ta"),
            ("VOX_OREO_CANDIDATE_TOP", "40"),
            ("VOX_OREO_FINAL_TOP", "10"),
            ("VOX_OREO_RRF_K", "30"),
            ("VOX_OREO_EMBEDDING_DIM", "128"),
        ]))
        .expect("config parses");
        assert_eq!(config.languages, vec![Language::En, Language::Ta]);
        assert_eq!(config.candidate_top, 40);
        assert_eq!(config.final_top, 10);
        assert_eq!(config.rrf_k, 30);
        assert_eq!(config.embedding_dim, 128);
    }

    #[test]
    fn from_source_should_reject_unknown_languages() {
        let err = OreoConfig::from_source(source_from(&[("VOX_OREO_LANGUAGES", "xx")]))
            .expect_err("unsupported language rejected");
        assert!(matches!(err, OreoError::Config(_)));
    }

    #[test]
    fn from_source_should_parse_chunking_variants() {
        let fixed = OreoConfig::from_source(source_from(&[("VOX_OREO_CHUNKING", "fixed:512:64")]))
            .expect("fixed");
        assert_eq!(
            fixed.chunking,
            ChunkingStrategy::Fixed {
                size: 512,
                overlap: 64
            }
        );
        let sliding =
            OreoConfig::from_source(source_from(&[("VOX_OREO_CHUNKING", "sliding:400:100")]))
                .expect("sliding");
        assert_eq!(
            sliding.chunking,
            ChunkingStrategy::Sliding {
                window: 400,
                stride: 100
            }
        );
    }

    #[test]
    fn validate_should_require_qdrant_url_for_qdrant_kind() {
        let config = OreoConfig {
            vector_store: VectorStoreKind::Qdrant,
            qdrant_url: String::new(),
            ..OreoConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(OreoError::Config(message)) if message.contains("qdrant_url")
        ));
    }

    #[test]
    fn validate_should_reject_final_top_above_candidate_pool() {
        let config = OreoConfig {
            candidate_top: 5,
            final_top: 10,
            ..OreoConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
