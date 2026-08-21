//! Embedding pipeline: text → dense vectors.
//!
//! [`Embedder`] is the substitution boundary; a neural multilingual encoder
//! (e.g. a MuRIL/fastembed ONNX model) can replace the default without
//! touching any other stage. The offline default, [`HashedEmbedder`], maps
//! word unigrams and character trigrams into a fixed-dimension vector via
//! signed feature hashing and L2 normalization. It is deterministic,
//! allocation-light, script-agnostic (works identically for Latin,
//! Devanagari, and Tamil), and gives lexical-similarity semantics that pair
//! well with BM25 in hybrid retrieval.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::error::OreoError;

/// Converts texts into fixed-length embedding vectors.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Backend name used in logs.
    fn name(&self) -> &'static str;

    /// Dimensionality of produced vectors.
    fn dim(&self) -> usize;

    /// Embeds a batch of texts; output order matches input order.
    ///
    /// # Errors
    /// Returns [`OreoError`] if the backend fails (network backends may fail
    /// at runtime; the default local embedder cannot).
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, OreoError>;
}

/// FNV-1a 64-bit hash of a byte string.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Deterministic hashed embedder over word unigrams + character trigrams.
#[derive(Debug, Clone)]
pub struct HashedEmbedder {
    dim: usize,
}

impl HashedEmbedder {
    /// Creates an embedder producing `dim`-dimensional vectors.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Lowercased word tokens of the text (Unicode-alphanumeric runs).
    #[must_use]
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            if ch.is_alphanumeric() {
                current.extend(ch.to_lowercase());
            } else if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    /// Character trigrams across the whole text (with boundary padding),
    /// giving sub-word robustness for Indic morphology.
    #[must_use]
    pub fn char_trigrams(text: &str) -> Vec<String> {
        let chars: Vec<char> = text
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .flat_map(|ch| ch.to_lowercase())
            .collect();
        let mut trigrams = Vec::new();
        if chars.len() < 3 {
            return trigrams;
        }
        for window in chars.windows(3) {
            trigrams.push(window.iter().collect());
        }
        trigrams
    }

    fn feature_index(&self, feature: &str) -> Option<(usize, f32)> {
        if self.dim == 0 {
            return None;
        }
        let hash = fnv1a(feature.as_bytes());
        let index = usize::try_from(hash % self.dim as u64).ok()?;
        // A second independent bit decides the sign, approximating a random
        // projection and reducing collision bias.
        let sign = if (hash >> 63) & 1 == 1 { 1.0 } else { -1.0 };
        Some((index, sign))
    }

    fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut weights = vec![0.0f32; self.dim];
        let mut features: HashMap<String, f32> = HashMap::new();
        for token in Self::tokenize(text) {
            *features.entry(token).or_insert(0.0) += 1.0;
        }
        for trigram in Self::char_trigrams(text) {
            *features.entry(trigram).or_insert(0.0) += 0.5;
        }
        for (feature, weight) in &features {
            if let Some((index, sign)) = self.feature_index(feature) {
                weights[index] += sign * weight;
            }
        }
        let norm = weights.iter().map(|w| w * w).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for weight in &mut weights {
                *weight /= norm;
            }
        }
        weights
    }
}

#[async_trait]
impl Embedder for HashedEmbedder {
    fn name(&self) -> &'static str {
        "hashed"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, OreoError> {
        Ok(texts.iter().map(|text| self.embed_one(text)).collect())
    }
}

/// Cosine similarity between two L2-normalized vectors (their dot product).
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn embed(embedder: &HashedEmbedder, text: &str) -> Vec<f32> {
        embedder
            .embed(&[text.to_owned()])
            .await
            .expect("embed")
            .remove(0)
    }

    #[tokio::test]
    async fn vectors_should_be_l2_normalized() {
        let embedder = HashedEmbedder::new(256);
        let vector = embed(&embedder, "Goods and Services Tax explained").await;
        assert_eq!(vector.len(), 256);
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
    }

    #[tokio::test]
    async fn identical_texts_should_embed_identically() {
        let embedder = HashedEmbedder::new(128);
        let first = embed(&embedder, "जीएसटी एक अप्रत्यक्ष कर है").await;
        let second = embed(&embedder, "जीएसटी एक अप्रत्यक्ष कर है").await;
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn similar_texts_should_score_higher_than_unrelated() {
        let embedder = HashedEmbedder::new(256);
        let query = embed(&embedder, "What is GST?").await;
        let related = embed(&embedder, "GST is an indirect tax introduced in India").await;
        let unrelated = embed(&embedder, "Chennai metro water new connection apply online").await;
        assert!(
            dot(&query, &related) > dot(&query, &unrelated),
            "related {} should beat unrelated {}",
            dot(&query, &related),
            dot(&query, &unrelated)
        );
    }

    #[test]
    fn tokenizer_should_lowercase_and_split_on_punctuation() {
        assert_eq!(
            HashedEmbedder::tokenize("Hello, GST-World!"),
            vec!["hello".to_owned(), "gst".to_owned(), "world".to_owned()]
        );
    }

    #[test]
    fn trigrams_should_cover_indic_scripts() {
        let trigrams = HashedEmbedder::char_trigrams("जीएसटी");
        assert!(!trigrams.is_empty());
        assert!(trigrams.iter().all(|t| t.chars().count() == 3));
    }

    #[tokio::test]
    async fn batch_order_should_match_input_order() {
        let embedder = HashedEmbedder::new(64);
        let vectors = embedder
            .embed(&["alpha beta".to_owned(), "gamma delta".to_owned()])
            .await
            .expect("batch");
        assert_eq!(vectors.len(), 2);
        assert_ne!(vectors[0], vectors[1]);
    }
}
