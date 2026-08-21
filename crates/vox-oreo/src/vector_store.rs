//! Dense vector storage: the [`VectorStore`] boundary plus an in-process
//! cosine store and a Qdrant REST client.
//!
//! Qdrant point ids are derived deterministically from chunk ids (FNV-1a),
//! so re-indexing the same corpus is idempotent and never duplicates points.
//! Payloads are the full [`ChunkRecord`] metadata, preserving provenance
//! end-to-end.

use std::sync::RwLock;

use async_trait::async_trait;
use serde::Deserialize;

use crate::chunk::ChunkRecord;
use crate::embed::dot;
use crate::error::OreoError;

/// One dense-search hit.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorHit {
    /// Matched chunk with full metadata.
    pub chunk: ChunkRecord,
    /// Similarity score (cosine; higher is better).
    pub score: f32,
}

/// Dense vector index boundary.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Backend name used in logs.
    fn name(&self) -> &'static str;

    /// Ensures the store exists and matches `dim`, then upserts `chunks`.
    ///
    /// # Errors
    /// Returns [`OreoError`] on transport or server failures.
    async fn upsert(&self, chunks: &[(ChunkRecord, Vec<f32>)]) -> Result<(), OreoError>;

    /// Nearest-neighbour search over stored vectors.
    ///
    /// # Errors
    /// Returns [`OreoError`] on transport or server failures.
    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>, OreoError>;

    /// Number of vectors currently stored.
    ///
    /// # Errors
    /// Returns [`OreoError`] on transport or server failures.
    async fn count(&self) -> Result<usize, OreoError>;
}

/// In-process brute-force cosine store (tests, demos, small corpora).
#[derive(Debug, Default)]
pub struct MemoryVectorStore {
    entries: RwLock<Vec<(Vec<f32>, ChunkRecord)>>,
}

impl MemoryVectorStore {
    /// Creates an empty store.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl VectorStore for MemoryVectorStore {
    fn name(&self) -> &'static str {
        "memory"
    }

    async fn upsert(&self, chunks: &[(ChunkRecord, Vec<f32>)]) -> Result<(), OreoError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| OreoError::Config("vector store lock poisoned".into()))?;
        for (chunk, vector) in chunks {
            if let Some(slot) = entries
                .iter_mut()
                .find(|(_, existing)| existing.chunk_id == chunk.chunk_id)
            {
                slot.0 = vector.clone();
                slot.1 = chunk.clone();
            } else {
                entries.push((vector.clone(), chunk.clone()));
            }
        }
        Ok(())
    }

    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>, OreoError> {
        let entries = self
            .entries
            .read()
            .map_err(|_| OreoError::Config("vector store lock poisoned".into()))?;
        let mut hits: Vec<VectorHit> = entries
            .iter()
            .map(|(vector, chunk)| VectorHit {
                chunk: chunk.clone(),
                score: dot(query, vector),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    async fn count(&self) -> Result<usize, OreoError> {
        Ok(self
            .entries
            .read()
            .map_err(|_| OreoError::Config("vector store lock poisoned".into()))?
            .len())
    }
}

/// Qdrant collection info returned by the REST API.
#[derive(Debug, Deserialize)]
struct QdrantSearchResponse {
    result: Option<Vec<QdrantScoredPoint>>,
}

#[derive(Debug, Deserialize)]
struct QdrantScoredPoint {
    score: f32,
    #[serde(default)]
    payload: Option<serde_json::Value>,
}

/// Qdrant access over its REST API (`/collections/...`).
#[derive(Debug, Clone)]
pub struct QdrantStore {
    base_url: String,
    collection: String,
    dim: usize,
    timeout: std::time::Duration,
    http: reqwest::Client,
}

impl QdrantStore {
    /// Creates a store bound to `base_url`/`collection` with vectors of
    /// dimensionality `dim`.
    ///
    /// # Errors
    /// Returns [`OreoError::Config`] when the HTTP client cannot be built.
    pub fn new(
        base_url: &str,
        collection: &str,
        dim: usize,
        timeout: std::time::Duration,
    ) -> Result<Self, OreoError> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|err| OreoError::Config(format!("qdrant http client: {err}")))?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            collection: collection.to_owned(),
            dim,
            timeout,
            http,
        })
    }

    /// Lightweight connectivity probe used by `oreo verify`.
    ///
    /// # Errors
    /// Returns [`OreoError`] when the server cannot be reached or answers
    /// with a non-success status.
    pub async fn ping(&self) -> Result<(), OreoError> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);
        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|err| OreoError::VectorStoreNetwork(err.to_string()))?;
        if !response.status().is_success() {
            return Err(OreoError::VectorStore {
                status: response.status().as_u16(),
                body: "collection check failed".to_owned(),
            });
        }
        Ok(())
    }

    async fn ensure_collection(&self) -> Result<(), OreoError> {
        let url = format!("{}/collections/{}/exists", self.base_url, self.collection);
        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|err| OreoError::VectorStoreNetwork(err.to_string()))?;
        let exists = response
            .json::<QdrantExistsResponse>()
            .await
            .ok()
            .and_then(|body| body.result)
            .is_some_and(|result| result.exists);
        if exists {
            return Ok(());
        }
        let url = format!("{}/collections/{}", self.base_url, self.collection);
        let body = serde_json::json!({
            "vectors": { "size": self.dim, "distance": "Cosine" },
        });
        let response = self
            .http
            .put(url)
            .json(&body)
            .send()
            .await
            .map_err(|err| OreoError::VectorStoreNetwork(err.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(OreoError::VectorStore {
                status: status.as_u16(),
                body: truncate(&text),
            });
        }
        Ok(())
    }

    fn point_id(chunk_id: &str) -> u64 {
        crate::embed::fnv1a(chunk_id.as_bytes())
    }
}

#[derive(Debug, Deserialize)]
struct QdrantExistsResponse {
    result: Option<QdrantExists>,
}

#[derive(Debug, Deserialize)]
struct QdrantExists {
    exists: bool,
}

/// Request bodies for the Qdrant points API.
mod wire {
    use serde::Serialize;

    #[derive(Debug, Serialize)]
    pub struct Point {
        pub id: u64,
        pub vector: Vec<f32>,
        pub payload: serde_json::Value,
    }

    #[derive(Debug, Serialize)]
    pub struct UpsertRequest {
        pub points: Vec<Point>,
    }

    #[derive(Debug, Serialize)]
    pub struct SearchRequest {
        pub vector: Vec<f32>,
        pub limit: usize,
        #[serde(rename = "with_payload")]
        pub with_payload: bool,
    }
}

fn truncate(text: &str) -> String {
    text.chars().take(200).collect()
}

#[async_trait]
impl VectorStore for QdrantStore {
    fn name(&self) -> &'static str {
        "qdrant"
    }

    async fn upsert(&self, chunks: &[(ChunkRecord, Vec<f32>)]) -> Result<(), OreoError> {
        self.ensure_collection().await?;
        let points: Vec<wire::Point> = chunks
            .iter()
            .map(|(chunk, vector)| wire::Point {
                id: Self::point_id(&chunk.chunk_id),
                vector: vector.clone(),
                payload: serde_json::to_value(chunk).unwrap_or(serde_json::Value::Null),
            })
            .collect();
        let url = format!(
            "{}/collections/{}/points?wait=true",
            self.base_url, self.collection
        );
        let response = self
            .http
            .put(url)
            .json(&wire::UpsertRequest { points })
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|err| OreoError::VectorStoreNetwork(err.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(OreoError::VectorStore {
                status: status.as_u16(),
                body: truncate(&text),
            });
        }
        Ok(())
    }

    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>, OreoError> {
        let url = format!(
            "{}/collections/{}/points/search",
            self.base_url, self.collection
        );
        let response = self
            .http
            .post(url)
            .json(&wire::SearchRequest {
                vector: query.to_vec(),
                limit: top_k,
                with_payload: true,
            })
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|err| OreoError::VectorStoreNetwork(err.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(OreoError::VectorStore {
                status: status.as_u16(),
                body: truncate(&text),
            });
        }
        let body: QdrantSearchResponse = response
            .json()
            .await
            .map_err(|err| OreoError::VectorStoreNetwork(err.to_string()))?;
        let hits = body
            .result
            .unwrap_or_default()
            .into_iter()
            .filter_map(|point| {
                let chunk: ChunkRecord = serde_json::from_value(point.payload?).ok()?;
                Some(VectorHit {
                    chunk,
                    score: point.score,
                })
            })
            .collect();
        Ok(hits)
    }

    async fn count(&self) -> Result<usize, OreoError> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);
        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|err| OreoError::VectorStoreNetwork(err.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(OreoError::VectorStore {
                status: status.as_u16(),
                body: "count failed".to_owned(),
            });
        }
        #[derive(Deserialize)]
        struct Info {
            result: Option<CollectionInfo>,
        }
        #[derive(Deserialize)]
        struct CollectionInfo {
            #[serde(default)]
            points_count: Option<u64>,
        }
        let body: Info = response
            .json()
            .await
            .map_err(|err| OreoError::VectorStoreNetwork(err.to_string()))?;
        Ok(body
            .result
            .and_then(|info| info.points_count)
            .and_then(|count| usize::try_from(count).ok())
            .unwrap_or(0))
    }
}

/// Helper mirroring the internal FNV hashing for stable point ids.
///
/// Public so ingestion tools can predict point ids without duplicating the
/// hash function.
#[must_use]
pub fn qdrant_point_id(chunk_id: &str) -> u64 {
    QdrantStore::point_id(chunk_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_types::Language;

    fn chunk(id: &str) -> ChunkRecord {
        ChunkRecord {
            document_id: id.to_owned(),
            chunk_id: format!("{id}#0000"),
            chunk_index: 0,
            text: format!("text of {id}"),
            language: Language::En,
            chunking_strategy: "fixed".to_owned(),
            source: "test".to_owned(),
        }
    }

    #[tokio::test]
    async fn memory_store_should_upsert_search_and_count() {
        let store = MemoryVectorStore::new();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        store
            .upsert(&[(chunk("a"), a.clone()), (chunk("b"), b)])
            .await
            .expect("upsert");
        assert_eq!(store.count().await.expect("count"), 2);

        let hits = store.search(&a, 2).await.expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk.document_id, "a");
        assert!(hits[0].score > hits[1].score);
    }

    #[tokio::test]
    async fn memory_store_should_dedupe_by_chunk_id() {
        let store = MemoryVectorStore::new();
        store
            .upsert(&[(chunk("a"), vec![1.0, 0.0])])
            .await
            .expect("first");
        store
            .upsert(&[(chunk("a"), vec![0.0, 1.0])])
            .await
            .expect("second");
        assert_eq!(store.count().await.expect("count"), 1);
    }

    #[test]
    fn point_ids_should_be_stable_across_calls() {
        assert_eq!(qdrant_point_id("doc#0000"), qdrant_point_id("doc#0000"));
        assert_ne!(qdrant_point_id("doc#0000"), qdrant_point_id("doc#0001"));
    }
}
