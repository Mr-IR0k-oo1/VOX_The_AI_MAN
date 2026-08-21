//! HTTP contract tests for the frozen v1 endpoints, exercised through the
//! real router with the embedded OREO engine as the retrieval backend.
//!
//! These lock the wire shape promised in Phase 0/Phase 2: `POST /v1/retrieve`
//! returns `documents` with `id`, `text`, `score`, `rank`, and full
//! provenance `metadata`, plus per-stage `timings_ms`.

use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use vox_api::{build_router, AppState, Config};
use vox_core::{PipelineConfig, VoiceRagPipeline};
use vox_llm::ExtractiveProvider;
use vox_retrieval::{EmbeddedOreoClient, RetrievalClient};
use vox_stt::StubSpeechRecognizer;

/// Builds an [`AppState`] whose retrieval backend is an embedded OREO engine
/// pre-indexed with the bundled multilingual sample corpus.
async fn oreo_app(tantivy_dir: &Path) -> axum::Router {
    let dir_string = tantivy_dir.to_string_lossy().into_owned();
    let config = Config::from_source(move |name| match name {
        "VOX_RETRIEVAL_MODE" => Some("oreo".to_owned()),
        "VOX_OREO_TANTIVY_DIR" => Some(dir_string.clone()),
        _ => None,
    })
    .expect("oreo test config");

    let oreo_config = config
        .retrieval
        .oreo
        .clone()
        .expect("oreo settings present in oreo mode");
    let engine = Arc::new(vox_oreo::OreoEngine::new(oreo_config).expect("engine"));
    engine
        .index_documents(vox_oreo::ingest::sample_corpus())
        .await
        .expect("index sample corpus");

    let retrieval: Arc<dyn RetrievalClient> = Arc::new(EmbeddedOreoClient::new(engine));
    let pipeline = Arc::new(VoiceRagPipeline::new(
        Arc::clone(&retrieval),
        Arc::new(ExtractiveProvider),
        PipelineConfig {
            grounding_min_score: config.pipeline.grounding_min_score,
        },
    ));

    build_router(Arc::new(AppState {
        pipeline,
        retrieval,
        stt: Arc::new(StubSpeechRecognizer),
        started_at: std::time::Instant::now(),
    }))
}

async fn post_json(app: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let parsed = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("json body")
    };
    (status, parsed)
}

#[tokio::test]
async fn retrieve_should_return_ranked_documents_with_full_metadata() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = oreo_app(dir.path()).await;

    let (status, body) = post_json(
        app,
        "/v1/retrieve",
        json!({
            "query": "What is GST?",
            "language": "en",
            "top_k": 5,
            "intent": "unknown"
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let documents = body["documents"].as_array().expect("documents array");
    assert!(
        !documents.is_empty(),
        "indexed corpus should produce results"
    );
    assert!(body["timings_ms"].is_object(), "stage timings expected");

    for document in documents {
        assert!(document["id"].is_string(), "id missing: {document}");
        assert!(document["text"].is_string(), "text missing: {document}");
        assert!(document["score"].is_number(), "score missing: {document}");
        assert!(document["rank"].is_number(), "rank missing: {document}");
        let metadata = &document["metadata"];
        assert!(
            metadata["document_id"].is_string(),
            "metadata.document_id missing: {document}"
        );
        assert!(
            metadata["chunk_id"].is_string(),
            "metadata.chunk_id missing: {document}"
        );
        assert!(
            metadata["language"].is_string(),
            "metadata.language missing: {document}"
        );
        assert!(
            metadata["chunking_strategy"].is_string(),
            "metadata.chunking_strategy missing: {document}"
        );
        assert!(
            metadata["source"].is_string(),
            "metadata.source missing: {document}"
        );
        assert_eq!(
            metadata["chunk_id"].as_str(),
            document["id"].as_str(),
            "chunk_id must equal the wire id"
        );
    }

    let ranks: Vec<u32> = documents
        .iter()
        .map(|d| d["rank"].as_u64().expect("rank") as u32)
        .collect();
    let expected: Vec<u32> = (1..=ranks.len() as u32).collect();
    assert_eq!(ranks, expected, "ranks must be 1..n in order");

    let scores: Vec<f64> = documents
        .iter()
        .map(|d| d["score"].as_f64().expect("score"))
        .collect();
    for pair in scores.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "scores must be non-increasing: {scores:?}"
        );
    }

    assert!(documents.len() <= 5, "final_top caps results at 5");
}

#[tokio::test]
async fn retrieve_should_surface_multilingual_evidence() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = oreo_app(dir.path()).await;

    let (status, body) = post_json(
        app,
        "/v1/retrieve",
        json!({
            "query": "जीएसटी क्या है",
            "language": "hi",
            "top_k": 5,
            "intent": "unknown"
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let documents = body["documents"].as_array().expect("documents array");
    assert!(!documents.is_empty(), "hindi query should match evidence");
    assert!(
        documents.iter().any(|d| d["metadata"]["language"] == "hi"),
        "at least one hindi document expected: {documents:?}"
    );
}

#[tokio::test]
async fn health_should_report_ok() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = oreo_app(dir.path()).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let body: Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(body["status"], "ok");
    assert_eq!(body["service"], "vox");
}

#[tokio::test]
async fn blank_query_should_fail_validation_with_422() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = oreo_app(dir.path()).await;

    let (status, body) = post_json(
        app,
        "/v1/retrieve",
        json!({ "query": "   ", "language": "en", "top_k": 5, "intent": "unknown" }),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"], "invalid_request");
}
