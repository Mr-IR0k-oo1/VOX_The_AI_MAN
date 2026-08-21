//! Integration tests exercising the exact router production serves.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use vox_api::{build_router, build_state, Config};

fn app() -> axum::Router {
    let config = Config::from_source(|_| None).expect("default config");
    let state = std::sync::Arc::new(build_state(&config).expect("state"));
    build_router(state)
}

async fn call_json(app: axum::Router, method: &str, uri: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let parsed: Value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("json body")
    };
    (status, parsed)
}

#[tokio::test]
async fn health_should_report_ok() {
    let (status, body) = call_json(app(), "GET", "/health", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["service"], "vox");
}

#[tokio::test]
async fn query_should_return_language_query_evidence_and_metrics() {
    let payload = json!({
        "query": "What is artificial intelligence?",
        "language": "en",
        "top_k": 3
    });
    let (status, body) = call_json(app(), "POST", "/v1/query", payload).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body["request_id"]
        .as_str()
        .expect("request_id")
        .starts_with("req-"));
    assert_eq!(body["language"], "en");
    assert_eq!(body["query"]["query"], "What is artificial intelligence?");
    assert_eq!(
        body["query"]["normalized_text"],
        "What is artificial intelligence?"
    );
    assert_eq!(body["query"]["intent"], "definition");
    assert_eq!(body["evidence"].as_array().expect("evidence").len(), 3);
    assert!(
        body["metrics"]["total"].as_f64().is_some(),
        "total latency expected"
    );
    assert!(
        body["metrics"]["retrieval"].as_f64().is_some(),
        "retrieval latency expected"
    );
    assert!(body.get("answer").is_none(), "no LLM in Phase 1");
}

#[tokio::test]
async fn query_should_auto_detect_tamil_without_hint() {
    let payload = json!({ "query": "குங்குமப்பூ என்றால் என்ன?" });
    let (status, body) = call_json(app(), "POST", "/v1/query", payload).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["language"], "ta");
    assert_eq!(body["query"]["intent"], "definition");
}

#[tokio::test]
async fn query_should_reject_blank_input_with_422() {
    let payload = json!({ "query": "   " });
    let (status, body) = call_json(app(), "POST", "/v1/query", payload).await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"], "invalid_request");
}

#[tokio::test]
async fn voice_query_should_return_transcript_and_evidence() {
    // "hello" as a minimal WAV-ish payload; the mock recognizer ignores it.
    let audio_base64 = "aGVsbG8=";
    let payload = json!({
        "audio_base64": audio_base64,
        "format": "wav",
        "top_k": 2
    });
    let (status, body) = call_json(app(), "POST", "/v1/voice/query", payload).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body["request_id"]
        .as_str()
        .expect("request_id")
        .starts_with("req-"));
    assert!(!body["transcript"]["text"]
        .as_str()
        .expect("text")
        .is_empty());
    assert!(body["language"].is_string());
    assert_eq!(body["evidence"].as_array().expect("evidence").len(), 2);
    assert!(body["metrics"]["stt"].as_f64().is_some());
    assert!(body["metrics"]["total"].as_f64().is_some());
}

#[tokio::test]
async fn voice_query_should_reject_oversized_audio_with_422() {
    // ~11.4 MB decoded, above the 10 MB ingest limit but below the router's
    // 16 MB body limit so the domain error (not the transport) answers.
    let oversized = "QUJD".repeat(3_800_000);
    let payload = json!({ "audio_base64": oversized, "format": "wav" });
    let (status, body) = call_json(app(), "POST", "/v1/voice/query", payload).await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"], "invalid_audio");
}

#[tokio::test]
async fn retrieve_should_serve_the_retrieval_boundary_directly() {
    let payload = json!({ "query": "what is gst", "language": "en", "top_k": 2 });
    let (status, body) = call_json(app(), "POST", "/v1/retrieve", payload).await;

    assert_eq!(status, StatusCode::OK);
    let documents = body["documents"].as_array().expect("documents");
    assert_eq!(documents.len(), 2);
    assert!(documents[0]["score"].as_f64().is_some());
}
