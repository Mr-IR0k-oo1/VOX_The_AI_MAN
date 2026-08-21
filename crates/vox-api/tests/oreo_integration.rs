//! End-to-end integration tests: the complete VOX pipeline driven through a
//! real HTTP retrieval backend speaking the frozen OREO `/v1/retrieve`
//! contract.
//!
//! The OREO stand-in is an ephemeral server that implements exactly the
//! frozen wire contract (`Query` in, `{documents: [...]}` out) backed by the
//! deterministic sample corpus — no OREO internals are imported, mirroring
//! the production boundary.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use vox_api::{build_router, build_state, Config};
use vox_retrieval::{MockRetrievalClient, RetrievalClient};
use vox_types::{Query, RetrievalResponse};

/// Binds an ephemeral OREO stand-in serving the frozen retrieval contract
/// over the deterministic sample corpus.
async fn spawn_oreo_double() -> String {
    async fn retrieve(Json(query): Json<Query>) -> Json<RetrievalResponse> {
        let client = MockRetrievalClient::default();
        Json(client.retrieve(query).await.expect("mock retrieve"))
    }

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let app = Router::new().route("/v1/retrieve", post(retrieve));
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    format!("http://{addr}")
}

/// OREO stand-in whose first call fails with a transient 500.
async fn spawn_flaky_oreo_double() -> (String, Arc<AtomicU32>) {
    async fn retrieve(State(calls): State<Arc<AtomicU32>>, Json(query): Json<Query>) -> Response {
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response();
        }
        let client = MockRetrievalClient::default();
        Json(client.retrieve(query).await.expect("mock retrieve")).into_response()
    }

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let app = Router::new()
        .route("/v1/retrieve", post(retrieve))
        .with_state(Arc::new(AtomicU32::new(0)));
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    (format!("http://{addr}"), Arc::new(AtomicU32::new(0)))
}

/// OREO stand-in that answers slower than any reasonable timeout.
async fn spawn_slow_oreo_double() -> String {
    async fn retrieve() -> impl IntoResponse {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        Json(json!({ "documents": [] }))
    }

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let app = Router::new().route("/v1/retrieve", post(retrieve));
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    format!("http://{addr}")
}

fn config_with(pairs: &[(&str, String)]) -> Config {
    let map: HashMap<&str, String> = pairs.iter().map(|(k, v)| (*k, v.clone())).collect();
    Config::from_source(move |name| map.get(name).cloned()).expect("config")
}

fn oreo_app(base_url: &str) -> axum::Router {
    let config = config_with(&[
        ("VOX_RETRIEVAL_MODE", "http".to_owned()),
        ("VOX_RETRIEVAL_BASE_URL", base_url.to_owned()),
    ]);
    let state = Arc::new(build_state(&config).expect("state"));
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
async fn query_should_answer_supported_questions_across_languages_over_oreo() {
    let base_url = spawn_oreo_double().await;
    let cases = [
        (
            json!({"query": "What is artificial intelligence?", "language": "en"}),
            "en",
            "Artificial intelligence",
        ),
        (
            json!({"query": "जीएसटी क्या है?", "language": "hi"}),
            "hi",
            "जीएसटी",
        ),
        (json!({"query": "குங்குமப்பூ என்றால் என்ன?"}), "ta", "குங்குமப்பூ"),
    ];
    for (payload, language, fragment) in cases {
        let (status, body) = call_json(oreo_app(&base_url), "POST", "/v1/query", payload).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["language"], language);
        assert_eq!(body["answerability"], "supported");
        assert_eq!(body["refusal_reason"], Value::Null);
        assert!(
            body["answer"].as_str().expect("answer").contains(fragment),
            "expected answer quoting the corpus for {language}"
        );
        assert!(body["evidence"].as_array().expect("evidence").len() >= 2);
        assert!(body["metrics"]["llm"].as_f64().is_some());
        assert!(body["metrics"]["guardrail"].as_f64().is_some());
    }
}

#[tokio::test]
async fn voice_query_should_traverse_the_full_pipeline_over_oreo() {
    // Definition of done: voice → STT → language → query → OREO → grounding
    // → LLM → guardrails → answer, without manual intervention.
    let base_url = spawn_oreo_double().await;
    let payload = json!({
        "audio_base64": "aGVsbG8=",
        "format": "wav",
        "top_k": 3
    });
    let (status, body) = call_json(oreo_app(&base_url), "POST", "/v1/voice/query", payload).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["transcript"]["text"], "What is GST in India?");
    assert_eq!(body["language"], "en");
    assert_eq!(body["query"]["intent"], "definition");
    assert_eq!(body["answerability"], "supported");
    assert_eq!(body["refusal_reason"], Value::Null);
    assert!(
        body["answer"]
            .as_str()
            .expect("answer")
            .contains("Goods and Services Tax"),
        "the voice answer must be grounded in OREO evidence"
    );
    assert!(!body["evidence"].as_array().expect("evidence").is_empty());
    for stage in ["stt", "retrieval", "grounding", "guardrail", "llm", "total"] {
        assert!(
            body["metrics"][stage].as_f64().is_some(),
            "{stage} latency expected"
        );
    }
}

#[tokio::test]
async fn unsupported_query_should_refuse_with_insufficient_context_over_oreo() {
    let base_url = spawn_oreo_double().await;
    let payload = json!({"query": "how to file itr online", "language": "en"});
    let (status, body) = call_json(oreo_app(&base_url), "POST", "/v1/query", payload).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["refusal_reason"], "insufficient_context");
    assert_eq!(body["answerability"], "no_evidence");
    assert_eq!(
        body["answer"],
        "I don't have enough information in the retrieved sources to answer that reliably."
    );
    assert!(body["metrics"]["llm"].as_f64().is_none());
}

#[tokio::test]
async fn off_topic_query_should_refuse_via_evidence_guard_over_oreo() {
    // Over HTTP the topic vocabulary is unknown, so the input guard's
    // off-topic check is disabled by design; the question reaches OREO,
    // retrieves nothing relevant, and the evidence guard refuses.
    let base_url = spawn_oreo_double().await;
    let payload = json!({"query": "who will win the cricket world cup"});
    let (status, body) = call_json(oreo_app(&base_url), "POST", "/v1/query", payload).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["refusal_reason"], "insufficient_context");
    assert_eq!(body["answerability"], "no_evidence");
    assert!(!body["evidence"].as_array().expect("evidence").is_empty());
    assert!(body["metrics"]["llm"].as_f64().is_none());
}

#[tokio::test]
async fn transient_oreo_failure_should_retry_once_and_recover() {
    let (base_url, _calls) = spawn_flaky_oreo_double().await;
    let payload = json!({"query": "What is GST?", "language": "en"});
    let (status, body) = call_json(oreo_app(&base_url), "POST", "/v1/query", payload).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["answerability"], "supported");
    assert!(body["metrics"]["retrieval"].as_f64().is_some());
}

#[tokio::test]
async fn oreo_timeout_should_map_to_504() {
    let base_url = spawn_slow_oreo_double().await;
    let config = config_with(&[
        ("VOX_RETRIEVAL_MODE", "http".to_owned()),
        ("VOX_RETRIEVAL_BASE_URL", base_url),
        ("VOX_RETRIEVAL_TIMEOUT_MS", "50".to_owned()),
        ("VOX_RETRIEVAL_MAX_RETRIES", "0".to_owned()),
    ]);
    let state = Arc::new(build_state(&config).expect("state"));
    let payload = json!({"query": "What is GST?", "language": "en"});
    let (status, body) = call_json(build_router(state), "POST", "/v1/query", payload).await;

    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(body["error"], "retrieval_timeout");
}
