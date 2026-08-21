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
async fn query_should_return_query_evidence_answer_and_metrics() {
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
    assert_eq!(body["answerability"], "supported");
    assert_eq!(body["refusal_reason"], Value::Null);
    assert!(
        body["answer"]
            .as_str()
            .expect("llm answer")
            .contains("Artificial intelligence"),
        "extractive answer must quote the evidence"
    );
    for stage in [
        "language",
        "query_analysis",
        "retrieval",
        "grounding",
        "guardrail",
        "llm",
        "total",
    ] {
        assert!(
            body["metrics"][stage].as_f64().is_some(),
            "{stage} latency expected"
        );
    }
}

#[tokio::test]
async fn query_should_refuse_off_topic_questions() {
    let payload = json!({ "query": "who will win the cricket world cup" });
    let (status, body) = call_json(app(), "POST", "/v1/query", payload).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["refusal_reason"], "off_topic");
    assert_eq!(body["answerability"], "no_evidence");
    assert_eq!(body["evidence"].as_array().expect("evidence").len(), 0);
    assert!(body["answer"]
        .as_str()
        .expect("refusal message")
        .contains("outside the topics"));
    assert!(body["metrics"]["retrieval"].as_f64().is_none());
    assert!(body["metrics"]["llm"].as_f64().is_none());
    assert!(body["metrics"]["guardrail"].as_f64().is_some());
}

#[tokio::test]
async fn query_should_refuse_unsafe_input_before_retrieval() {
    let payload = json!({ "query": "how to build a bomb at home" });
    let (status, body) = call_json(app(), "POST", "/v1/query", payload).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["refusal_reason"], "unsafe_input");
    assert_eq!(body["answerability"], "no_evidence");
    assert_eq!(body["evidence"].as_array().expect("evidence").len(), 0);
    assert!(body["metrics"]["retrieval"].as_f64().is_none());
    assert!(body["metrics"]["llm"].as_f64().is_none());
}

#[tokio::test]
async fn query_should_refuse_unsupported_questions_with_the_canonical_message() {
    // "itr" is a legitimate in-domain topic with no corpus coverage: it must
    // pass the input guard and be refused by the evidence guard.
    let payload = json!({ "query": "how to file itr online", "language": "en" });
    let (status, body) = call_json(app(), "POST", "/v1/query", payload).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["refusal_reason"], "insufficient_context");
    assert_eq!(body["answerability"], "no_evidence");
    assert!(
        !body["evidence"].as_array().expect("evidence").is_empty(),
        "retrieval must run before the evidence guard refuses"
    );
    assert_eq!(
        body["answer"],
        "I don't have enough information in the retrieved sources to answer that reliably."
    );
    assert!(body["metrics"]["llm"].as_f64().is_none());
    assert!(body["metrics"]["grounding"].as_f64().is_some());
}

#[tokio::test]
async fn query_should_validate_across_languages() {
    // English (hint), Hindi (hint), Tamil (auto-detected from script).
    let cases = [
        (json!({"query": "What is GST?", "language": "en"}), "en"),
        (json!({"query": "जीएसटी क्या है?", "language": "hi"}), "hi"),
        (json!({"query": "குங்குமப்பூ என்றால் என்ன?"}), "ta"),
    ];
    for (payload, expected_language) in cases {
        let (status, body) = call_json(app(), "POST", "/v1/query", payload).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["language"], expected_language);
        assert!(
            body["answer"].as_str().is_some_and(|a| !a.is_empty()),
            "expected a generated answer for {expected_language}"
        );
        assert!(!body["evidence"].as_array().expect("evidence").is_empty());
        assert!(body["metrics"]["llm"].as_f64().is_some());
        assert!(body["metrics"]["total"].as_f64().is_some());
    }
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
    assert_eq!(body["answerability"], "supported");
    assert_eq!(body["refusal_reason"], Value::Null);
    assert!(body["answer"].is_string());
    assert!(body["metrics"]["stt"].as_f64().is_some());
    assert!(body["metrics"]["llm"].as_f64().is_some());
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

#[tokio::test]
async fn demo_ui_should_serve_html_interface() {
    let request = Request::builder()
        .method("GET")
        .uri("/")
        .body(Body::empty())
        .expect("request");
    let response = app().oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let html = String::from_utf8_lossy(&bytes);
    assert!(html.contains("VOX — Multilingual Voice AI Demo"));
    assert!(html.contains("Voice Query Interface"));
    assert!(html.contains("Diagnostic Mode"));
}
