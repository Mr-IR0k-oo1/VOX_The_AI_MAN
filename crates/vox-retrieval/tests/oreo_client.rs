//! Contract and resilience tests for [`OREORetrievalClient`] against
//! ephemeral HTTP doubles of the frozen `/v1/retrieve` contract.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Json;
use serde_json::{json, Value};
use vox_retrieval::{OREORetrievalClient, RetrievalClient, RetrievalError, RetryPolicy};
use vox_types::{Language, Query};

/// Binds an ephemeral server and returns its base URL.
async fn spawn(router: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    format!("http://{addr}")
}

fn client(base_url: String) -> OREORetrievalClient {
    OREORetrievalClient::new(base_url, Duration::from_secs(5), RetryPolicy::default())
        .expect("client builds")
}

fn query() -> Query {
    Query::new("what is gst", Some(Language::En), 3)
}

#[tokio::test]
async fn retrieve_should_send_query_language_and_top_k() {
    let captured = Arc::new(std::sync::Mutex::new(None::<Value>));
    let state = Arc::clone(&captured);
    let base_url = spawn(axum::Router::new().route(
        "/v1/retrieve",
        post(move |Json(body): Json<Value>| async move {
            *state.lock().expect("lock") = Some(body);
            Json(json!({ "documents": [] }))
        }),
    ))
    .await;

    client(base_url)
        .retrieve(query())
        .await
        .expect("retrieve succeeds");

    let body = captured
        .lock()
        .expect("lock")
        .clone()
        .expect("request body captured");
    assert_eq!(body["query"], "what is gst");
    assert_eq!(body["language"], "en");
    assert_eq!(body["top_k"], 3);
}

#[tokio::test]
async fn retrieve_should_parse_documents_scores_and_metadata() {
    let base_url = spawn(axum::Router::new().route(
        "/v1/retrieve",
        post(|| async {
            Json(json!({
                "documents": [
                    {
                        "id": "oreo-0001",
                        "text": "GST is an indirect tax used in India.",
                        "score": 0.93,
                        "rank": 1,
                        "metadata": { "source": "oreo", "collection": "tax-kb" }
                    },
                    {
                        "id": "oreo-0002",
                        "text": "GST replaced cascading taxes.",
                        "score": 0.81,
                        "rank": 2,
                        "metadata": null
                    }
                ]
            }))
        }),
    ))
    .await;

    let response = client(base_url).retrieve(query()).await.expect("parse");

    assert_eq!(response.documents.len(), 2);
    assert_eq!(response.documents[0].id, "oreo-0001");
    assert_eq!(response.documents[0].score, 0.93);
    assert_eq!(response.documents[0].rank, 1);
    assert_eq!(response.documents[0].metadata["source"], "oreo");
    assert_eq!(response.documents[1].metadata, Value::Null);
}

#[tokio::test]
async fn timeout_should_surface_as_a_timeout_error() {
    async fn slow() -> impl IntoResponse {
        tokio::time::sleep(Duration::from_millis(500)).await;
        Json(json!({ "documents": [] }))
    }

    let base_url = spawn(axum::Router::new().route("/v1/retrieve", post(slow))).await;

    let slow_client = OREORetrievalClient::new(
        base_url,
        Duration::from_millis(50),
        RetryPolicy {
            max_retries: 0,
            backoff: Duration::ZERO,
        },
    )
    .expect("client builds");

    let err = slow_client
        .retrieve(query())
        .await
        .expect_err("must time out");
    assert!(matches!(err, RetrievalError::Timeout { timeout_ms: 50 }));
}

#[tokio::test]
async fn transient_500_should_be_retried_once_by_default() {
    async fn flaky(State(calls): State<Arc<AtomicU32>>, Json(_query): Json<Query>) -> Response {
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response();
        }
        Json(json!({
            "documents": [
                { "id": "d1", "text": "GST is a tax.", "score": 0.9, "rank": 1, "metadata": null }
            ]
        }))
        .into_response()
    }

    let calls = Arc::new(AtomicU32::new(0));
    let base_url = spawn(
        axum::Router::new()
            .route("/v1/retrieve", post(flaky))
            .with_state(Arc::clone(&calls)),
    )
    .await;

    let response = client(base_url).retrieve(query()).await.expect("recovers");
    assert_eq!(response.documents[0].id, "d1");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "exactly one safe retry after a transient 500"
    );
}

#[tokio::test]
async fn persistent_500_should_fail_after_the_single_retry() {
    async fn always_500(State(calls): State<Arc<AtomicU32>>) -> Response {
        calls.fetch_add(1, Ordering::SeqCst);
        (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response()
    }

    let calls = Arc::new(AtomicU32::new(0));
    let base_url = spawn(
        axum::Router::new()
            .route("/v1/retrieve", post(always_500))
            .with_state(Arc::clone(&calls)),
    )
    .await;

    let err = client(base_url).retrieve(query()).await.expect_err("fails");
    assert!(matches!(
        err,
        RetrievalError::HttpStatus {
            status: StatusCode::INTERNAL_SERVER_ERROR
        }
    ));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "initial attempt plus one retry, nothing more"
    );
}

#[tokio::test]
async fn client_errors_should_not_be_retried() {
    async fn not_found(State(calls): State<Arc<AtomicU32>>) -> Response {
        calls.fetch_add(1, Ordering::SeqCst);
        (StatusCode::NOT_FOUND, "nope").into_response()
    }

    let calls = Arc::new(AtomicU32::new(0));
    let base_url = spawn(
        axum::Router::new()
            .route("/v1/retrieve", post(not_found))
            .with_state(Arc::clone(&calls)),
    )
    .await;

    let err = client(base_url).retrieve(query()).await.expect_err("fails");
    assert!(matches!(
        err,
        RetrievalError::HttpStatus {
            status: StatusCode::NOT_FOUND
        }
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1, "4xx is never retried");
}

#[tokio::test]
async fn malformed_success_body_should_map_to_invalid_response_without_retry() {
    async fn malformed(State(calls): State<Arc<AtomicU32>>) -> Response {
        calls.fetch_add(1, Ordering::SeqCst);
        (StatusCode::OK, "{\"documents\": \"not-a-list\"}").into_response()
    }

    let calls = Arc::new(AtomicU32::new(0));
    let base_url = spawn(
        axum::Router::new()
            .route("/v1/retrieve", post(malformed))
            .with_state(Arc::clone(&calls)),
    )
    .await;

    let err = client(base_url).retrieve(query()).await.expect_err("fails");
    assert!(matches!(
        err,
        RetrievalError::InvalidResponse("malformed json body")
    ));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "contract violations are not transient and must not be retried"
    );
}

#[tokio::test]
async fn invalid_queries_should_fail_validation_before_any_upstream_call() {
    async fn ok(State(calls): State<Arc<AtomicU32>>) -> Json<Value> {
        calls.fetch_add(1, Ordering::SeqCst);
        Json(json!({ "documents": [] }))
    }

    let calls = Arc::new(AtomicU32::new(0));
    let base_url = spawn(
        axum::Router::new()
            .route("/v1/retrieve", post(ok))
            .with_state(Arc::clone(&calls)),
    )
    .await;

    let err = client(base_url)
        .retrieve(Query::new("   ", None, 3))
        .await
        .expect_err("blank query must be rejected locally");
    assert!(matches!(err, RetrievalError::Validation(_)));
    assert_eq!(calls.load(Ordering::SeqCst), 0, "nothing sent upstream");
}
