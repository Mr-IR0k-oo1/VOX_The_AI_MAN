//! OpenAI-compatible chat-completions provider.
//!
//! Speaks the de-facto standard `POST {base_url}/chat/completions` contract,
//! which covers OpenAI and compatible gateways (vLLM, Ollama, Groq, …).
//! The API key always comes from configuration/environment — never hard-coded.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::prompt::build_prompt;
use crate::{LlmError, LlmProvider, LlmRequest, LlmResponse, MAX_RETRIES};

/// Client for any OpenAI-compatible chat-completions endpoint.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleProvider {
    api_key: String,
    model: String,
    base_url: String,
    timeout: Duration,
    http: reqwest::Client,
}

impl OpenAiCompatibleProvider {
    /// Creates a provider targeting `base_url` (trailing slashes trimmed).
    ///
    /// # Errors
    /// Returns [`reqwest::Error`] if the underlying HTTP client cannot be
    /// built (e.g. TLS backend initialization failure).
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder().timeout(timeout).build()?;
        Ok(Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            timeout,
            http,
        })
    }

    fn url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn payload(&self, request: &LlmRequest) -> serde_json::Value {
        let prompt = build_prompt(request);
        json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "temperature": 0.0,
        })
    }

    async fn attempt_once(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let response = self
            .http
            .post(self.url())
            .bearer_auth(&self.api_key)
            .json(&self.payload(request))
            .send()
            .await
            .map_err(map_transport_error(self.timeout))?;

        let status = response.status();
        if !status.is_success() {
            return Err(LlmError::HttpStatus { status });
        }

        // Decode explicitly so a contract-violating body becomes
        // `InvalidResponse` rather than a generic network error.
        let raw = response
            .text()
            .await
            .map_err(map_transport_error(self.timeout))?;
        let body: ChatCompletionsResponse = serde_json::from_str(&raw)
            .map_err(|_| LlmError::InvalidResponse("malformed json body"))?;

        let content = body
            .choices
            .first()
            .and_then(|choice| choice.message.as_ref())
            .and_then(|message| message.content.as_deref())
            .filter(|content| !content.trim().is_empty())
            .ok_or(LlmError::InvalidResponse("missing answer content"))?;

        Ok(LlmResponse {
            answer: content.to_owned(),
            model: body.model,
        })
    }
}

fn map_transport_error(timeout: Duration) -> impl Fn(reqwest::Error) -> LlmError + Send {
    move |err| {
        if err.is_timeout() {
            LlmError::Timeout {
                timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            }
        } else {
            LlmError::Network(err)
        }
    }
}

/// Subset of the chat-completions response that VOX consumes.
#[derive(Debug, Deserialize)]
struct ChatCompletionsResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    #[serde(default)]
    message: Option<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    fn name(&self) -> &'static str {
        "openai-compatible"
    }

    async fn generate(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let mut attempt: u32 = 0;
        loop {
            match self.attempt_once(request).await {
                Ok(response) => return Ok(response),
                Err(err) if err.is_retryable() && attempt < MAX_RETRIES => {
                    attempt += 1;
                    tracing::warn!(attempt, error = %err, "retryable llm failure, retrying");
                }
                Err(err) => return Err(err),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use serde_json::{json, Value};
    use vox_types::{Language, RetrievedDocument};

    use super::*;

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

    async fn slow_handler() -> impl IntoResponse {
        tokio::time::sleep(Duration::from_millis(500)).await;
        (StatusCode::OK, "{}")
    }

    async fn flaky_handler(State(calls): State<Arc<AtomicU32>>) -> Response {
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response();
        }
        axum::Json(json!({
            "model": "test-model",
            "choices": [{"message": {"role": "assistant", "content": "GST is a tax."}}]
        }))
        .into_response()
    }

    async fn unauthorized_handler() -> impl IntoResponse {
        (StatusCode::UNAUTHORIZED, "bad key")
    }

    async fn malformed_handler() -> impl IntoResponse {
        (StatusCode::OK, "{\"choices\": \"not-a-list\"}")
    }

    fn provider(base_url: String, timeout: Duration) -> OpenAiCompatibleProvider {
        OpenAiCompatibleProvider::new("test-key", "test-model", base_url, timeout)
            .expect("client builds")
    }

    fn request() -> LlmRequest {
        LlmRequest::new(
            "What is GST?",
            Language::Hi,
            vec![RetrievedDocument {
                id: "d1".to_owned(),
                text: "GST is an indirect tax.".to_owned(),
                score: 0.9,
                rank: 1,
                metadata: Value::Null,
            }],
        )
    }

    #[tokio::test]
    async fn generate_should_parse_a_well_formed_completion() {
        let router = axum::Router::new().route(
            "/chat/completions",
            post(|| async {
                axum::Json(json!({
                    "model": "test-model",
                    "choices": [
                        {"message": {"role": "assistant", "content": "जीएसटी एक कर है।"}}
                    ]
                }))
            }),
        );
        let response = provider(spawn(router).await, Duration::from_secs(5))
            .generate(&request())
            .await
            .expect("generate");

        assert_eq!(response.answer, "जीएसटी एक कर है।");
        assert_eq!(response.model.as_deref(), Some("test-model"));
    }

    #[tokio::test]
    async fn payload_should_embed_the_grounded_prompt() {
        let captured = Arc::new(std::sync::Mutex::new(Value::Null));
        let state = Arc::clone(&captured);
        let router = axum::Router::new().route(
            "/chat/completions",
            post(move |body: axum::Json<Value>| {
                *state.lock().expect("lock") = body.0;
                async { axum::Json(json!({"choices": [{"message": {"content": "ok"}}]})) }
            }),
        );
        let _ = provider(spawn(router).await, Duration::from_secs(5))
            .generate(&request())
            .await
            .expect("generate");

        let body = captured.lock().expect("lock").clone();
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["temperature"], 0.0);
        let messages = body["messages"].as_array().expect("messages");
        assert_eq!(messages[0]["role"], "system");
        assert!(
            messages[0]["content"]
                .as_str()
                .expect("system")
                .contains("Hindi (hi)"),
            "system prompt must carry the language instruction"
        );
        assert!(
            messages[1]["content"]
                .as_str()
                .expect("user")
                .contains("GST is an indirect tax."),
            "user prompt must carry the evidence"
        );
    }

    #[tokio::test]
    async fn generate_should_time_out_when_the_server_is_slow() {
        let router = axum::Router::new().route("/chat/completions", post(slow_handler));
        let err = provider(spawn(router).await, Duration::from_millis(50))
            .generate(&request())
            .await
            .expect_err("must time out");

        assert!(matches!(err, LlmError::Timeout { timeout_ms: 50 }));
    }

    #[tokio::test]
    async fn generate_should_retry_once_on_transient_server_errors() {
        let calls = Arc::new(AtomicU32::new(0));
        let router = axum::Router::new()
            .route("/chat/completions", post(flaky_handler))
            .with_state(calls.clone());

        let response = provider(spawn(router).await, Duration::from_secs(5))
            .generate(&request())
            .await
            .expect("second attempt must succeed");

        assert_eq!(response.answer, "GST is a tax.");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "exactly one retry");
    }

    #[tokio::test]
    async fn generate_should_not_retry_client_errors() {
        let router = axum::Router::new().route("/chat/completions", post(unauthorized_handler));
        let err = provider(spawn(router).await, Duration::from_secs(5))
            .generate(&request())
            .await
            .expect_err("401 must fail");

        assert!(matches!(
            err,
            LlmError::HttpStatus {
                status: StatusCode::UNAUTHORIZED
            }
        ));
    }

    #[tokio::test]
    async fn generate_should_reject_contract_violating_bodies() {
        let router = axum::Router::new().route("/chat/completions", post(malformed_handler));
        let err = provider(spawn(router).await, Duration::from_secs(5))
            .generate(&request())
            .await
            .expect_err("malformed body must fail");

        assert!(matches!(err, LlmError::InvalidResponse(_)));
    }

    #[tokio::test]
    async fn generate_should_reject_empty_answers() {
        let router = axum::Router::new().route(
            "/chat/completions",
            post(|| async { axum::Json(json!({"choices": [{"message": {"content": "   "}}]})) }),
        );
        let err = provider(spawn(router).await, Duration::from_secs(5))
            .generate(&request())
            .await
            .expect_err("empty answer must fail");

        assert!(matches!(err, LlmError::InvalidResponse(_)));
    }

    #[test]
    fn retryability_should_follow_safe_transient_rules() {
        assert!(!LlmError::Backend("x".to_owned()).is_retryable());
        assert!(!LlmError::InvalidResponse("bad").is_retryable());
        assert!(!LlmError::HttpStatus {
            status: StatusCode::BAD_REQUEST
        }
        .is_retryable());
        assert!(LlmError::HttpStatus {
            status: StatusCode::BAD_GATEWAY
        }
        .is_retryable());
        assert!(LlmError::HttpStatus {
            status: StatusCode::TOO_MANY_REQUESTS
        }
        .is_retryable());
        assert!(LlmError::Timeout { timeout_ms: 1 }.is_retryable());
    }
}
