//! Client for Sarvam AI's speech-to-text service (`POST /speech-to-text`).
//!
//! Contract (as of the Saarika/Saaras API):
//! - header `api-subscription-key: <key>`
//! - multipart form: `file` (audio bytes), `model` (e.g. `saarika:v2.5`)
//! - 200 response JSON: `{ "transcript": "...", "language_code": "ta-IN",
//!   "language_probability": 0.98, ... }`
//!
//! The only coupling is this HTTP contract; the pipeline talks to
//! [`SpeechRecognizer`] exclusively.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use vox_types::{AudioFormat, Language, Transcript};

use crate::{SpeechRecognizer, SttError};

/// One retry is allowed for transient failures.
const MAX_RETRIES: u32 = 1;

/// Recognizer backed by Sarvam AI's synchronous STT endpoint.
#[derive(Debug, Clone)]
pub struct SarvamRecognizer {
    api_key: String,
    model: String,
    base_url: String,
    timeout: Duration,
    http: reqwest::Client,
}

impl SarvamRecognizer {
    /// Creates a recognizer targeting `base_url` (trailing slashes trimmed).
    ///
    /// The API key must be supplied by the caller; it is read from the
    /// environment at configuration time and never hard-coded.
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
        format!("{}/speech-to-text", self.base_url)
    }

    async fn attempt_once(
        &self,
        audio: &[u8],
        format: AudioFormat,
    ) -> Result<Transcript, SttError> {
        let form = build_form(audio, format, &self.model)?;

        let response = self
            .http
            .post(self.url())
            .header("api-subscription-key", &self.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(map_transport_error(self.timeout))?;

        let status = response.status();
        if !status.is_success() {
            return Err(SttError::HttpStatus { status });
        }

        // Decode explicitly so a contract-violating body becomes
        // `InvalidResponse` rather than a generic network error.
        let raw = response
            .text()
            .await
            .map_err(map_transport_error(self.timeout))?;
        let body: SarvamSttResponse = serde_json::from_str(&raw)
            .map_err(|_| SttError::InvalidResponse("malformed json body"))?;

        let text = body
            .transcript
            .ok_or(SttError::InvalidResponse("missing transcript"))?;
        if text.trim().is_empty() {
            return Err(SttError::InvalidResponse("empty transcript"));
        }

        Ok(Transcript {
            text,
            detected_language: body.language_code.as_deref().and_then(parse_language),
            confidence: body.language_probability,
        })
    }
}

/// Multipart payload for Sarvam's `/speech-to-text`.
fn build_form(
    audio: &[u8],
    format: AudioFormat,
    model: &str,
) -> Result<reqwest::multipart::Form, SttError> {
    let mime = match format {
        AudioFormat::Wav => "audio/wav",
        AudioFormat::Mp3 => "audio/mpeg",
        AudioFormat::Flac => "audio/flac",
    };
    let part = reqwest::multipart::Part::bytes(audio.to_vec())
        .file_name(format!("audio.{}", format.as_str()))
        .mime_str(mime)
        .map_err(|err| SttError::Backend(err.to_string()))?;
    Ok(reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", model.to_owned()))
}

/// Subset of Sarvam's STT response that VOX consumes.
///
/// Unknown fields are ignored; `transcript` is the only required field.
#[derive(Debug, Deserialize)]
struct SarvamSttResponse {
    transcript: Option<String>,
    #[serde(default)]
    language_code: Option<String>,
    #[serde(default)]
    language_probability: Option<f32>,
}

/// Maps a BCP-47 code such as `ta-IN` onto a [`Language`].
///
/// Unrecognized codes yield `None`; downstream language resolution handles
/// the fallback.
fn parse_language(code: &str) -> Option<Language> {
    code.split(['-', '_'])
        .next()
        .and_then(|prefix| prefix.parse::<Language>().ok())
}

fn map_transport_error(timeout: Duration) -> impl Fn(reqwest::Error) -> SttError + Send {
    move |err| {
        if err.is_timeout() {
            SttError::Timeout {
                timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            }
        } else {
            SttError::Network(err)
        }
    }
}

#[async_trait]
impl SpeechRecognizer for SarvamRecognizer {
    fn name(&self) -> &'static str {
        "sarvam"
    }

    async fn transcribe(
        &self,
        audio: &[u8],
        format: AudioFormat,
        _language: Option<Language>,
    ) -> Result<Transcript, SttError> {
        let mut attempt: u32 = 0;
        loop {
            match self.attempt_once(audio, format).await {
                Ok(transcript) => return Ok(transcript),
                Err(err) if err.is_retryable() && attempt < MAX_RETRIES => {
                    attempt += 1;
                    tracing::warn!(attempt, error = %err, "retryable stt failure, retrying");
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
    use axum::Json;
    use serde_json::json;

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
        Json(json!({
            "request_id": "srv-1",
            "transcript": "vanakkam",
            "language_code": "ta-IN",
            "language_probability": 0.97,
        }))
        .into_response()
    }

    async fn bad_request_handler() -> impl IntoResponse {
        (StatusCode::UNAUTHORIZED, "bad key")
    }

    async fn malformed_handler() -> impl IntoResponse {
        (StatusCode::OK, "{\"transcript\": 42}")
    }

    fn recognizer(base_url: String, timeout: Duration) -> SarvamRecognizer {
        SarvamRecognizer::new("test-key", "saarika:v2.5", base_url, timeout).expect("client builds")
    }

    #[tokio::test]
    async fn transcribe_should_parse_transcript_language_and_confidence() {
        let router = axum::Router::new().route(
            "/speech-to-text",
            post(|| async {
                Json(json!({
                    "transcript": "வணக்கம்",
                    "language_code": "ta-IN",
                    "language_probability": 0.93,
                }))
            }),
        );
        let base_url = spawn(router).await;

        let transcript = recognizer(base_url, Duration::from_secs(5))
            .transcribe(b"audio", AudioFormat::Wav, None)
            .await
            .expect("transcribe");

        assert_eq!(transcript.text, "வணக்கம்");
        assert_eq!(transcript.detected_language, Some(Language::Ta));
        assert_eq!(transcript.confidence, Some(0.93));
    }

    #[tokio::test]
    async fn transcribe_should_time_out_when_the_server_is_slow() {
        let router = axum::Router::new().route("/speech-to-text", post(slow_handler));
        let base_url = spawn(router).await;

        let err = recognizer(base_url, Duration::from_millis(50))
            .transcribe(b"audio", AudioFormat::Wav, None)
            .await
            .expect_err("must time out");

        assert!(matches!(err, SttError::Timeout { timeout_ms: 50 }));
    }

    #[tokio::test]
    async fn transcribe_should_retry_transient_failures_once() {
        let calls = Arc::new(AtomicU32::new(0));
        let router = axum::Router::new()
            .route("/speech-to-text", post(flaky_handler))
            .with_state(Arc::clone(&calls));
        let base_url = spawn(router).await;

        let transcript = recognizer(base_url, Duration::from_secs(5))
            .transcribe(b"audio", AudioFormat::Wav, None)
            .await
            .expect("second attempt should succeed");

        assert_eq!(transcript.text, "vanakkam");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "exactly one retry");
    }

    #[tokio::test]
    async fn transcribe_should_not_retry_client_errors() {
        let calls = Arc::new(AtomicU32::new(0));
        let counter = Arc::clone(&calls);
        let router = axum::Router::new().route(
            "/speech-to-text",
            post(move || {
                let calls = Arc::clone(&counter);
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    bad_request_handler().await
                }
            }),
        );
        let base_url = spawn(router).await;

        let err = recognizer(base_url, Duration::from_secs(5))
            .transcribe(b"audio", AudioFormat::Wav, None)
            .await
            .expect_err("401 must fail");

        assert!(matches!(
            err,
            SttError::HttpStatus {
                status: StatusCode::UNAUTHORIZED
            }
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1, "no retry on 4xx");
    }

    #[tokio::test]
    async fn transcribe_should_reject_contract_violating_bodies() {
        let router = axum::Router::new().route("/speech-to-text", post(malformed_handler));
        let base_url = spawn(router).await;

        let err = recognizer(base_url, Duration::from_secs(5))
            .transcribe(b"audio", AudioFormat::Wav, None)
            .await
            .expect_err("malformed body must fail");

        assert!(matches!(err, SttError::InvalidResponse(_)));
    }

    #[test]
    fn parse_language_should_map_bcp47_codes_and_ignore_unknowns() {
        assert_eq!(parse_language("ta-IN"), Some(Language::Ta));
        assert_eq!(parse_language("hi"), Some(Language::Hi));
        assert_eq!(parse_language("en_US"), Some(Language::En));
        assert_eq!(parse_language("zz-ZZ"), None);
    }
}
