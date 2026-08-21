//! HTTP client for the real retrieval service (`POST /v1/retrieve`).

use std::time::Duration;

use async_trait::async_trait;
use vox_types::{Query, RetrievalResponse};

use crate::error::RetrievalError;
use crate::RetrievalClient;

/// Bounded retry behavior for transient upstream failures.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// How many times to retry after a retryable failure (0 = single attempt).
    pub max_retries: u32,
    /// Base delay before the n-th retry is `backoff * n` (linear backoff).
    pub backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            backoff: Duration::from_millis(50),
        }
    }
}

/// Client for OREO's retrieval service.
///
/// The only coupling to the retrieval implementation is this endpoint's JSON
/// contract; Qdrant/Tantivy internals stay behind the service.
#[derive(Debug, Clone)]
pub struct HttpRetrievalClient {
    base_url: String,
    timeout: Duration,
    retry: RetryPolicy,
    http: reqwest::Client,
}

impl HttpRetrievalClient {
    /// Creates a client targeting `base_url` (trailing slashes are trimmed).
    ///
    /// # Errors
    /// Returns [`reqwest::Error`] if the underlying HTTP client cannot be
    /// built (e.g. TLS backend initialization failure).
    pub fn new(
        base_url: impl Into<String>,
        timeout: Duration,
        retry: RetryPolicy,
    ) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder().timeout(timeout).build()?;
        Ok(Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            timeout,
            retry,
            http,
        })
    }

    fn url(&self) -> String {
        format!("{}/v1/retrieve", self.base_url)
    }

    async fn attempt_once(&self, request: &Query) -> Result<RetrievalResponse, RetrievalError> {
        request.validate()?;

        let response = self
            .http
            .post(self.url())
            .json(request)
            .send()
            .await
            .map_err(map_transport_error(self.timeout))?;

        let status = response.status();
        if !status.is_success() {
            return Err(RetrievalError::HttpStatus { status });
        }

        let body: RetrievalResponse = response
            .json()
            .await
            .map_err(map_transport_error(self.timeout))?;

        // A NaN score would silently defeat every threshold comparison
        // downstream (grounding check), so reject it at the boundary.
        if body.documents.iter().any(|doc| !doc.score.is_finite()) {
            return Err(RetrievalError::InvalidResponse("non-finite score"));
        }
        Ok(body)
    }
}

fn map_transport_error(timeout: Duration) -> impl Fn(reqwest::Error) -> RetrievalError + Send {
    move |err| {
        if err.is_timeout() {
            RetrievalError::Timeout {
                timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            }
        } else {
            RetrievalError::Network(err)
        }
    }
}

#[async_trait]
impl RetrievalClient for HttpRetrievalClient {
    fn name(&self) -> &'static str {
        "http"
    }

    async fn retrieve(&self, request: Query) -> Result<RetrievalResponse, RetrievalError> {
        let mut attempt: u32 = 0;
        loop {
            match self.attempt_once(&request).await {
                Ok(response) => return Ok(response),
                Err(err) if err.is_retryable() && attempt < self.retry.max_retries => {
                    attempt += 1;
                    tracing::warn!(attempt, error = %err, "retryable retrieval failure, retrying");
                    tokio::time::sleep(self.retry.backoff * attempt).await;
                }
                Err(err) => return Err(err),
            }
        }
    }
}
