//! Client boundary for the VOX retrieval service (`POST /v1/retrieve`).
//!
//! The pipeline depends only on [`RetrievalClient`]; the concrete backend is
//! selected by configuration — [`mock::MockRetrievalClient`] until the real
//! retrieval API is proven stable, then [`http::HttpRetrievalClient`].

pub mod error;
pub mod http;
pub mod mock;

use async_trait::async_trait;
use vox_core::{RetrieveRequest, RetrieveResponse};

pub use crate::error::RetrievalError;
pub use crate::http::{HttpRetrievalClient, RetryPolicy};
pub use crate::mock::MockRetrievalClient;

/// Substitution boundary for retrieval backends.
///
/// Implementations must be safe to share across threads and must validate the
/// request before sending it upstream.
#[async_trait]
pub trait RetrievalClient: Send + Sync {
    /// Backend name used in logs and metrics.
    fn name(&self) -> &'static str;

    /// Executes hybrid retrieval for `request`.
    ///
    /// # Errors
    /// Returns [`RetrievalError`] on validation failure, network failure,
    /// timeout, non-success upstream status, or a contract-violating body.
    async fn retrieve(
        &self,
        request: RetrieveRequest,
    ) -> Result<RetrieveResponse, RetrievalError>;
}
