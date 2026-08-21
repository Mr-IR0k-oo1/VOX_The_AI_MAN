//! Process-wide tracing initialization.

use tracing_subscriber::EnvFilter;

use crate::config::LogFormat;

/// Installs the global tracing subscriber.
///
/// Level/filter come from `RUST_LOG` (default `info`). Idempotent: a second
/// call is a no-op, which keeps tests that exercise bootstrap paths safe.
pub fn init_tracing(format: LogFormat) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let result = match format {
        LogFormat::Text => tracing_subscriber::fmt().with_env_filter(filter).try_init(),
        LogFormat::Json => tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .try_init(),
    };
    if let Err(err) = result {
        tracing::debug!(error = %err, "tracing subscriber already installed");
    }
}
