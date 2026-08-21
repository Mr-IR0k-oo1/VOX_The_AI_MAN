//! Server-generated request identifiers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Generates a unique, roughly time-ordered request id.
///
/// Format: `req-<12 hex nanos>-<4 hex counter>`. Monotonic enough for log
/// correlation without pulling in a UUID dependency.
#[must_use]
pub fn new_request_id() -> String {
    let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed) & 0xFFFF;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("req-{nanos:012x}-{seq:04x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_should_be_prefixed_and_unique() {
        let first = new_request_id();
        let second = new_request_id();
        assert!(first.starts_with("req-"));
        assert!(second.starts_with("req-"));
        assert_ne!(first, second);
    }
}
