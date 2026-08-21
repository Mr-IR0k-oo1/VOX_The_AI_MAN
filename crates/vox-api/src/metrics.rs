//! Prometheus metrics setup.

use std::sync::OnceLock;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Installs the global Prometheus recorder exactly once and returns its
/// render handle.
///
/// Idempotent: repeated calls (e.g. from tests) return the original handle
/// instead of re-installing a recorder.
#[must_use]
pub fn init_metrics() -> &'static PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        // A second global recorder cannot be installed; losing that race
        // only matters if some other component already installed one, in
        // which case its handle is the live one anyway.
        let _ = metrics::set_global_recorder(recorder);
        handle
    })
}
