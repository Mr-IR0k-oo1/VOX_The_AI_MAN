//! VOX API binary: loads configuration, installs tracing, serves the router.

use std::sync::Arc;

use vox_api::{build_router, build_state, logging, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    logging::init_tracing(config.log_format);

    let state = Arc::new(build_state(&config)?);
    let app = build_router(state);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "vox api listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Resolves when the process is asked to terminate (Ctrl-C).
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}
