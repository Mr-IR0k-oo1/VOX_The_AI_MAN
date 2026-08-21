# syntax=docker/dockerfile:1

# ------------------------------------------------------------------------------
# Stage 1: Build Stage (Debian Bookworm with Rust)
# ------------------------------------------------------------------------------
FROM rust:1.81-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifests
COPY Cargo.toml Cargo.lock ./

# Copy all crates in the workspace
COPY crates/ crates/

# Build optimized release binaries
RUN cargo build --release -p vox-api -p vox-oreo

# ------------------------------------------------------------------------------
# Stage 2: Runtime Image (Minimal Debian Bookworm Slim)
# ------------------------------------------------------------------------------
FROM debian:bookworm-slim AS runner

# Install essential runtime utilities (ca-certificates for TLS, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and directories
RUN groupadd -g 10001 vox && \
    useradd -u 10001 -g vox -m -s /bin/bash vox && \
    mkdir -p /data/tantivy /app/data && \
    chown -R vox:vox /data /app

# Copy binaries from builder
COPY --from=builder /app/target/release/vox-api /usr/local/bin/vox-api
COPY --from=builder /app/target/release/vox-oreo /usr/local/bin/vox-oreo

# Copy bundled sample corpus
COPY crates/vox-oreo/data/sample-docs.jsonl /app/data/sample-docs.jsonl
RUN chown -R vox:vox /app/data

WORKDIR /app

# Set default production environment variables
ENV VOX_HOST=0.0.0.0 \
    VOX_PORT=8080 \
    RUST_LOG=info,vox=debug \
    VOX_LOG_FORMAT=text \
    VOX_RETRIEVAL_MODE=oreo \
    VOX_OREO_TANTIVY_DIR=/data/tantivy

# Health check probe against /health
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

USER vox

CMD ["/usr/local/bin/vox-api"]
