# VOX — Evidence-Aware Multilingual Voice RAG (Rust)

Low-latency multilingual voice RAG for Indian languages: spoken question → STT →
hybrid retrieval (dense + BM25 via `POST /v1/retrieve`) → grounding check →
grounded answer, with explicit refusal when evidence is insufficient.

## Status: Phase 1 scaffold

Working end-to-end text pipeline (`/v1/query`) against a deterministic mock
retrieval backend. Not yet implemented: STT ingestion (`/v1/voice/query`
returns 501), real LLM provider (extractive stub in place), real retrieval
service, off-topic guard, latency benchmarks. No benchmark numbers are claimed.

## Workspace layout

| Crate | Responsibility |
|---|---|
| `vox-core` | Domain types: `Language`, retrieve/answer contracts, `StageTimings`, validation errors |
| `vox-retrieval` | `RetrievalClient` trait + `MockRetrievalClient` + HTTP client (timeouts, bounded retries) |
| `vox-pipeline` | Orchestration: language → query analysis → retrieval → grounding → generation → guardrail, per-stage timings |
| `vox-api` | Axum server: `/health`, `/metrics`, `/v1/query`, `/v1/voice/query` |

Integration boundary with the retrieval service is exclusively
`POST /v1/retrieve`; nothing here couples to Qdrant/Tantivy internals.

## Quickstart

```sh
cargo run -p vox-api            # serves on 0.0.0.0:8080 with the mock backend

curl -s localhost:8080/health
curl -s -X POST localhost:8080/v1/query \
  -H 'content-type: application/json' \
  -d '{"query":"What is GST?","language":"hi","top_k":3}'
curl -s localhost:8080/metrics
```

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `VOX_HOST` / `VOX_PORT` | `0.0.0.0` / `8080` | Bind address |
| `VOX_LOG_FORMAT` | `text` | `text` or `json` (level via `RUST_LOG`) |
| `VOX_RETRIEVAL_MODE` | `mock` | `mock` or `http` |
| `VOX_RETRIEVAL_BASE_URL` | — | Required when mode is `http` (OREO service) |
| `VOX_RETRIEVAL_TIMEOUT_MS` | `800` | Per-attempt timeout for `/v1/retrieve` |
| `VOX_RETRIEVAL_MAX_RETRIES` | `2` | Retries on network errors, timeouts, 5xx, 429 only |
| `VOX_RETRIEVAL_BACKOFF_MS` | `50` | Linear backoff base between attempts |
| `VOX_MOCK_DELAY_MS` | `0` | Simulated latency injected by the mock backend |
| `VOX_GROUNDING_MIN_SCORE` | `0.30` | Best-score threshold below which VOX refuses |

## API

- `GET /health` → liveness + uptime.
- `GET /metrics` → Prometheus exposition.
- `POST /v1/query` → `{query, language, top_k}` → grounded answer or refusal,
  always with per-stage `timings_ms`.
- `POST /v1/voice/query` → reserved for voice ingestion; currently `501`.

Errors: `422` invalid input, `502` upstream failure, `504` upstream timeout,
`503` retrieval unavailable, `501` not yet implemented.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
