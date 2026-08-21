# VOX — Evidence-Aware Multilingual Voice RAG (Rust)

Low-latency multilingual voice RAG for Indian languages: spoken question → STT →
hybrid retrieval (dense + BM25 via `POST /v1/retrieve`) → grounding check →
grounded answer, with explicit refusal when evidence is insufficient.

## Status: Phase 0 — architecture freeze

Repository and interface boundaries are frozen before feature development.
Shared domain types, backend traits (`RetrievalClient`, `SpeechRecognizer`,
`LlmProvider`), the five v1 endpoints, env configuration, and tracing are in
place. The text pipeline (`/v1/query`, `/v1/retrieve`) runs end-to-end against
a deterministic mock retrieval backend and an extractive answer stub.

Not yet implemented (deliberately): real STT (`/v1/voice/query` returns 501
via the stub recognizer), real LLM provider, off-topic guard, real retrieval
service, audio transcoding, benchmark scenarios. No benchmark numbers are
claimed.

## Workspace layout

| Crate | Responsibility |
|---|---|
| `vox-types` | Shared domain types: `Query`, `RetrievedDocument`, `Transcript`, `Language`, verdicts, `LatencyMetrics`, validation errors |
| `vox-core` | Pipeline orchestration: language → query analysis → retrieval → grounding → generation → guardrail, per-stage timings |
| `vox-api` | Axum server exposing the frozen v1 endpoints; config, logging, metrics |
| `vox-stt` | `SpeechRecognizer` trait + stub (real engine lands later) |
| `vox-retrieval` | `RetrievalClient` trait + mock backend + HTTP client for OREO (timeouts, bounded retries) |
| `vox-ingest` | Voice ingestion boundary: base64 decode, size caps, container sniffing |
| `vox-grounding` | Evidence-sufficiency assessment → `Answerability` |
| `vox-llm` | `LlmProvider` trait + extractive stub provider |
| `vox-guard` | Guardrail decisions over evidence verdicts and generated answers |
| `vox-bench` | Latency percentile utilities; scenario harnesses land later |

Dependency direction: everything depends on `vox-types`; adapters
(`vox-stt`, `vox-retrieval`, `vox-llm`) define their own traits;
`vox-core` sequences the stages; `vox-api` wires concrete backends.

Integration boundary with the retrieval service is exclusively
`POST /v1/retrieve`; nothing here couples to Qdrant/Tantivy internals.

## Quickstart

```sh
cargo run -p vox-api            # serves on 0.0.0.0:8080 with the mock backend

curl -s localhost:8080/health
curl -s -X POST localhost:8080/v1/query \
  -H 'content-type: application/json' \
  -d '{"query":"What is GST?","language":"hi","top_k":3}'
curl -s -X POST localhost:8080/v1/retrieve \
  -H 'content-type: application/json' \
  -d '{"query":"What is GST?","language":"hi","top_k":3}'
curl -s localhost:8080/metrics
```

## Frozen API (Phase 0)

| Endpoint | Contract |
|---|---|
| `GET /health` | Liveness + uptime |
| `GET /metrics` | Prometheus exposition |
| `POST /v1/retrieve` | `Query` → `RetrievalResponse` (`{documents: [...]}`) |
| `POST /v1/query` | `Query` → grounded `AnswerResponse` or refusal, always with `timings_ms` |
| `POST /v1/voice/query` | `VoiceRequest` → `VoiceResponse`; currently `501` until a real STT engine lands |

Errors: `422` invalid input/audio, `502` upstream failure, `504` upstream
timeout, `503` retrieval unavailable, `501` not yet implemented.

## Configuration (environment)

See [.env.example](.env.example).

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

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
