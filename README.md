# VOX — Evidence-Aware Multilingual Voice RAG (Rust)

Low-latency multilingual voice RAG for Indian languages: spoken or typed
question → STT → language detection → query analysis → hybrid retrieval
(dense + BM25 via `POST /v1/retrieve`) → evidence, with per-stage latency
metrics. Grounded answer generation and refusal land in later phases.

## Status: Phase 1 — IR0K-side pipeline foundation

The full pipeline runs end-to-end independently of OREO:

```
voice/text → STT → language detection → query analysis → mock retrieval
           → evidence → latency metrics
```

- **STT**: `MockRecognizer` (deterministic) and `SarvamRecognizer`
  (Sarvam AI `/speech-to-text` via multipart, env-provided API key,
  explicit timeout, one retry on transient failures only).
- **Language detection**: hint → STT-reported → local script fallback
  (Devanagari → Hindi, Tamil script → Tamil, Latin → English; no ML).
- **Query analysis**: conservative normalization (whitespace/punctuation
  cleanup only — no translation) + deterministic intent heuristics
  (`factual`, `definition`, `comparison`, `procedural`, `unknown`) with
  English, Hindi, and Tamil markers.
- **Retrieval**: deterministic `MockRetrievalClient` with a small canned
  corpus. Qdrant/Tantivy/OREO integration is deliberately deferred.
- **Metrics**: `stt`, `language`, `query_analysis`, `retrieval`, `total`
  (ms); stages that did not run are omitted from the JSON.

Not yet implemented (deliberately): LLM answer generation, grounding
check, guardrails, real retrieval service, UI, deployment.

## Workspace layout

| Crate | Responsibility |
|---|---|
| `vox-types` | Shared domain types: `Query`, `QueryIntent`, `Transcript`, `RetrievedDocument`, `Language`, responses, `LatencyMetrics`, validation errors |
| `vox-core` | Pipeline orchestration: STT → language → query analysis → retrieval, per-stage timings |
| `vox-api` | Axum server exposing the frozen v1 endpoints; config, logging, metrics |
| `vox-stt` | `SpeechRecognizer` trait + mock + Sarvam HTTP backend |
| `vox-retrieval` | `RetrievalClient` trait + mock backend + HTTP client for OREO (timeouts, bounded retries) |
| `vox-ingest` | Voice ingestion boundary: base64 decode, size caps, container sniffing |
| `vox-grounding` | Evidence-sufficiency assessment → `Answerability` (later phase) |
| `vox-llm` | `LlmProvider` trait + extractive stub provider (later phase) |
| `vox-guard` | Guardrail decisions over evidence verdicts and generated answers (later phase) |
| `vox-bench` | Latency percentile utilities; scenario harnesses land later |

Dependency direction: everything depends on `vox-types`; adapters
(`vox-stt`, `vox-retrieval`, `vox-llm`) define their own traits;
`vox-core` sequences the stages; `vox-api` wires concrete backends.

Integration boundary with the retrieval service is exclusively
`POST /v1/retrieve`; nothing here couples to Qdrant/Tantivy internals.

## Quickstart

```sh
cargo run -p vox-api            # serves on 0.0.0.0:8080 with mock backends

curl -s localhost:8080/health

# Text flow
curl -s -X POST localhost:8080/v1/query \
  -H 'content-type: application/json' \
  -d '{"query":"What is artificial intelligence?","language":"en"}'

# Voice flow (base64 audio; mock STT returns a fixed transcript)
curl -s -X POST localhost:8080/v1/voice/query \
  -H 'content-type: application/json' \
  -d "{\"audio_base64\":\"$(base64 -w0 sample.wav)\",\"format\":\"wav\"}"

# Direct retrieval boundary
curl -s -X POST localhost:8080/v1/retrieve \
  -H 'content-type: application/json' \
  -d '{"query":"What is GST?","language":"en","top_k":3}'

curl -s localhost:8080/metrics
```

Example response (`POST /v1/query`):

```json
{
  "request_id": "req-68a5f0c2e1a3-0007",
  "language": "en",
  "query": {
    "query": "What is artificial intelligence?",
    "normalized_text": "What is artificial intelligence?",
    "language": "en",
    "intent": "definition",
    "top_k": 5
  },
  "evidence": [
    { "id": "mock-0000", "text": "Artificial intelligence (AI) …", "score": 0.95, "rank": 1, "metadata": {"source":"mock","language":"en"} }
  ],
  "metrics": { "language": 0.001, "query_analysis": 0.002, "retrieval": 0.15, "total": 0.16 }
}
```

## Frozen API (Phase 0)

| Endpoint | Contract |
|---|---|
| `GET /health` | Liveness + uptime |
| `GET /metrics` | Prometheus exposition |
| `POST /v1/retrieve` | `Query` → `RetrievalResponse` (`{documents: [...]}`) |
| `POST /v1/query` | `Query` → `{request_id, language, query, evidence, metrics}` |
| `POST /v1/voice/query` | `VoiceRequest` → `{request_id, transcript, language, query, evidence, metrics}` |

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
| `VOX_STT_MODE` | `mock` | `mock` or `sarvam` |
| `VOX_SARVAM_API_KEY` | — | Required when mode is `sarvam`; never hard-code |
| `VOX_SARVAM_MODEL` | `saarika:v2.5` | Sarvam model identifier |
| `VOX_SARVAM_BASE_URL` | `https://api.sarvam.ai` | Sarvam API base URL |
| `VOX_SARVAM_TIMEOUT_MS` | `3000` | Per-request timeout for the STT call |

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
