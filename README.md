# VOX — Evidence-Aware Multilingual Voice RAG (Rust)

Low-latency multilingual voice RAG for Indian languages: spoken question → STT →
hybrid retrieval (dense + BM25 via `POST /v1/retrieve`) → grounding check →
grounded answer, with explicit refusal when evidence is insufficient.

## Status: Phase 2 — OREO retrieval foundation

Phase 0 froze the architecture (shared types, backend traits, five v1
endpoints, env config, tracing). Phase 2 adds **OREO** (`vox-oreo`), an
independent multilingual retrieval engine: dataset ingestion → preprocessing
(clean → NFC normalize → script-based language detection) → chunking →
embeddings → Qdrant/Tantivy indexes → hybrid RRF fusion → rerank. The API can
now serve `POST /v1/retrieve` from a real index via the embedded backend
(`VOX_RETRIEVAL_MODE=oreo`), or from the standalone `vox-oreo` service.

Retrieval defaults: en/hi/ta languages, sentence chunking, deterministic
hashed embedder (offline placeholder for a future neural encoder), in-memory
cosine store or Qdrant over REST for dense, Tantivy BM25 for sparse, RRF
(k=60) fusing a top-20 candidate pool down to top-5 after lexical-overlap
reranking. Every response carries per-stage timings (`timings_ms`).

Not yet implemented (deliberately): real STT (`/v1/voice/query` returns 501
via the stub recognizer), real LLM provider, off-topic guard, neural
embeddings/rerankers, audio transcoding, benchmark scenarios.

## Workspace layout

| Crate | Responsibility |
|---|---|
| `vox-types` | Shared domain types: `Query`, `RetrievedDocument`, `Transcript`, `Language`, verdicts, `LatencyMetrics`, validation errors |
| `vox-core` | Pipeline orchestration: language → query analysis → retrieval → grounding → generation → guardrail, per-stage timings |
| `vox-api` | Axum server exposing the frozen v1 endpoints; config, logging, metrics |
| `vox-stt` | `SpeechRecognizer` trait + stub (real engine lands later) |
| `vox-retrieval` | `RetrievalClient` trait + mock backend + embedded OREO backend + HTTP client (timeouts, bounded retries) |
| `vox-oreo` | OREO retrieval engine: ingest, preprocess, chunk, embed, Qdrant/Tantivy, hybrid RRF, rerank; standalone `POST /v1/retrieve` service |
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

### Real retrieval (OREO)

```sh
# Index the bundled multilingual sample corpus (or point --input at a
# JSONL/JSON file or directory; MSMARCO-XI layouts are tolerated).
cargo run -p vox-oreo -- index
cargo run -p vox-oreo -- verify
cargo run -p vox-oreo -- bench --queries 25

# Option A: embed the engine in the API process.
VOX_RETRIEVAL_MODE=oreo cargo run -p vox-api

# Option B: serve retrieval standalone and point the API at it over HTTP.
cargo run -p vox-oreo -- serve --port 8090
VOX_RETRIEVAL_MODE=http VOX_RETRIEVAL_BASE_URL=http://localhost:8090 cargo run -p vox-api
```

Local benchmark on the 20-document sample corpus (debug build, Windows):
p50 ≈ 1.5 ms, p95 ≈ 3.6 ms per hybrid query, dominated by BM25 + rerank.
Dense vectors default to the in-process store; set
`VOX_OREO_VECTOR_STORE=qdrant` (+ `VOX_OREO_QDRANT_URL`) to use Qdrant.

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
| `VOX_RETRIEVAL_MODE` | `mock` | `mock`, `oreo` (embedded engine), or `http` |
| `VOX_RETRIEVAL_BASE_URL` | — | Required when mode is `http` (OREO service) |
| `VOX_RETRIEVAL_TIMEOUT_MS` | `800` | Per-attempt timeout for `/v1/retrieve` |
| `VOX_RETRIEVAL_MAX_RETRIES` | `2` | Retries on network errors, timeouts, 5xx, 429 only |
| `VOX_RETRIEVAL_BACKOFF_MS` | `50` | Linear backoff base between attempts |
| `VOX_MOCK_DELAY_MS` | `0` | Simulated latency injected by the mock backend |
| `VOX_GROUNDING_MIN_SCORE` | `0.30` | Best-score threshold below which VOX refuses |
| `VOX_OREO_LANGUAGES` | `en,hi,ta` | Languages kept by OREO preprocessing |
| `VOX_OREO_CHUNKING` | `sentence:700:80` | `fixed:size:overlap`, `sentence:max:min`, or `sliding:window:stride` |
| `VOX_OREO_EMBEDDING_DIM` | `256` | Embedding dimensionality |
| `VOX_OREO_VECTOR_STORE` | `memory` | `memory` or `qdrant` |
| `VOX_OREO_QDRANT_URL` | `http://localhost:6333` | Qdrant base URL (qdrant mode) |
| `VOX_OREO_COLLECTION` | `vox-chunks` | Qdrant collection name |
| `VOX_OREO_TANTIVY_DIR` | `data/oreo-index` | Tantivy BM25 index directory |
| `VOX_OREO_CANDIDATE_TOP` | `20` | Fused candidate pool size before rerank |
| `VOX_OREO_FINAL_TOP` | `5` | Final results returned after rerank |
| `VOX_OREO_RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `VOX_OREO_RERANKER` | `lexical` | `lexical` or `none` |

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
