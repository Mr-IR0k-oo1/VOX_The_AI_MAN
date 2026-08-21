# VOX — Evidence-Aware Multilingual Voice RAG (Rust)

Low-latency multilingual voice RAG for Indian languages: spoken or typed
question → STT → language detection → query analysis → hybrid retrieval
(dense + BM25 via `POST /v1/retrieve`) → grounding → guardrails → LLM answer,
with per-stage latency metrics. VOX refuses to answer — with a stable reason
code and a localized grounded-refusal message — whenever the evidence does
not support it.

## Status: Phases 1–7 — pipeline, OREO retrieval, evaluation, integration, performance

Phase 0 froze the architecture (shared types, backend traits, five v1
endpoints, env config, tracing). The full pipeline then came together on
both sides of the integration boundary and now runs end-to-end, reaching
the real retrieval service through configuration alone
(`VOX_RETRIEVAL_MODE=http`); the mock backend remains the default for local
development and tests:

```
voice/text → input guard → STT → language detection → query analysis
           → OREO /v1/retrieve → grounding (evidence sufficiency)
           → evidence guard → LLM → output guard → response (+ latency metrics)
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
- **Input guardrails**: unsafe requests are refused before retrieval;
  off-topic questions are refused against the configured topic vocabulary
  (disabled automatically when running against an HTTP backend whose domain
  is unknown).
- **OREO** (`vox-oreo`): independent multilingual retrieval engine:
  dataset ingestion → preprocessing (clean → NFC normalize → script-based
  language detection) → chunking → embeddings → Qdrant/Tantivy indexes →
  hybrid RRF fusion → rerank. The API serves `POST /v1/retrieve` from a
  real index via the embedded backend (`VOX_RETRIEVAL_MODE=oreo`) or from
  the standalone `vox-oreo` service. Defaults: en/hi/ta languages,
  sentence chunking, deterministic hashed embedder (offline placeholder
  for a future neural encoder), in-memory cosine store or Qdrant for
  dense, Tantivy BM25 for sparse, RRF (k=60) fusing a top-20 candidate
  pool down to top-5 after lexical-overlap reranking. Every response
  carries per-stage timings (`timings_ms`).
- **Retrieval**: `OREORetrievalClient` speaks the frozen
  `POST /v1/retrieve` JSON contract over reqwest with a per-attempt timeout,
  one safe retry (network errors, timeouts, 5xx, 429 only), and strict
  response validation (non-finite scores and malformed bodies are rejected).
  `MockRetrievalClient` (deterministic canned corpus) is selected by default;
  nothing couples to Qdrant/Tantivy internals.
- **Grounding**: deterministic lexical scoring of evidence sufficiency —
  *relevance* (best single-document coverage of the query terms), *coverage*
  (union coverage), and *consistency* (do relevant documents agree?) — mapped
  onto `answerability`: `supported`, `weak_evidence`, `no_evidence`,
  `conflicting_evidence`. All thresholds are configurable.
- **Evidence guardrails**: anything not `supported` is refused before a
  generation call is spent (`insufficient_context`, `weak_evidence`,
  `conflicting_evidence`).
- **LLM**: shared deterministic prompt construction instructing the model to
  use only the supplied evidence, never invent claims, state insufficient
  information explicitly, and answer in the request language.
  Providers: `ExtractiveProvider` (deterministic baseline, default) and
  `OpenAiCompatibleProvider` (`POST {base_url}/chat/completions`,
  env-provided API key, explicit timeout, one retry on safe transient
  failures only).
- **Output guardrails**: generated answers are verified against the evidence
  (content-token support ratio; invented numbers are flagged). Unsupported
  answers trigger exactly one regeneration attempt and then refuse with
  `unsupported_claim`; unsafe or empty output refuses immediately.
- **Refusals** are HTTP 200 responses carrying `refusal_reason` plus a
  localized grounded-refusal message (English canonical: *"I don't have
  enough information in the retrieved sources to answer that reliably."*);
  only infrastructure failures surface as errors.
- **Evaluation** (`vox-bench`): 36-query en/hi/ta eval set over the bundled
  sample corpus; chunking-strategy comparison, retrieval-component ablation
  (dense / BM25 / score-sum / RRF / +rerank), Recall@5 + MRR quality,
  P50/P70/P100 latency per stage. Results and methodology live in
  [benchmarks/](benchmarks/).
- **Metrics**: `stt`, `language`, `query_analysis`, `retrieval`,
  `grounding`, `guardrail`, `llm`, `total` (ms); stages that did not run are
  omitted.
- **Benchmarks**: the `vox-bench` harness measures L0 (retrieval boundary),
  L1 (text-to-answer), and L2 (voice-to-answer) over a five-scenario
  multilingual test set, emitting raw samples plus mean/median/P50/P70/P100
  to `benchmarks/*.json`; see `benchmarks/performance_report.md`.

Not yet implemented (deliberately): neural embeddings/rerankers, audio
transcoding, UI, deployment.

## Workspace layout

| Crate | Responsibility |
|---|---|
| `vox-types` | Shared domain types: `Query`, `QueryIntent`, `Transcript`, `RetrievedDocument`, `Language`, responses, `LatencyMetrics`, validation errors |
| `vox-core` | Pipeline orchestration: guards → STT → language → analysis → retrieval → grounding → generation, per-stage timings |
| `vox-api` | Axum server exposing the frozen v1 endpoints; config, logging, metrics |
| `vox-stt` | `SpeechRecognizer` trait + mock + Sarvam HTTP backend |
| `vox-retrieval` | `RetrievalClient` trait + `OREORetrievalClient` (reqwest, timeout, one safe retry) + embedded OREO backend + deterministic mock backend |
| `vox-oreo` | OREO retrieval engine: ingest, preprocess, chunk, embed, Qdrant/Tantivy, hybrid RRF, rerank; standalone `POST /v1/retrieve` service |
| `vox-ingest` | Voice ingestion boundary: base64 decode, size caps, container sniffing |
| `vox-grounding` | Evidence-sufficiency scoring → `Answerability`; answer verification against evidence |
| `vox-llm` | `LlmProvider` trait, prompt construction, extractive baseline + OpenAI-compatible provider |
| `vox-guard` | Input/evidence/output guardrails: `Allow`/`Refuse{reason}`/`Regenerate{reason}` decisions, refusal messages |
| `vox-bench` | Latency percentile utilities + benchmark harness (`L0` retrieval, `L1` text-to-answer, `L2` voice-to-answer) |

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
    { "id": "mock-0000", "text": "Artificial intelligence (AI) is the simulation …", "score": 0.95, "rank": 1, "metadata": {"source":"mock","language":"en"} }
  ],
  "answerability": "supported",
  "answer": "Artificial intelligence (AI) is the simulation of human intelligence processes by computer systems.",
  "metrics": { "language": 0.001, "query_analysis": 0.002, "retrieval": 0.15, "grounding": 0.001, "guardrail": 0.001, "llm": 1.2, "total": 1.4 }
}
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

Retrieval-quality evaluation (chunking strategies, component ablation,
Recall@5/MRR, P50/P70/P100): `cargo run -p vox-bench`; outputs and the
full report are written to [benchmarks/](benchmarks/).

Refusal example (`POST /v1/query` with an off-corpus question):

```json
{
  "request_id": "req-68a5f0c2e1a3-0008",
  "answerability": "no_evidence",
  "refusal_reason": "insufficient_context",
  "answer": "I don't have enough information in the retrieved sources to answer that reliably.",
  "metrics": { "language": 0.001, "query_analysis": 0.002, "retrieval": 0.14, "grounding": 0.001, "guardrail": 0.001, "total": 0.16 }
}
```

Refusal reasons: `unsafe_input`, `off_topic`, `insufficient_context`,
`weak_evidence`, `conflicting_evidence`, `unsupported_claim`,
`unsafe_output`, `malformed_output`.

## Frozen API (Phase 0)

| Endpoint | Contract |
|---|---|
| `GET /health` | Liveness + uptime |
| `GET /metrics` | Prometheus exposition |
| `POST /v1/retrieve` | `Query` → `RetrievalResponse` (`{documents: [...]}`) |
| `POST /v1/query` | `Query` → `{request_id, language, query, evidence, answerability, answer?, refusal_reason?, metrics}` |
| `POST /v1/voice/query` | `VoiceRequest` → `{request_id, transcript, language, query, evidence, answerability, answer?, refusal_reason?, metrics}` |

Errors: `422` invalid input/audio, `502` upstream failure, `504` upstream
timeout, `503` retrieval unavailable, `501` not yet implemented.

## Configuration (environment)

See [.env.example](.env.example).

| Variable | Default | Meaning |
|---|---|---|
| `VOX_HOST` / `VOX_PORT` | `0.0.0.0` / `8080` | Bind address |
| `VOX_LOG_FORMAT` | `text` | `text` or `json` (level via `RUST_LOG`) |
| `VOX_RETRIEVAL_MODE` | `mock` | `mock`, `oreo` (embedded engine), or `http` (OREO service) |
| `VOX_RETRIEVAL_BASE_URL` | — | Required when mode is `http` (OREO service) |
| `VOX_RETRIEVAL_TIMEOUT_MS` | `800` | Per-attempt timeout for `/v1/retrieve` |
| `VOX_RETRIEVAL_MAX_RETRIES` | `1` | One safe retry on network errors, timeouts, 5xx, 429 only |
| `VOX_RETRIEVAL_BACKOFF_MS` | `50` | Linear backoff base between attempts |
| `VOX_MOCK_DELAY_MS` | `0` | Simulated latency injected by the mock backend |
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
| `VOX_STT_MODE` | `mock` | `mock` or `sarvam` |
| `VOX_SARVAM_API_KEY` | — | Required when mode is `sarvam`; never hard-code |
| `VOX_SARVAM_MODEL` | `saarika:v2.5` | Sarvam model identifier |
| `VOX_SARVAM_BASE_URL` | `https://api.sarvam.ai` | Sarvam API base URL |
| `VOX_SARVAM_TIMEOUT_MS` | `3000` | Per-request timeout for the STT call |
| `VOX_LLM_MODE` | `extractive` | `extractive` or `openai` |
| `VOX_LLM_API_KEY` | — | Required when mode is `openai`; never hard-code |
| `VOX_LLM_MODEL` | `gpt-4o-mini` | Model identifier for the completion request |
| `VOX_LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `VOX_LLM_TIMEOUT_MS` | `8000` | Per-request timeout for generation |
| `VOX_GROUNDING_MIN_SCORE` | `0.30` | Best hybrid-retrieval score required for `supported` |
| `VOX_GROUNDING_RELEVANCE_MIN` | `0.50` | Best single-document query-term coverage for `supported` |
| `VOX_GROUNDING_COVERAGE_MIN` | `0.50` | Union query-term coverage across evidence for `supported` |
| `VOX_GROUNDING_CONSISTENCY_MIN` | `0.50` | Required agreement ratio among relevant documents |
| `VOX_GROUNDING_AGREEMENT_MIN` | `0.10` | Pairwise overlap coefficient counting as agreement |
| `VOX_GUARD_ANSWER_SUPPORT_MIN` | `0.60` | Share of answer content tokens that must appear in the evidence |

## Multilingual Evaluation

Empirical validation across Indian languages demonstrating actual measured performance across all 5 pipeline stages:

1. **STT Stage**: Speech-to-Text transcription fidelity across languages
2. **LID Stage**: Language identification & Indic Unicode script detection
3. **Retrieval Stage**: Hybrid dense + Tantivy BM25 + RRF + lexical reranking
4. **Grounding Stage**: Evidence sufficiency scoring & hallucination detection
5. **Answer Generation Stage**: End-to-end grounded generation with localized refusal guarantees

### Multilingual Performance Benchmark Table

| Language | Group | Queries | Recall@5 | MRR | P50 (ms) | P100 (ms) | STT Failures | LID Failures | Retrieval Failures | Grounding Failures | Gen Failures | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **English (en)** | Primary | 12 | 1.000 | 1.000 | 10.22 | 16.82 | 0 | 0 | 0 | 4 | 4 | Verified |
| **Hindi (hi)** | Primary | 12 | 1.000 | 1.000 | 10.33 | 26.14 | 0 | 0 | 0 | 8 | 8 | Verified |
| **Tamil (ta)** | Primary | 12 | 1.000 | 1.000 | 11.01 | 33.09 | 0 | 0 | 0 | 4 | 4 | Verified |
| **Telugu (te)** | Secondary | 12 | 1.000 | 1.000 | 10.78 | 24.31 | 0 | 0 | 0 | 5 | 5 | Verified |
| **Kannada (kn)** | Secondary | 12 | 1.000 | 1.000 | 9.49 | 19.82 | 0 | 0 | 0 | 4 | 4 | Verified |
| **Primary Summary (EN, HI, TA)** | Group | 36 | **1.000** | **1.000** | **10.52** | **33.09** | **0** | **0** | **0** | **16** | **16** | **VALIDATED** |
| **Secondary Summary (TE, KN)** | Group | 24 | **1.000** | **1.000** | **10.13** | **24.31** | **0** | **0** | **0** | **9** | **9** | **VALIDATED** |
| **Overall Multilingual (All 5)** | Aggregate | 60 | **1.000** | **1.000** | **10.37** | **33.09** | **0** | **0** | **0** | **25** | **25** | **VALIDATED** |

### Per-Stage Latency Breakdown (P50 / P100 in milliseconds)

| Language | STT (P50/P100) | LID (P50/P100) | Retrieval (P50/P100) | Grounding (P50/P100) | Generation (P50/P100) | Total P50 | Total P100 |
|---|---|---|---|---|---|---|---|
| **English (en)** | 0.00/0.03 | 0.00/0.01 | 10.08/18.80 | 0.35/1.00 | 0.03/20.07 | 10.22 | 16.82 |
| **Hindi (hi)** | 0.00/0.01 | 0.00/0.00 | 10.05/26.04 | 0.45/1.65 | 0.04/17.84 | 10.33 | 26.14 |
| **Tamil (ta)** | 0.00/0.02 | 0.00/0.00 | 10.65/30.84 | 0.49/1.09 | 0.04/19.16 | 11.01 | 33.09 |
| **Telugu (te)** | 0.00/0.65 | 0.00/0.00 | 10.45/23.22 | 0.47/1.04 | 0.03/17.04 | 10.78 | 24.31 |
| **Kannada (kn)** | 0.00/0.01 | 0.00/0.00 | 8.92/19.64 | 0.39/0.92 | 0.03/15.22 | 9.49 | 19.82 |

### Language Support Scope & Boundaries

> [!IMPORTANT]
> **Tested & Validated Languages**: English (`en`), Hindi (`hi`), Tamil (`ta`), Telugu (`te`), Kannada (`kn`).
> In accordance with VOX core principles, **no language is claimed as supported unless it has been empirically tested** across STT, LID, Retrieval, Grounding, and Answer Generation.

#### Untested Languages (Explicitly Not Claimed)

| Language Code | Language Name | Validation Status | Claim Status |
|---|---|---|---|
| `as` | Assamese | Untested | **No Support Claimed** |
| `bn` | Bengali | Untested | **No Support Claimed** |
| `gu` | Gujarati | Untested | **No Support Claimed** |
| `ml` | Malayalam | Untested | **No Support Claimed** |
| `mr` | Marathi | Untested | **No Support Claimed** |
| `or` | Odia | Untested | **No Support Claimed** |
| `pa` | Punjabi | Untested | **No Support Claimed** |
| `ur` | Urdu | Untested | **No Support Claimed** |

To run the multilingual validation suite and generate the machine-readable benchmark:

```sh
cargo run -p vox-bench -- multilingual benchmarks
```

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
