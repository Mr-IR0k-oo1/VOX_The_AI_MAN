# VOX — Evidence-Aware Multilingual Voice RAG (Rust)

Low-latency multilingual voice RAG for Indian languages: spoken or typed
question → STT → language detection → query analysis → hybrid retrieval
(dense + BM25 via `POST /v1/retrieve`) → grounding → guardrails → LLM answer,
with per-stage latency metrics. VOX refuses to answer — with a stable reason
code and a localized grounded-refusal message — whenever the evidence does
not support it.

## Status: Phase 7 — Performance engineering

The pipeline runs end-to-end and reaches the real retrieval service through
configuration alone (`VOX_RETRIEVAL_MODE=http`); the mock backend remains the
default for local development and tests:

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
- **Metrics**: `stt`, `language`, `query_analysis`, `retrieval`,
  `grounding`, `guardrail`, `llm`, `total` (ms); stages that did not run are
  omitted.
- **Benchmarks**: the `vox-bench` harness measures L0 (retrieval boundary),
  L1 (text-to-answer), and L2 (voice-to-answer) over a five-scenario
  multilingual test set, emitting raw samples plus mean/median/P50/P70/P100
  to `benchmarks/*.json`; see `benchmarks/performance_report.md`.

Not yet implemented (deliberately): real retrieval service, UI, deployment.

## Workspace layout

| Crate | Responsibility |
|---|---|
| `vox-types` | Shared domain types: `Query`, `QueryIntent`, `Transcript`, `RetrievedDocument`, `Language`, responses, `LatencyMetrics`, validation errors |
| `vox-core` | Pipeline orchestration: guards → STT → language → analysis → retrieval → grounding → generation, per-stage timings |
| `vox-api` | Axum server exposing the frozen v1 endpoints; config, logging, metrics |
| `vox-stt` | `SpeechRecognizer` trait + mock + Sarvam HTTP backend |
| `vox-retrieval` | `RetrievalClient` trait + `OREORetrievalClient` (reqwest, timeout, one safe retry) + deterministic mock backend |
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
| `VOX_RETRIEVAL_MODE` | `mock` | `mock` or `http` (OREO service) |
| `VOX_RETRIEVAL_BASE_URL` | — | Required when mode is `http` (OREO service) |
| `VOX_RETRIEVAL_TIMEOUT_MS` | `800` | Per-attempt timeout for `/v1/retrieve` |
| `VOX_RETRIEVAL_MAX_RETRIES` | `1` | One safe retry on network errors, timeouts, 5xx, 429 only |
| `VOX_RETRIEVAL_BACKOFF_MS` | `50` | Linear backoff base between attempts |
| `VOX_MOCK_DELAY_MS` | `0` | Simulated latency injected by the mock backend |
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

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
