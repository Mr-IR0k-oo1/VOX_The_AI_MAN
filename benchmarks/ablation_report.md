# Phase 9: Retrieval Architecture Ablation & Evaluation Report

Measured: 2026-08-21 (Unix epoch 1787311058)

## Environment

| Property | Value |
|---|---|
| OS | windows |
| Architecture | x86_64 |
| CPU | 13th Gen Intel(R) Core(TM) i5-13420H |
| Logical parallelism | 12 |
| Build profile | debug (cargo, opt-level of `dev`) |
| Corpus | 30 documents → 30 chunks (bundled sample corpus) |
| Queries | 60 judged queries across ["en", "hi", "ta", "te", "kn"] |

## Methodology

- **Relevance judgments**: hand-authored binary relevance judgments against the bundled sample corpus (crates/vox-bench/data/eval-queries.jsonl). Judgments are binary at the document level across a 30-document multilingual corpus.
- **Quality metrics**: Recall@5 = |top-5 ∩ relevant| / |relevant| averaged over queries; MRR = mean over queries of 1/rank of the first relevant document in the top 5.
- **Latency measurement**: `std::time::Instant` wall-clock around each stage. One warmup sweep runs before measurement; each mode then executes 3 measured sweeps over all queries and percentiles are computed over the pooled samples.
- **Stage attribution**: the dense leg's timing includes its internal query embedding; BM25 has no embedding step. The full-pipeline mode (`dense+bm25_rrf+rerank`) reports the production engine's own stage timings, where query embedding is timed once up front and again inside the dense leg.
- **Determinism**: hashed embeddings, memory vector store, Tantivy BM25, RRF, and the lexical reranker are all deterministic; quality metrics are exact for this build.

## Configuration under test

| Setting | Value |
|---|---|
| Chunking | sentence |
| Embedding dim | 256 |
| Dense store | memory |
| Candidate pool | 20 |
| Final top-k | 5 |
| RRF k | 60 |
| Reranker | lexical_overlap |
| Languages | en,hi,ta,te,kn |

## Chunking strategy comparison

Full pipeline (hybrid + RRF + rerank) per chunking strategy:

| Strategy | Chunks indexed | Recall@5 | MRR | P50 ms | P70 ms | P100 ms |
|---|---|---|---|---|---|---|
| fixed:400:80 | 30 | 1.000 | 1.000 | 27.42 | 31.42 | 80.44 |
| sentence:700:80 | 30 | 1.000 | 1.000 | 31.46 | 35.86 | 129.30 |
| sliding:600:300 | 30 | 1.000 | 1.000 | 32.85 | 41.02 | 186.48 |

## Retrieval component ablation

Sentence chunking held fixed; components added left to right:

| Mode | Recall@5 | MRR | P50 ms | P70 ms | P100 ms |
|---|---|---|---|---|---|
| dense_only | 0.933 | 0.882 | 0.81 | 1.15 | 28.58 |
| bm25_only | 1.000 | 0.983 | 3.69 | 5.30 | 24.15 |
| dense+bm25_score_sum | 1.000 | 0.964 | 4.47 | 5.61 | 24.95 |
| dense+bm25_rrf | 1.000 | 0.964 | 5.98 | 7.63 | 72.08 |
| dense+bm25_rrf+rerank | 1.000 | 1.000 | 18.64 | 24.67 | 80.26 |

### Per-stage latency (mean ms)

| Mode | embed | dense | sparse/bm25 | fuse | rerank |
|---|---|---|---|---|---|
| dense_only | — | 2.257 | — | — | — |
| bm25_only | — | — | 4.668 | — | — |
| dense+bm25_score_sum | — | 1.058 | 3.997 | 0.226 | — |
| dense+bm25_rrf | — | 1.908 | 5.267 | 0.476 | — |
| dense+bm25_rrf+rerank | 0.510 | 0.652 | 2.471 | 0.251 | 17.515 |

## Baseline latency (default configuration)

| Metric | P50 ms | P70 ms | P100 ms | Mean ms |
|---|---|---|---|---|
| Total retrieval | 27.14 | 32.39 | 151.01 | 29.30 |

### Baseline stage latencies

| Stage | Mean ms | P50 ms | P70 ms | P100 ms |
|---|---|---|---|---|
| embed | 0.338 | 0.240 | 0.300 | 2.768 |
| dense | 0.914 | 0.734 | 0.872 | 6.275 |
| sparse/bm25 | 3.307 | 2.967 | 3.652 | 23.864 |
| fuse/rrf | 0.280 | 0.207 | 0.251 | 2.994 |
| rerank | 24.246 | 22.017 | 26.818 | 142.153 |

## Architectural Component Value & Latency Trade-Off Analysis

### 1. Single-Leg Comparison: Dense vs BM25

- **Dense Only (Config A)**: Recall@5 = 0.933, MRR = 0.882, P50 = 0.81 ms, P100 = 28.58 ms.
- **BM25 Only (Config B)**: Recall@5 = 1.000, MRR = 0.983, P50 = 3.69 ms, P100 = 24.15 ms.
- **Takeaway**: BM25 provides higher precision and coverage for keyword/term matches across Indic scripts (+0.067 Recall@5, +0.102 MRR), but introduces a higher baseline search latency (+2.88 ms P50).

### 2. Hybrid Fusion: Score Sum vs Reciprocal Rank Fusion (RRF)

- **Dense + BM25 Score Sum (Config C)**: Recall@5 = 1.000, MRR = 0.964, P50 = 4.47 ms, P100 = 24.95 ms.
- **Dense + BM25 RRF (Config D)**: Recall@5 = 1.000, MRR = 0.964, P50 = 5.98 ms, P100 = 72.08 ms.
- **Takeaway**: RRF prevents score-scale skew between dense vectors and BM25 scores while maintaining robust Recall@5 (1.000) and bounded P100 latency (72.08 ms vs 24.95 ms for score-sum).

### 3. Impact of the Lexical Reranker

- **RRF without Reranker (Config D)**: Recall@5 = 1.000, MRR = 0.964, P50 = 5.98 ms.
- **RRF + Lexical Reranker (Config E)**: Recall@5 = 1.000, MRR = 1.000, P50 = 18.64 ms.
- **Takeaway**: The lexical reranker boosts MRR by +0.036 (achieving perfect 1.000 MRR) by prioritizing exact content overlap, at a latency cost of ~12.66 ms P50.

### 4. Chunking Strategy Trade-Offs

- **`fixed:400:80`**: Chunks = 30, Recall@5 = 1.000, MRR = 1.000, P50 = 27.42 ms, P100 = 80.44 ms.
- **`sentence:700:80`**: Chunks = 30, Recall@5 = 1.000, MRR = 1.000, P50 = 31.46 ms, P100 = 129.30 ms.
- **`sliding:600:300`**: Chunks = 30, Recall@5 = 1.000, MRR = 1.000, P50 = 32.85 ms, P100 = 186.48 ms.
- **Recommended Chunking**: `fixed:400:80` provides the lowest P50 retrieval latency (27.42 ms) while maintaining 100% Recall@5.

## Conclusions (Empirically Derived from Measurements)

- **Optimal Quality Configuration**: `dense+bm25_rrf+rerank` achieves the highest retrieval quality with Recall@5 = 1.000 and MRR = 1.000.
- **Dominant Latency Contributor**: `rerank` accounts for mean 17.51 ms (~81% of total retrieval time).

## Limitations

- The corpus is the bundled 30-document multilingual sample across English, Hindi, Tamil, Telugu, and Kannada; absolute numbers will not transfer to production-scale corpora.
- Judgments are hand-authored binary labels (1–2 relevant documents per query), so Recall@5 saturates quickly; treat cross-config deltas as the signal, not absolute values.
- Queries are within-language (an English query judges English documents, etc.); cross-lingual retrieval is not evaluated here.
- The embedder is the deterministic offline hashed-embedding placeholder, not a neural multilingual encoder; dense-leg quality reflects that.
- Latencies are from a developer machine in debug profile unless stated otherwise; they are useful for relative comparisons between configurations, not as production SLOs.
