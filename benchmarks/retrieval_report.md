# Phase 3 — Retrieval Evaluation Report

Measured: 2026-08-21 (Unix epoch 1787302302)

## Environment

| Property | Value |
|---|---|
| OS | windows |
| Architecture | x86_64 |
| CPU | 13th Gen Intel(R) Core(TM) i5-13420H |
| Logical parallelism | 12 |
| Build profile | debug (cargo, opt-level of `dev`) |
| Corpus | 20 documents → 20 chunks (bundled sample corpus) |
| Queries | 36 judged queries across ["en", "hi", "ta"] |

## Methodology

- **Relevance judgments**: hand-authored binary relevance judgments against the bundled sample corpus (crates/vox-bench/data/eval-queries.jsonl). Judgments are binary at the document level; every query has 1–2 relevant documents in a 20-document corpus.
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
| Languages | en,hi,ta |

## Chunking strategy comparison

Full pipeline (hybrid + RRF + rerank) per chunking strategy:

| Strategy | Chunks indexed | Recall@5 | MRR | P50 ms | P70 ms | P100 ms |
|---|---|---|---|---|---|---|
| fixed:400:80 | 20 | 1.000 | 1.000 | 4.84 | 5.08 | 6.90 |
| sentence:700:80 | 20 | 1.000 | 1.000 | 4.81 | 5.03 | 11.18 |
| sliding:600:300 | 20 | 1.000 | 1.000 | 4.75 | 5.01 | 8.45 |

## Retrieval component ablation

Sentence chunking held fixed; components added left to right:

| Mode | Recall@5 | MRR | P50 ms | P70 ms | P100 ms |
|---|---|---|---|---|---|
| dense_only | 0.972 | 0.914 | 0.13 | 0.16 | 0.35 |
| bm25_only | 1.000 | 0.986 | 0.37 | 0.42 | 1.05 |
| dense+bm25_score_sum | 1.000 | 0.986 | 0.55 | 0.62 | 1.01 |
| dense+bm25_rrf | 1.000 | 0.954 | 0.63 | 0.71 | 1.42 |
| dense+bm25_rrf+rerank | 1.000 | 1.000 | 4.91 | 5.15 | 7.29 |

### Per-stage latency (mean ms)

| Mode | embed | dense | sparse/bm25 | fuse | rerank |
|---|---|---|---|---|---|
| dense_only | — | 0.142 | — | — | — |
| bm25_only | — | — | 0.395 | — | — |
| dense+bm25_score_sum | — | 0.119 | 0.417 | 0.031 | — |
| dense+bm25_rrf | — | 0.126 | 0.473 | 0.041 | — |
| dense+bm25_rrf+rerank | 0.056 | 0.122 | 0.656 | 0.052 | 4.031 |

## Baseline latency (default configuration)

| Metric | P50 ms | P70 ms | P100 ms | Mean ms |
|---|---|---|---|---|
| Total retrieval | 5.01 | 5.30 | 14.98 | 5.32 |

### Baseline stage latencies

| Stage | Mean ms | P50 ms | P70 ms | P100 ms |
|---|---|---|---|---|
| embed | 0.060 | 0.053 | 0.057 | 0.248 |
| dense | 0.130 | 0.120 | 0.126 | 0.320 |
| sparse/bm25 | 0.774 | 0.711 | 0.816 | 2.512 |
| fuse/rrf | 0.058 | 0.053 | 0.058 | 0.194 |
| rerank | 4.230 | 4.066 | 4.220 | 11.668 |

## Conclusions (derived from the measurements above)

- **Best chunking strategy**: 3 strategies tie on quality (Recall@5 1.000, MRR 1.000); `sliding:600:300` is fastest at P50 4.75 ms.
- **Hybrid vs single-leg**: best single leg is `bm25_only` at Recall@5 1.000; score-sum fusion reaches 1.000 (+0.000) and RRF fusion reaches 1.000 (+0.000).
- **Does reranking improve quality?** RRF alone: Recall@5 1.000, MRR 0.954. RRF + lexical rerank: Recall@5 1.000 (+0.000), MRR 1.000 (+0.046).
- **Dominant latency stage**: `rerank` at mean 4.03 ms (~81% of total retrieval latency).
- **Retrieval latency** (full pipeline): P50 4.91 ms, P70 5.15 ms, P100 7.29 ms.
- **Best retrieval configuration overall**: `dense+bm25_rrf+rerank` (Recall@5 1.000, MRR 1.000, P50 4.91 ms).

## Limitations

- The corpus is the bundled 20-document sample; absolute numbers will not transfer to production-scale corpora.
- Judgments are hand-authored binary labels (1–2 relevant documents per query), so Recall@5 saturates quickly; treat cross-config deltas as the signal, not absolute values.
- Queries are within-language (an English query judges English documents, etc.); cross-lingual retrieval is not evaluated here.
- The embedder is the deterministic offline hashed-embedding placeholder, not a neural multilingual encoder; dense-leg quality reflects that.
- Latencies are from a developer machine in debug profile unless stated otherwise; they are useful for relative comparisons between configurations, not as production SLOs.
