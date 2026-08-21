# Phase 3 — Retrieval Evaluation Report

Measured: 2026-08-21 (Unix epoch 1787309557)

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
| fixed:400:80 | 20 | 1.000 | 1.000 | 8.88 | 10.48 | 17.72 |
| sentence:700:80 | 20 | 1.000 | 1.000 | 7.86 | 8.70 | 13.04 |
| sliding:600:300 | 20 | 1.000 | 1.000 | 10.36 | 11.54 | 21.60 |

## Retrieval component ablation

Sentence chunking held fixed; components added left to right:

| Mode | Recall@5 | MRR | P50 ms | P70 ms | P100 ms |
|---|---|---|---|---|---|
| dense_only | 0.972 | 0.914 | 0.15 | 0.19 | 1.06 |
| bm25_only | 1.000 | 0.986 | 1.00 | 1.34 | 3.00 |
| dense+bm25_score_sum | 1.000 | 0.986 | 1.60 | 2.05 | 3.41 |
| dense+bm25_rrf | 1.000 | 0.954 | 1.45 | 1.82 | 4.58 |
| dense+bm25_rrf+rerank | 1.000 | 1.000 | 11.22 | 12.91 | 21.72 |

### Per-stage latency (mean ms)

| Mode | embed | dense | sparse/bm25 | fuse | rerank |
|---|---|---|---|---|---|
| dense_only | — | 0.165 | — | — | — |
| bm25_only | — | — | 1.139 | — | — |
| dense+bm25_score_sum | — | 0.304 | 1.287 | 0.099 | — |
| dense+bm25_rrf | — | 0.276 | 1.162 | 0.098 | — |
| dense+bm25_rrf+rerank | 0.122 | 0.277 | 1.536 | 0.146 | 8.935 |

## Baseline latency (default configuration)

| Metric | P50 ms | P70 ms | P100 ms | Mean ms |
|---|---|---|---|---|
| Total retrieval | 9.47 | 10.61 | 16.59 | 9.69 |

### Baseline stage latencies

| Stage | Mean ms | P50 ms | P70 ms | P100 ms |
|---|---|---|---|---|
| embed | 0.110 | 0.093 | 0.120 | 0.377 |
| dense | 0.241 | 0.208 | 0.266 | 0.621 |
| sparse/bm25 | 1.400 | 1.239 | 1.553 | 4.260 |
| fuse/rrf | 0.113 | 0.097 | 0.119 | 0.473 |
| rerank | 7.710 | 7.559 | 8.642 | 13.896 |

## Conclusions (derived from the measurements above)

- **Best chunking strategy**: 3 strategies tie on quality (Recall@5 1.000, MRR 1.000); `sentence:700:80` is fastest at P50 7.86 ms.
- **Hybrid vs single-leg**: best single leg is `bm25_only` at Recall@5 1.000; score-sum fusion reaches 1.000 (+0.000) and RRF fusion reaches 1.000 (+0.000).
- **Does reranking improve quality?** RRF alone: Recall@5 1.000, MRR 0.954. RRF + lexical rerank: Recall@5 1.000 (+0.000), MRR 1.000 (+0.046).
- **Dominant latency stage**: `rerank` at mean 8.94 ms (~80% of total retrieval latency).
- **Retrieval latency** (full pipeline): P50 11.22 ms, P70 12.91 ms, P100 21.72 ms.
- **Best retrieval configuration overall**: `dense+bm25_rrf+rerank` (Recall@5 1.000, MRR 1.000, P50 11.22 ms).

## Limitations

- The corpus is the bundled 20-document sample; absolute numbers will not transfer to production-scale corpora.
- Judgments are hand-authored binary labels (1–2 relevant documents per query), so Recall@5 saturates quickly; treat cross-config deltas as the signal, not absolute values.
- Queries are within-language (an English query judges English documents, etc.); cross-lingual retrieval is not evaluated here.
- The embedder is the deterministic offline hashed-embedding placeholder, not a neural multilingual encoder; dense-leg quality reflects that.
- Latencies are from a developer machine in debug profile unless stated otherwise; they are useful for relative comparisons between configurations, not as production SLOs.
