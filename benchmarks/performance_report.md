# VOX Performance Report — Phase 7

All numbers in this report were produced by the `vox-bench` harness
(`crates/vox-bench`) against the real pipeline code paths. Nothing is
estimated or transcribed from memory: the raw per-run samples behind every
figure are committed alongside this report.

- `retrieval.json` — L0 raw samples and summaries
- `text_e2e.json` — L1 raw samples and summaries
- `voice_e2e.json` — L2 raw samples and summaries

## Methodology

- **Build profile:** release (`cargo run -p vox-bench --release`); the
  `profile` field inside each JSON confirms this at capture time.
- **Levels:**
  - **L0** — query → retrieval boundary (`RetrievalClient::retrieve`).
  - **L1** — query → analysis → retrieval → grounding → LLM → guardrails
    (`VoxPipeline::run_text`).
  - **L2** — audio → STT → full text tail (`VoxPipeline::run_voice` with the
    mock recognizer; STT itself is a stub, so L2 ≈ L1 + transcript handling).
- **Test set (5 scenarios, all exercised at every level):**
  - `supported_en` — "What is artificial intelligence?" (En)
  - `supported_hi` — "जीएसटी क्या है?" (Hi)
  - `supported_ta` — "குங்குமப்பூ என்றால் என்ன?" (auto-detected)
  - `unsupported_itr` — "how to file itr online" (in-domain, no corpus
    coverage → insufficiency refusal path)
  - `offtopic_cricket` — "who will win the cricket world cup" (off-topic →
    evidence-guard refusal path)
- **Rounds:** 10 untimed warmup rounds, then 100 timed rounds × 5 scenarios
  = **500 samples per level**.
- **Timing:** totals are external wall-clock (`Instant`) around the public
  API call; stage times come from the pipeline's own instrumentation
  (`LatencyMetrics`). Percentiles use linear interpolation between closest
  ranks; P100 is the maximum observed.
- **Backends:** mock retrieval, extractive LLM, mock STT — deterministic by
  design. Dense/BM25/RRF/reranker stages execute inside the OREO service
  behind the frozen `/v1/retrieve` contract, so they are *not* separately
  visible here; the measured `retrieval` stage is end-to-end for that
  boundary. A `--retrieval-url` flag exists to benchmark against a live OREO
  instance; committed numbers use the in-process mock so that latencies
  reflect VOX's own code, not network conditions of one particular run.

## Results

### End-to-end percentiles (ms)

| Level | Metric | Before | After | Δ |
|-------|--------|--------|-------|---|
| L0 retrieval | mean | 0.0015 | 0.0010 | −33% |
| L0 retrieval | P50 | 0.0014 | 0.0008 | −43% |
| L0 retrieval | P70 | 0.0016 | 0.0010 | −38% |
| L0 retrieval | P100 | 0.0126 | 0.0106 | −16% |
| L1 text-to-answer | mean | 0.0646 | 0.0330 | −49% |
| L1 text-to-answer | P50 | 0.0527 | 0.0193 | −63% |
| L1 text-to-answer | P70 | 0.1150 | 0.0535 | −53% |
| L1 text-to-answer | P100 | 0.1987 | 0.1183 | −40% |
| L2 voice-to-answer | mean | 0.0596 | 0.0354 | −41% |
| L2 voice-to-answer | P50 | 0.0506 | 0.0196 | −61% |
| L2 voice-to-answer | P70 | 0.0975 | 0.0533 | −45% |
| L2 voice-to-answer | P100 | 0.1856 | 0.1353 | −27% |

"Before" is the pre-optimization baseline captured with the same harness,
rounds, and warmup before any Phase 7 code changes; "After" is the final
state stored in `benchmarks/`.

### Stage breakdown, L1 after optimization (ms, mean / P50 / P70 / P100)

| Stage | mean | P50 | P70 | P100 |
|-------|------|-----|-----|------|
| grounding | 0.0201 | 0.0240 | 0.0272 | 0.0596 |
| guardrail | 0.0079 | 0.0043 | 0.0133 | 0.0381 |
| retrieval | 0.0014 | 0.0013 | 0.0016 | 0.0041 |
| query_analysis | 0.0006 | 0.0005 | 0.0006 | 0.0022 |
| llm | 0.0003 | 0.0003 | 0.0004 | 0.0011 |
| language | ~0.0000 | 0.0000 | 0.0000 | 0.0001 |

L2 adds `stt` (~0.0001 ms mean). Note that answer verification
(`verify_answer_indexed`) runs between the LLM and guardrail timers and is
therefore not attributed to any single stage metric; its cost is included in
the wall-clock totals above.

## Bottleneck analysis and optimizations

The first measurement round showed grounding + guardrails dominating:
together they accounted for roughly 70% of L1 stage time, an order of
magnitude above everything else. Two targeted optimizations followed, each
verified by re-running the identical harness:

1. **Evidence tokenized once per request (vox-grounding).** Previously
   `assess()` tokenized every retrieved document into hash sets, and
   `verify_answer()` re-tokenized the same documents again to build an
   evidence union — double work on the hot path. Added `EvidenceIndex`
   (single tokenization pass, deduplicated per-document token lists) plus
   `assess_indexed`/`verify_answer_indexed`; the pipeline builds the index
   once and shares it. Membership checks use linear scans over small token
   lists instead of hashing. The original functions remain as thin wrappers,
   so behavior and public API are unchanged.
   Effect: L1 total mean 0.0646 → 0.0476 ms (−26%).

2. **Allocation-free unsafe-term matching (vox-guard).** `is_unsafe`
   formatted two candidate plural strings (`format!("{term}s")`,
   `format!("{term}es")`) for every token × term pair — ~200 heap
   allocations per check, twice per request. Replaced with zero-allocation
   prefix/suffix comparison (`starts_with` + suffix match), preserving exact
   matching semantics including plurals.
   Effect: guardrail stage mean 0.0224 → 0.0079 ms (−65%); L1 total mean
   0.0476 → 0.0330 ms (−31% further).

Combined: **L1 mean −49%, L1 P50 −63%, L2 mean −41%** with byte-identical
pipeline decisions (all 173 workspace tests pass unchanged).

## What was deliberately not optimized

Per the spec, components irrelevant to the identified bottleneck were left
alone:

- **STT, language detection, query analysis, LLM** — each contributes
  ≤ 0.0007 ms mean; there is nothing meaningful to win.
- **Retrieval boundary (L0)** — already sub-microsecond against the mock;
  against a live OREO deployment its latency will be dominated by the
  network and OREO's own dense/BM25/rerank stages, which live outside this
  repository's control.
- **Grounding beyond round 1** — still the largest stage (~0.02 ms mean),
  but the remaining cost is dominated by owned-`String` token allocation
  inherent to the current `tokenize() -> Vec<String>` API. Removing it means
  an arena/interning redesign across three crates for a fraction of a
  millisecond on refusals-and-answers alike; not justified at current
  traffic assumptions.

## Reproducing

```sh
cargo run -p vox-bench --release -- --rounds 100 --warmup 10 --out benchmarks
```

Flags: `--level all|L0|L1|L2`, `--rounds N`, `--warmup N`, `--out DIR`,
`--retrieval-url URL` (switches L0–L2 to a live OREO backend).
