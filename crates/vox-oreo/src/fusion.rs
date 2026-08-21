//! Reciprocal Rank Fusion (RRF).
//!
//! Pure function so it is trivially testable: given ordered result lists
//! (rank 1 = best), each document scores
//!
//! ```text
//! score(d) = Σ_lists [d ∈ list] * 1 / (k + rank_list(d))
//! ```
//!
//! with the classic constant `k = 60`. Ties break deterministically on chunk
//! id so identical inputs always produce identical rankings.

use crate::chunk::ChunkRecord;

/// Fuses ordered result lists into one ranking.
///
/// `lists` holds already-ordered chunks per retrieval leg; positions map to
/// ranks starting at 1. Output is truncated to `limit` and sorted by
/// descending fused score, ties broken by ascending chunk id.
#[must_use]
pub fn rrf_fuse(lists: &[Vec<ChunkRecord>], k: usize, limit: usize) -> Vec<(ChunkRecord, f32)> {
    let k = k.max(1);
    let mut scores: std::collections::HashMap<String, (f32, ChunkRecord)> =
        std::collections::HashMap::new();
    for list in lists {
        for (index, chunk) in list.iter().enumerate() {
            let contribution = 1.0 / (k as f32 + index as f32 + 1.0);
            let entry = scores
                .entry(chunk.chunk_id.clone())
                .or_insert_with(|| (0.0, chunk.clone()));
            entry.0 += contribution;
        }
    }
    let mut fused: Vec<(ChunkRecord, f32)> = scores
        .into_values()
        .map(|(score, chunk)| (chunk, score))
        .collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.chunk_id.cmp(&b.0.chunk_id))
    });
    fused.truncate(limit);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_types::Language;

    fn record(id: &str) -> ChunkRecord {
        ChunkRecord {
            document_id: id.to_owned(),
            chunk_id: format!("{id}#0000"),
            chunk_index: 0,
            text: format!("text {id}"),
            language: Language::En,
            chunking_strategy: "fixed".to_owned(),
            source: "test".to_owned(),
        }
    }

    #[test]
    fn documents_in_both_lists_should_outrank_single_list_documents() {
        let dense = vec![record("a"), record("b"), record("c")];
        let sparse = vec![record("c"), record("d"), record("e")];
        let fused = rrf_fuse(&[dense, sparse], 60, 10);
        assert_eq!(fused[0].0.chunk_id, "c#0000");
        // 'a' holds rank 1 of one list (1/61); 'b' and 'd' tie at 1/62 and
        // break the tie on ascending chunk id.
        assert_eq!(fused[1].0.chunk_id, "a#0000");
        assert_eq!(fused[2].0.chunk_id, "b#0000");
        assert_eq!(fused[3].0.chunk_id, "d#0000");
    }

    #[test]
    fn scores_should_follow_reciprocal_rank_formula() {
        let list = vec![record("x"), record("y")];
        let fused = rrf_fuse(&[list], 60, 10);
        assert!((fused[0].1 - 1.0 / 61.0).abs() < 1e-9);
        assert!((fused[1].1 - 1.0 / 62.0).abs() < 1e-9);
    }

    #[test]
    fn duplicates_within_one_list_should_not_double_count() {
        // Real retrievers never return duplicates, but the fusion must stay
        // stable if one ever does: same chunk id ⇒ same accumulator entry.
        let list = vec![record("dup"), record("dup")];
        let fused = rrf_fuse(&[list], 60, 10);
        assert_eq!(fused.len(), 1);
    }

    #[test]
    fn limit_should_truncate_output() {
        let list: Vec<ChunkRecord> = (0..30).map(|i| record(&format!("d{i}"))).collect();
        let fused = rrf_fuse(&[list], 60, 20);
        assert_eq!(fused.len(), 20);
    }

    #[test]
    fn empty_input_should_produce_empty_output() {
        assert!(rrf_fuse(&[], 60, 20).is_empty());
        assert!(rrf_fuse(&[Vec::new()], 60, 20).is_empty());
    }

    #[test]
    fn ordering_should_be_deterministic_across_runs() {
        let build = || {
            vec![
                vec![record("a"), record("b"), record("z")],
                vec![record("z"), record("m"), record("a")],
            ]
        };
        let first = rrf_fuse(&build(), 60, 5);
        let second = rrf_fuse(&build(), 60, 5);
        assert_eq!(first, second);
    }
}
