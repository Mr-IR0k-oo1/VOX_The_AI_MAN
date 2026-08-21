//! Retrieval quality metrics: Recall@k and Mean Reciprocal Rank over
//! document-level binary judgments.

/// Fraction of judged-relevant documents present in the top `k` retrieved
/// documents. Averaging across queries happens at the call site.
#[must_use]
pub fn recall_at_k(retrieved: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let top_k: std::collections::HashSet<&String> = retrieved.iter().take(k).collect();
    let hits = relevant.iter().filter(|doc| top_k.contains(doc)).count();
    f64::from(u32::try_from(hits).unwrap_or(u32::MAX))
        / f64::from(u32::try_from(relevant.len()).unwrap_or(u32::MAX))
}

/// Reciprocal rank of the first relevant document in `retrieved`
/// (0 when nothing relevant is found). Averaging across queries yields MRR.
#[must_use]
pub fn mrr(retrieved: &[String], relevant: &[String]) -> f64 {
    for (index, doc) in retrieved.iter().enumerate() {
        if relevant.contains(doc) {
            return 1.0 / (index + 1) as f64;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_owned()).collect()
    }

    #[test]
    fn recall_should_count_relevant_hits_in_top_k() {
        let retrieved = ids(&["a", "b", "c", "d", "e"]);
        assert_eq!(recall_at_k(&retrieved, &ids(&["b", "d"]), 5), 1.0);
        assert_eq!(recall_at_k(&retrieved, &ids(&["b", "z"]), 5), 0.5);
        // Outside the cutoff it does not count.
        assert_eq!(recall_at_k(&retrieved, &ids(&["a"]), 0), 0.0);
    }

    #[test]
    fn mrr_should_invert_first_relevant_rank() {
        let retrieved = ids(&["x", "y", "b"]);
        assert_eq!(mrr(&retrieved, &ids(&["b"])), 1.0 / 3.0);
        assert_eq!(mrr(&retrieved, &ids(&["x"])), 1.0);
        assert_eq!(mrr(&retrieved, &ids(&["nope"])), 0.0);
    }

    #[test]
    fn empty_relevance_should_score_zero_not_panic() {
        assert_eq!(recall_at_k(&ids(&["a"]), &[], 5), 0.0);
        assert_eq!(mrr(&ids(&["a"]), &[]), 0.0);
    }
}
