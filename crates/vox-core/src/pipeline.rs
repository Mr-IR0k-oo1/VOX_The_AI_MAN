//! The user-facing RAG pipeline: language → query analysis → retrieval →
//! grounding → generation → guardrail, with per-stage timing.

use std::sync::Arc;
use std::time::Instant;

use tracing::instrument;
use vox_grounding::assess;
use vox_guard::{decide, evaluate_answer};
use vox_llm::LlmProvider;
use vox_retrieval::RetrievalClient;
use vox_types::{ms, AnswerResponse, GuardrailDecision, Language, LatencyMetrics, Query};

use crate::error::PipelineError;

/// Tunables for [`VoiceRagPipeline`].
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Best evidence score below which the pipeline refuses to answer.
    pub grounding_min_score: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            grounding_min_score: 0.30,
        }
    }
}

/// Orchestrates one text query from validation to a grounded answer or an
/// explicit refusal.
pub struct VoiceRagPipeline {
    retrieval: Arc<dyn RetrievalClient>,
    llm: Arc<dyn LlmProvider>,
    config: PipelineConfig,
}

impl VoiceRagPipeline {
    /// Creates a pipeline over the given retrieval and generation backends.
    #[must_use]
    pub fn new(
        retrieval: Arc<dyn RetrievalClient>,
        llm: Arc<dyn LlmProvider>,
        config: PipelineConfig,
    ) -> Self {
        Self {
            retrieval,
            llm,
            config,
        }
    }

    /// Runs the full pipeline for `request`.
    ///
    /// # Errors
    /// Returns [`PipelineError`] on invalid input, retrieval failure after
    /// retries, or generation failure. Refusals are returned as successful
    /// responses with `grounded = false`.
    #[instrument(
        name = "pipeline.run",
        skip_all,
        fields(query_len = request.query.len(), language = %request.language)
    )]
    pub async fn run(&self, request: Query) -> Result<AnswerResponse, PipelineError> {
        let started = Instant::now();
        let mut timings = LatencyMetrics::default();

        request.validate()?;
        let echoed_query = request.query.clone();

        let stage = Instant::now();
        let language: Language = request.language;
        timings.language = Some(ms(stage.elapsed()));

        let (normalized_query, analysis) = analyze_query(&request.query);
        timings.query_analysis = Some(analysis);

        let stage = Instant::now();
        let retrieve_query = Query {
            query: normalized_query.clone(),
            ..request
        };
        let evidence = self.retrieval.retrieve(retrieve_query).await?.documents;
        timings.retrieval = Some(ms(stage.elapsed()));

        let stage = Instant::now();
        let answerability = assess(&evidence, self.config.grounding_min_score);
        timings.grounding = Some(ms(stage.elapsed()));

        let (answer, grounded, refusal_reason) = match decide(answerability) {
            GuardrailDecision::Refuse { reason } => (String::new(), false, Some(reason)),
            GuardrailDecision::Allow => {
                let stage = Instant::now();
                let output = self
                    .llm
                    .generate(&normalized_query, language, &evidence)
                    .await?;
                timings.llm = Some(ms(stage.elapsed()));

                let stage = Instant::now();
                let answer = output.answer.trim().to_owned();
                let verdict = evaluate_answer(&answer);
                timings.guardrail = Some(ms(stage.elapsed()));

                match verdict {
                    GuardrailDecision::Allow => (answer, true, None),
                    GuardrailDecision::Refuse { reason } => (String::new(), false, Some(reason)),
                }
            }
        };

        timings.total = Some(ms(started.elapsed()));
        tracing::info!(
            grounded,
            evidence_documents = evidence.len(),
            backend = self.retrieval.name(),
            generator = self.llm.name(),
            total_ms = timings.total.unwrap_or_default(),
            "pipeline complete"
        );

        Ok(AnswerResponse {
            query: echoed_query,
            language,
            answer,
            grounded,
            refusal_reason,
            timings_ms: timings,
        })
    }
}

/// Normalizes whitespace in the query and reports the stage duration.
///
/// Phase 0 scope: collapse whitespace. Script transliteration, intent
/// detection (populating [`QueryIntent`]), and entity extraction land with
/// query analysis proper.
fn analyze_query(query: &str) -> (String, f64) {
    let stage = Instant::now();
    let mut normalized = String::with_capacity(query.len());
    let mut first = true;
    for word in query.split_whitespace() {
        if !first {
            normalized.push(' ');
        }
        normalized.push_str(word);
        first = false;
    }
    (normalized, ms(stage.elapsed()))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use async_trait::async_trait;
    use vox_llm::{LlmError, LlmOutput};
    use vox_retrieval::{MockRetrievalClient, RetrievalError};
    use vox_types::{QueryIntent, RetrievalResponse, RetrievedDocument, ValidationError};

    use super::*;

    struct EmptyRetrievalClient;

    #[async_trait]
    impl RetrievalClient for EmptyRetrievalClient {
        fn name(&self) -> &'static str {
            "empty"
        }

        async fn retrieve(&self, _request: Query) -> Result<RetrievalResponse, RetrievalError> {
            Ok(RetrievalResponse {
                documents: Vec::new(),
            })
        }
    }

    struct FailingRetrievalClient;

    #[async_trait]
    impl RetrievalClient for FailingRetrievalClient {
        fn name(&self) -> &'static str {
            "failing"
        }

        async fn retrieve(&self, _request: Query) -> Result<RetrievalResponse, RetrievalError> {
            Err(RetrievalError::Validation(ValidationError::EmptyQuery))
        }
    }

    struct EmptyLlmProvider;

    #[async_trait]
    impl LlmProvider for EmptyLlmProvider {
        fn name(&self) -> &'static str {
            "empty"
        }

        async fn generate(
            &self,
            _query: &str,
            _language: Language,
            _evidence: &[RetrievedDocument],
        ) -> Result<LlmOutput, LlmError> {
            Ok(LlmOutput {
                answer: String::new(),
            })
        }
    }

    fn mock_pipeline() -> VoiceRagPipeline {
        VoiceRagPipeline::new(
            Arc::new(MockRetrievalClient::new(Duration::ZERO)),
            Arc::new(vox_llm::ExtractiveProvider),
            PipelineConfig::default(),
        )
    }

    fn request(query: &str) -> Query {
        Query {
            query: query.to_owned(),
            language: Language::Ta,
            top_k: 3,
            intent: QueryIntent::Unknown,
        }
    }

    #[tokio::test]
    async fn run_should_return_grounded_answer_with_all_stage_timings() {
        let response = mock_pipeline()
            .run(request("  what   is  gst "))
            .await
            .expect("pipeline run");

        assert!(response.grounded);
        assert_eq!(response.refusal_reason, None);
        assert!(!response.answer.is_empty());
        // The extractive stub answers from the best chunk of the *normalized*
        // query.
        assert!(response.answer.contains("what is gst"));
        let t = &response.timings_ms;
        for stage in [
            t.language,
            t.query_analysis,
            t.retrieval,
            t.grounding,
            t.llm,
            t.guardrail,
            t.total,
        ] {
            assert!(stage.is_some(), "missing stage timing");
        }
        assert!(t.stt.is_none(), "text flow must not report stt");
    }

    #[tokio::test]
    async fn run_should_refuse_when_evidence_is_empty() {
        let pipeline = VoiceRagPipeline::new(
            Arc::new(EmptyRetrievalClient),
            Arc::new(vox_llm::ExtractiveProvider),
            PipelineConfig::default(),
        );

        let response = pipeline.run(request("anything")).await.expect("run");

        assert!(!response.grounded);
        assert_eq!(
            response.refusal_reason.as_deref(),
            Some("insufficient_evidence")
        );
        assert!(response.answer.is_empty());
        assert!(response.timings_ms.llm.is_none(), "llm must be skipped");
    }

    #[tokio::test]
    async fn run_should_refuse_empty_generations_via_guardrail() {
        let pipeline = VoiceRagPipeline::new(
            Arc::new(MockRetrievalClient::new(Duration::ZERO)),
            Arc::new(EmptyLlmProvider),
            PipelineConfig::default(),
        );

        let response = pipeline.run(request("anything")).await.expect("run");

        assert!(!response.grounded);
        assert_eq!(response.refusal_reason.as_deref(), Some("empty_generation"));
    }

    #[tokio::test]
    async fn run_should_propagate_validation_errors() {
        let err = mock_pipeline()
            .run(request("   "))
            .await
            .expect_err("should fail validation");
        assert!(matches!(
            err,
            PipelineError::Validation(ValidationError::EmptyQuery)
        ));
    }

    #[tokio::test]
    async fn run_should_propagate_retrieval_errors() {
        let pipeline = VoiceRagPipeline::new(
            Arc::new(FailingRetrievalClient),
            Arc::new(vox_llm::ExtractiveProvider),
            PipelineConfig::default(),
        );

        let err = pipeline.run(request("q")).await.expect_err("should fail");
        assert!(matches!(err, PipelineError::Retrieval(_)));
    }
}
