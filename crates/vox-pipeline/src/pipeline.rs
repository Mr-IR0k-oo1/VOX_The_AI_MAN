//! The user-facing RAG pipeline: language → query analysis → retrieval →
//! grounding → generation → guardrail, with per-stage timing.

use std::sync::Arc;
use std::time::Instant;

use tracing::instrument;
use vox_core::{ms, AnswerResponse, Language, RetrieveRequest, StageTimings};
use vox_retrieval::RetrievalClient;

use crate::error::PipelineError;
use crate::guardrail::{evaluate_answer, evaluate_grounding, GuardrailVerdict};
use crate::llm::LlmClient;

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
    llm: Arc<dyn LlmClient>,
    config: PipelineConfig,
}

impl VoiceRagPipeline {
    /// Creates a pipeline over the given retrieval and generation backends.
    #[must_use]
    pub fn new(
        retrieval: Arc<dyn RetrievalClient>,
        llm: Arc<dyn LlmClient>,
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
    #[instrument(name = "pipeline.run", skip_all, fields(query_len = request.query.len(), language = %request.language))]
    pub async fn run(&self, request: RetrieveRequest) -> Result<AnswerResponse, PipelineError> {
        let started = Instant::now();
        let mut timings = StageTimings::default();

        request.validate()?;
        let echoed_query = request.query.clone();

        let stage = Instant::now();
        let language: Language = request.language;
        timings.language = Some(ms(stage.elapsed()));

        let (normalized_query, analysis) = analyze_query(&request.query);
        timings.query_analysis = Some(analysis);

        let stage = Instant::now();
        let retrieve_request = RetrieveRequest {
            query: normalized_query.clone(),
            language,
            top_k: request.top_k,
        };
        let evidence = self
            .retrieval
            .retrieve(retrieve_request)
            .await?
            .results;
        timings.retrieval = Some(ms(stage.elapsed()));

        let stage = Instant::now();
        let grounding = evaluate_grounding(&evidence, self.config.grounding_min_score);
        timings.grounding = Some(ms(stage.elapsed()));

        let (answer, grounded, refusal_reason) = match grounding {
            GuardrailVerdict::Refuse { reason } => (String::new(), false, Some(reason)),
            GuardrailVerdict::Allow => {
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
                    GuardrailVerdict::Allow => (answer, true, None),
                    GuardrailVerdict::Refuse { reason } => (String::new(), false, Some(reason)),
                }
            }
        };

        timings.total = Some(ms(started.elapsed()));
        tracing::info!(
            grounded,
            evidence_chunks = evidence.len(),
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
            refusal_reason: refusal_reason.map(str::to_owned),
            timings_ms: timings,
        })
    }
}

/// Normalizes whitespace in the query and reports the stage duration.
///
/// Phase 1 scope: collapse whitespace. Script transliteration, intent
/// detection, and entity extraction land with query analysis proper.
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
    use vox_core::{CoreError, Language, RetrievedChunk, RetrieveResponse};
    use vox_retrieval::{RetrievalError, RetrievalClient};

    use super::*;
    use crate::llm::{ExtractiveLlmClient, LlmOutput};

    struct EmptyRetrievalClient;

    #[async_trait]
    impl RetrievalClient for EmptyRetrievalClient {
        fn name(&self) -> &'static str {
            "empty"
        }

        async fn retrieve(
            &self,
            _request: RetrieveRequest,
        ) -> Result<RetrieveResponse, RetrievalError> {
            Ok(RetrieveResponse { results: Vec::new() })
        }
    }

    struct FailingRetrievalClient;

    #[async_trait]
    impl RetrievalClient for FailingRetrievalClient {
        fn name(&self) -> &'static str {
            "failing"
        }

        async fn retrieve(
            &self,
            _request: RetrieveRequest,
        ) -> Result<RetrieveResponse, RetrievalError> {
            Err(RetrievalError::Validation(CoreError::EmptyQuery))
        }
    }

    struct EmptyLlmClient;

    #[async_trait]
    impl LlmClient for EmptyLlmClient {
        fn name(&self) -> &'static str {
            "empty"
        }

        async fn generate(
            &self,
            _query: &str,
            _language: Language,
            _evidence: &[RetrievedChunk],
        ) -> Result<LlmOutput, PipelineError> {
            Ok(LlmOutput {
                answer: String::new(),
            })
        }
    }

    fn mock_pipeline() -> VoiceRagPipeline {
        VoiceRagPipeline::new(
            Arc::new(vox_retrieval::MockRetrievalClient::new(Duration::ZERO)),
            Arc::new(ExtractiveLlmClient),
            PipelineConfig::default(),
        )
    }

    fn request(query: &str) -> RetrieveRequest {
        RetrieveRequest {
            query: query.to_owned(),
            language: Language::Ta,
            top_k: 3,
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
            Arc::new(ExtractiveLlmClient),
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
            Arc::new(vox_retrieval::MockRetrievalClient::new(Duration::ZERO)),
            Arc::new(EmptyLlmClient),
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
            PipelineError::Validation(CoreError::EmptyQuery)
        ));
    }

    #[tokio::test]
    async fn run_should_propagate_retrieval_errors() {
        let pipeline = VoiceRagPipeline::new(
            Arc::new(FailingRetrievalClient),
            Arc::new(ExtractiveLlmClient),
            PipelineConfig::default(),
        );

        let err = pipeline.run(request("q")).await.expect_err("should fail");
        assert!(matches!(err, PipelineError::Retrieval(_)));
    }
}
