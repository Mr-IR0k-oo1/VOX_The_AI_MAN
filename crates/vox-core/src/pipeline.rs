//! The IR0K voice/text pipeline: STT → language → query analysis →
//! retrieval → grounding preparation → generation, with per-stage latency
//! measurement.
//!
//! Grounding preparation is a preliminary evidence-sufficiency check only;
//! full verification (entailment, citation coverage) and guardrails are
//! later phases.

use std::sync::Arc;
use std::time::Instant;

use tracing::instrument;
use vox_ingest::decode_audio;
use vox_llm::{LlmProvider, LlmRequest};
use vox_retrieval::RetrievalClient;
use vox_stt::SpeechRecognizer;
use vox_types::{ms, LatencyMetrics, Query, QueryResponse, VoiceRequest, VoiceResponse};

use crate::analysis::analyze;
use crate::context::PipelineContext;
use crate::error::PipelineError;
use crate::language::detect_language;

/// Orchestrates one request from input to generated answer.
pub struct VoxPipeline {
    stt: Arc<dyn SpeechRecognizer>,
    retrieval: Arc<dyn RetrievalClient>,
    llm: Arc<dyn LlmProvider>,
    grounding_min_score: f32,
}

impl VoxPipeline {
    /// Creates a pipeline over the given backends.
    ///
    /// `grounding_min_score` drives the preliminary answerability check:
    /// evidence whose best score falls below it is flagged as insufficient.
    #[must_use]
    pub fn new(
        stt: Arc<dyn SpeechRecognizer>,
        retrieval: Arc<dyn RetrievalClient>,
        llm: Arc<dyn LlmProvider>,
        grounding_min_score: f32,
    ) -> Self {
        Self {
            stt,
            retrieval,
            llm,
            grounding_min_score,
        }
    }

    /// Runs the text flow: language detection → query analysis → retrieval →
    /// grounding preparation → generation.
    ///
    /// # Errors
    /// Returns [`PipelineError`] on invalid input, retrieval failure, or
    /// generation failure.
    #[instrument(
        name = "pipeline.run_text",
        skip_all,
        fields(request_id = %request_id, query_len = query.text.len())
    )]
    pub async fn run_text(
        &self,
        request_id: String,
        mut query: Query,
    ) -> Result<QueryResponse, PipelineError> {
        let started = Instant::now();
        let mut metrics = LatencyMetrics::default();

        query.validate()?;

        // Language stage: hint → script fallback (no STT on the text flow).
        let stage = Instant::now();
        let language = detect_language(query.language, None, &query.text);
        metrics.language = Some(ms(stage.elapsed()));
        query.language = Some(language);

        let context = self
            .analyze_and_generate(&request_id, None, &mut query, &mut metrics)
            .await?;
        metrics.total = Some(ms(started.elapsed()));

        tracing::info!(
            request_id = %request_id,
            language = %language,
            intent = ?query.intent,
            evidence = context.evidence.len(),
            answerability = ?context.answerability,
            backend = self.retrieval.name(),
            llm = self.llm.name(),
            total_ms = metrics.total.unwrap_or_default(),
            "text pipeline complete"
        );

        let mut response = context.into_query_response();
        response.metrics = metrics;
        Ok(response)
    }

    /// Runs the voice flow: STT → language detection → query analysis →
    /// retrieval → grounding preparation → generation.
    ///
    /// # Errors
    /// Returns [`PipelineError`] on invalid input, ingestion failure, STT
    /// failure, retrieval failure, or generation failure.
    #[instrument(name = "pipeline.run_voice", skip_all, fields(request_id = %request_id))]
    pub async fn run_voice(
        &self,
        request_id: String,
        request: VoiceRequest,
    ) -> Result<VoiceResponse, PipelineError> {
        let started = Instant::now();
        let mut metrics = LatencyMetrics::default();

        // Ingestion + STT stage.
        request.validate()?;
        let audio = decode_audio(&request.audio_base64, request.format)?;
        let stage = Instant::now();
        let transcript = self
            .stt
            .transcribe(&audio.bytes, audio.format, request.language)
            .await?;
        metrics.stt = Some(ms(stage.elapsed()));

        // Language stage: hint → STT-provided → script fallback.
        let stage = Instant::now();
        let language = detect_language(
            request.language,
            transcript.detected_language,
            &transcript.text,
        );
        metrics.language = Some(ms(stage.elapsed()));

        let mut query = Query::new(transcript.text.clone(), Some(language), request.top_k);
        let context = self
            .analyze_and_generate(
                &request_id,
                Some(transcript.clone()),
                &mut query,
                &mut metrics,
            )
            .await?;
        metrics.total = Some(ms(started.elapsed()));

        tracing::info!(
            request_id = %request_id,
            language = %language,
            intent = ?query.intent,
            evidence = context.evidence.len(),
            answerability = ?context.answerability,
            stt_backend = self.stt.name(),
            llm = self.llm.name(),
            total_ms = metrics.total.unwrap_or_default(),
            "voice pipeline complete"
        );

        let mut response = context.into_voice_response(transcript);
        response.metrics = metrics;
        Ok(response)
    }

    /// Shared tail of both flows: query analysis → retrieval → grounding
    /// preparation → generation.
    ///
    /// Mutates `query` in place (normalized text, resolved language, refined
    /// intent) and records the stage timings into `metrics`.
    async fn analyze_and_generate(
        &self,
        request_id: &str,
        transcript: Option<vox_types::Transcript>,
        query: &mut Query,
        metrics: &mut LatencyMetrics,
    ) -> Result<PipelineContext, PipelineError> {
        // Query-analysis stage.
        let stage = Instant::now();
        let (normalized_text, intent) = analyze(&query.text);
        if query.intent == vox_types::QueryIntent::Unknown {
            query.intent = intent;
        }
        query.normalized_text = normalized_text;
        metrics.query_analysis = Some(ms(stage.elapsed()));

        // Retrieval stage.
        let stage = Instant::now();
        let response = self.retrieval.retrieve(query.clone()).await?;
        metrics.retrieval = Some(ms(stage.elapsed()));
        let evidence = response.documents;

        // Grounding-preparation stage: preliminary sufficiency check only.
        let stage = Instant::now();
        let answerability = vox_grounding::assess(&evidence, self.grounding_min_score);
        metrics.grounding = Some(ms(stage.elapsed()));

        // Generation stage.
        let stage = Instant::now();
        let request = LlmRequest::new(
            query.normalized_text.clone(),
            query.language.unwrap_or(vox_types::Language::En),
            evidence,
        );
        let generated = self.llm.generate(&request).await?;
        metrics.llm = Some(ms(stage.elapsed()));
        let answer = non_empty(generated.answer);
        let evidence = request.evidence;

        Ok(PipelineContext {
            request_id: request_id.to_owned(),
            transcript,
            language: query.language.unwrap_or(vox_types::Language::En),
            query: query.clone(),
            evidence,
            answerability,
            answer,
            metrics: metrics.clone(),
        })
    }
}

/// Maps an empty generation output to `None` (refusal signal).
fn non_empty(answer: String) -> Option<String> {
    let trimmed = answer.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use serde_json::Value;
    use vox_llm::{LlmError, LlmProvider, LlmRequest, LlmResponse};
    use vox_retrieval::{MockRetrievalClient, RetrievalError};
    use vox_stt::{MockRecognizer, SpeechRecognizer, SttError};
    use vox_types::{
        Answerability, AudioFormat, Language, QueryIntent, RetrievalResponse, RetrievedDocument,
        ValidationError,
    };

    use super::*;

    const MIN_SCORE: f32 = 0.30;

    struct FailingRetrieval;

    #[async_trait]
    impl RetrievalClient for FailingRetrieval {
        fn name(&self) -> &'static str {
            "failing"
        }

        async fn retrieve(&self, _request: Query) -> Result<RetrievalResponse, RetrievalError> {
            Err(RetrievalError::InvalidResponse("boom"))
        }
    }

    struct FailingStt;

    #[async_trait]
    impl SpeechRecognizer for FailingStt {
        fn name(&self) -> &'static str {
            "failing"
        }

        async fn transcribe(
            &self,
            _audio: &[u8],
            _format: AudioFormat,
            _language: Option<Language>,
        ) -> Result<vox_types::Transcript, SttError> {
            Err(SttError::Backend("no engine".to_owned()))
        }
    }

    struct FailingLlm;

    #[async_trait]
    impl LlmProvider for FailingLlm {
        fn name(&self) -> &'static str {
            "failing"
        }

        async fn generate(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
            Err(LlmError::Backend("no model".to_owned()))
        }
    }

    /// Serves weak evidence so the answerability check flips to insufficient.
    struct WeakRetrieval;

    #[async_trait]
    impl RetrievalClient for WeakRetrieval {
        fn name(&self) -> &'static str {
            "weak"
        }

        async fn retrieve(&self, _request: Query) -> Result<RetrievalResponse, RetrievalError> {
            Ok(RetrievalResponse {
                documents: vec![RetrievedDocument {
                    id: "weak-1".to_owned(),
                    text: "barely related text".to_owned(),
                    score: 0.05,
                    rank: 1,
                    metadata: Value::Null,
                }],
            })
        }
    }

    fn mock_pipeline() -> VoxPipeline {
        VoxPipeline::new(
            Arc::new(MockRecognizer::new()),
            Arc::new(MockRetrievalClient::default()),
            Arc::new(vox_llm::ExtractiveProvider),
            MIN_SCORE,
        )
    }

    fn voice_request(audio_base64: &str, language: Option<Language>) -> VoiceRequest {
        VoiceRequest {
            audio_base64: audio_base64.to_owned(),
            format: AudioFormat::Wav,
            language,
            top_k: 3,
        }
    }

    #[tokio::test]
    async fn text_flow_should_return_answer_evidence_and_metrics() {
        let query = Query::new(
            "  What   is artificial intelligence?? ",
            Some(Language::En),
            3,
        );
        let response = mock_pipeline()
            .run_text("req-1".to_owned(), query)
            .await
            .expect("run");

        assert_eq!(response.request_id, "req-1");
        assert_eq!(response.language, Language::En);
        assert_eq!(
            response.query.normalized_text,
            "What is artificial intelligence?"
        );
        assert_eq!(response.query.intent, QueryIntent::Definition);
        assert_eq!(response.evidence.len(), 3);
        assert_eq!(response.answerability, Answerability::Answerable);
        assert!(
            response.answer.as_deref().is_some_and(|a| !a.is_empty()),
            "extractive provider must produce an answer from corpus evidence"
        );
        for stage in [
            response.metrics.language,
            response.metrics.query_analysis,
            response.metrics.retrieval,
            response.metrics.grounding,
            response.metrics.llm,
            response.metrics.total,
        ] {
            assert!(stage.is_some(), "missing stage timing");
        }
        assert!(
            response.metrics.stt.is_none(),
            "text flow must not report stt"
        );
    }

    #[tokio::test]
    async fn voice_flow_should_expose_transcript_answer_and_metrics() {
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::with_transcript(
                "what is gst",
                Some(Language::Hi),
            )),
            Arc::new(MockRetrievalClient::default()),
            Arc::new(vox_llm::ExtractiveProvider),
            MIN_SCORE,
        );

        // No hint: the STT-provided language must win over script detection
        // (the text is Latin script).
        let response = pipeline
            .run_voice("req-2".to_owned(), voice_request("aGVsbG8=", None))
            .await
            .expect("run");

        assert_eq!(response.request_id, "req-2");
        assert_eq!(response.transcript.text, "what is gst");
        assert_eq!(response.language, Language::Hi);
        assert_eq!(response.query.intent, QueryIntent::Definition);
        assert!(!response.evidence.is_empty());
        assert_eq!(response.answerability, Answerability::Answerable);
        assert!(response.answer.is_some());
        assert!(response.metrics.stt.is_some());
        assert!(response.metrics.llm.is_some());
        assert!(response.metrics.total.is_some());
    }

    #[tokio::test]
    async fn voice_flow_should_fall_back_to_script_detection() {
        let pipeline = VoxPipeline::new(
            // Recognizer reports no language; script detection must kick in.
            Arc::new(MockRecognizer::with_transcript("வணக்கம் தமிழ்", None)),
            Arc::new(MockRetrievalClient::default()),
            Arc::new(vox_llm::ExtractiveProvider),
            MIN_SCORE,
        );

        let response = pipeline
            .run_voice("req-3".to_owned(), voice_request("aGVsbG8=", None))
            .await
            .expect("run");

        assert_eq!(response.language, Language::Ta);
    }

    #[tokio::test]
    async fn hint_should_take_priority_over_stt_language() {
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::with_transcript(
                "what is gst",
                Some(Language::Hi),
            )),
            Arc::new(MockRetrievalClient::default()),
            Arc::new(vox_llm::ExtractiveProvider),
            MIN_SCORE,
        );

        let response = pipeline
            .run_voice(
                "req-4".to_owned(),
                voice_request("aGVsbG8=", Some(Language::En)),
            )
            .await
            .expect("run");

        assert_eq!(response.language, Language::En);
    }

    #[tokio::test]
    async fn weak_evidence_should_flag_insufficient_answerability() {
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::new()),
            Arc::new(WeakRetrieval),
            Arc::new(vox_llm::ExtractiveProvider),
            MIN_SCORE,
        );

        let response = pipeline
            .run_text("req-5".to_owned(), Query::new("q", Some(Language::En), 3))
            .await
            .expect("run");

        assert_eq!(response.answerability, Answerability::InsufficientEvidence);
        // Generation still runs in this phase; enforcement lands with the
        // guardrail phase.
        assert!(response.answer.is_some());
    }

    #[tokio::test]
    async fn empty_generation_output_should_map_to_no_answer() {
        struct EmptyRetrieval;
        #[async_trait]
        impl RetrievalClient for EmptyRetrieval {
            fn name(&self) -> &'static str {
                "empty"
            }
            async fn retrieve(&self, _: Query) -> Result<RetrievalResponse, RetrievalError> {
                Ok(RetrievalResponse { documents: vec![] })
            }
        }

        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::new()),
            Arc::new(EmptyRetrieval),
            Arc::new(vox_llm::ExtractiveProvider),
            MIN_SCORE,
        );
        let response = pipeline
            .run_text("req-6".to_owned(), Query::new("q", Some(Language::En), 3))
            .await
            .expect("run");
        assert_eq!(response.answerability, Answerability::InsufficientEvidence);
        assert_eq!(response.answer, None, "empty output must map to None");
    }

    #[tokio::test]
    async fn text_flow_should_propagate_validation_errors() {
        let err = mock_pipeline()
            .run_text("req-7".to_owned(), Query::new("   ", None, 5))
            .await
            .expect_err("blank query must fail");
        assert!(matches!(
            err,
            PipelineError::Validation(ValidationError::EmptyQuery)
        ));
    }

    #[tokio::test]
    async fn voice_flow_should_propagate_stt_failures() {
        let pipeline = VoxPipeline::new(
            Arc::new(FailingStt),
            Arc::new(MockRetrievalClient::default()),
            Arc::new(vox_llm::ExtractiveProvider),
            MIN_SCORE,
        );
        let err = pipeline
            .run_voice("req-8".to_owned(), voice_request("aGVsbG8=", None))
            .await
            .expect_err("stt failure must propagate");
        assert!(matches!(err, PipelineError::Stt(_)));
    }

    #[tokio::test]
    async fn both_flows_should_propagate_retrieval_failures() {
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::new()),
            Arc::new(FailingRetrieval),
            Arc::new(vox_llm::ExtractiveProvider),
            MIN_SCORE,
        );

        let text_err = pipeline
            .run_text("req-9".to_owned(), Query::new("q", Some(Language::En), 3))
            .await
            .expect_err("retrieval failure must propagate");
        assert!(matches!(text_err, PipelineError::Retrieval(_)));

        let voice_err = pipeline
            .run_voice("req-10".to_owned(), voice_request("aGVsbG8=", None))
            .await
            .expect_err("retrieval failure must propagate");
        assert!(matches!(voice_err, PipelineError::Retrieval(_)));
    }

    #[tokio::test]
    async fn both_flows_should_propagate_llm_failures() {
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::new()),
            Arc::new(MockRetrievalClient::default()),
            Arc::new(FailingLlm),
            MIN_SCORE,
        );

        let text_err = pipeline
            .run_text("req-11".to_owned(), Query::new("q", Some(Language::En), 3))
            .await
            .expect_err("llm failure must propagate");
        assert!(matches!(text_err, PipelineError::Llm(_)));

        let voice_err = pipeline
            .run_voice("req-12".to_owned(), voice_request("aGVsbG8=", None))
            .await
            .expect_err("llm failure must propagate");
        assert!(matches!(voice_err, PipelineError::Llm(_)));
    }
}
