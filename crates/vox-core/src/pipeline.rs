//! The IR0K voice/text pipeline: input guard → STT → language → query
//! analysis → retrieval → grounding → evidence guard → generation → output
//! guard, with per-stage latency measurement.
//!
//! Every refusal is a successful response carrying a machine-readable
//! `refusal_reason` and a localized grounded-refusal message instead of an
//! answer; only infrastructure failures surface as errors.

use std::sync::Arc;
use std::time::Instant;

use tracing::instrument;
use vox_grounding::GroundingConfig;
use vox_guard::{GuardService, REASON_UNSUPPORTED_CLAIM};
use vox_ingest::decode_audio;
use vox_llm::{LlmProvider, LlmRequest};
use vox_retrieval::RetrievalClient;
use vox_stt::SpeechRecognizer;
use vox_types::{
    ms, Answerability, LatencyMetrics, Query, QueryResponse, RetrievedDocument, Transcript,
    VoiceRequest, VoiceResponse,
};

use crate::analysis::analyze;
use crate::context::PipelineContext;
use crate::error::PipelineError;
use crate::language::detect_language;

/// How many times generation may run per request (initial try plus one
/// regeneration when the output guard flags unsupported claims).
const MAX_GENERATION_ATTEMPTS: u32 = 2;

/// Orchestrates one request from input to generated (or refused) answer.
pub struct VoxPipeline {
    stt: Arc<dyn SpeechRecognizer>,
    retrieval: Arc<dyn RetrievalClient>,
    llm: Arc<dyn LlmProvider>,
    grounding: GroundingConfig,
    guards: Arc<GuardService>,
}

impl VoxPipeline {
    /// Creates a pipeline over the given backends.
    ///
    /// `grounding` supplies the evidence-sufficiency thresholds and the
    /// answer-support minimum; `guards` screens inputs, gates on evidence,
    /// and verifies generated answers.
    #[must_use]
    pub fn new(
        stt: Arc<dyn SpeechRecognizer>,
        retrieval: Arc<dyn RetrievalClient>,
        llm: Arc<dyn LlmProvider>,
        grounding: GroundingConfig,
        guards: Arc<GuardService>,
    ) -> Self {
        Self {
            stt,
            retrieval,
            llm,
            grounding,
            guards,
        }
    }

    /// Runs the text flow: input guard → language detection → query analysis
    /// → retrieval → grounding → evidence guard → generation → output guard.
    ///
    /// # Errors
    /// Returns [`PipelineError`] on invalid input, retrieval failure, or
    /// generation failure. Grounded refusals are responses, not errors.
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

        // Input guard stage.
        let stage = Instant::now();
        let input_decision = self.guards.check_input(&query.text);
        metrics.guardrail = Some(ms(stage.elapsed()));

        let context = match input_decision {
            vox_types::GuardrailDecision::Refuse { reason } => {
                tracing::warn!(request_id = %request_id, %reason, "input guard refused");
                refusal_context(
                    &request_id,
                    None,
                    &query,
                    Vec::new(),
                    Answerability::NoEvidence,
                    &metrics,
                    &reason,
                )
            }
            _ => {
                self.analyze_and_generate(&request_id, None, &mut query, &mut metrics)
                    .await?
            }
        };
        metrics.total = Some(ms(started.elapsed()));

        tracing::info!(
            request_id = %request_id,
            language = %language,
            intent = ?query.intent,
            evidence = context.evidence.len(),
            answerability = ?context.answerability,
            refusal = ?context.refusal_reason,
            backend = self.retrieval.name(),
            llm = self.llm.name(),
            total_ms = metrics.total.unwrap_or_default(),
            "text pipeline complete"
        );

        let mut response = context.into_query_response();
        response.metrics = metrics;
        Ok(response)
    }

    /// Runs the voice flow: STT → language detection → input guard → query
    /// analysis → retrieval → grounding → evidence guard → generation →
    /// output guard.
    ///
    /// # Errors
    /// Returns [`PipelineError`] on invalid input, ingestion failure, STT
    /// failure, retrieval failure, or generation failure. Grounded refusals
    /// are responses, not errors.
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

        // Input guard stage (post-STT: the transcript is the user input).
        let stage = Instant::now();
        let input_decision = self.guards.check_input(&transcript.text);
        metrics.guardrail = Some(ms(stage.elapsed()));

        let context = match input_decision {
            vox_types::GuardrailDecision::Refuse { reason } => {
                tracing::warn!(request_id = %request_id, %reason, "input guard refused");
                refusal_context(
                    &request_id,
                    Some(transcript.clone()),
                    &query,
                    Vec::new(),
                    Answerability::NoEvidence,
                    &metrics,
                    &reason,
                )
            }
            _ => {
                self.analyze_and_generate(
                    &request_id,
                    Some(transcript.clone()),
                    &mut query,
                    &mut metrics,
                )
                .await?
            }
        };
        metrics.total = Some(ms(started.elapsed()));

        tracing::info!(
            request_id = %request_id,
            language = %language,
            intent = ?query.intent,
            evidence = context.evidence.len(),
            answerability = ?context.answerability,
            refusal = ?context.refusal_reason,
            stt_backend = self.stt.name(),
            llm = self.llm.name(),
            total_ms = metrics.total.unwrap_or_default(),
            "voice pipeline complete"
        );

        let mut response = context.into_voice_response(transcript);
        response.metrics = metrics;
        Ok(response)
    }

    /// Shared guarded tail of both flows: query analysis → retrieval →
    /// grounding → evidence guard → generation with output verification.
    ///
    /// Mutates `query` in place (normalized text, resolved language, refined
    /// intent) and records the stage timings into `metrics`.
    async fn analyze_and_generate(
        &self,
        request_id: &str,
        transcript: Option<Transcript>,
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

        // Grounding stage: evidence-sufficiency scoring. The evidence index
        // is built once and reused for answer verification below.
        let stage = Instant::now();
        let grounding_input = if query.normalized_text.is_empty() {
            query.text.clone()
        } else {
            query.normalized_text.clone()
        };
        let evidence_index = vox_grounding::EvidenceIndex::new(&evidence);
        let assessment =
            vox_grounding::assess_indexed(&grounding_input, &evidence_index, &self.grounding);
        metrics.grounding = Some(ms(stage.elapsed()));

        // Evidence guard: refuse before spending a generation call.
        let stage = Instant::now();
        let evidence_decision = self.guards.check_evidence(assessment.answerability);
        add_stage(&mut metrics.guardrail, stage.elapsed());
        if let vox_types::GuardrailDecision::Refuse { reason } = evidence_decision {
            tracing::warn!(request_id = %request_id, %reason, "evidence guard refused");
            return Ok(refusal_context(
                request_id,
                transcript,
                query,
                evidence,
                assessment.answerability,
                metrics,
                &reason,
            ));
        }

        // Generation stage with output-guard verification. Unsupported
        // answers trigger exactly one regeneration attempt; anything still
        // unsupported (or unsafe/malformed) refuses.
        let language = query.language.unwrap_or(vox_types::Language::En);
        let request = LlmRequest::new(query.normalized_text.clone(), language, evidence);
        let mut attempts = 0u32;
        let outcome = loop {
            attempts += 1;
            let stage = Instant::now();
            let generated = self.llm.generate(&request).await?;
            add_stage(&mut metrics.llm, stage.elapsed());

            let answer = generated.answer.trim().to_owned();
            let verification =
                vox_grounding::verify_answer_indexed(&answer, &evidence_index, &self.grounding);

            let stage = Instant::now();
            let decision = self.guards.check_output(&answer, &verification);
            add_stage(&mut metrics.guardrail, stage.elapsed());

            match decision {
                vox_types::GuardrailDecision::Allow => break GenerationOutcome::from(answer),
                vox_types::GuardrailDecision::Regenerate { reason }
                    if attempts < MAX_GENERATION_ATTEMPTS =>
                {
                    tracing::warn!(request_id = %request_id, %reason, attempt = attempts, "regenerating answer");
                }
                vox_types::GuardrailDecision::Regenerate { reason }
                | vox_types::GuardrailDecision::Refuse { reason } => {
                    tracing::warn!(request_id = %request_id, %reason, "output guard refused");
                    break refusal_message_fallback(&reason, language);
                }
            }
        };

        let refused = outcome.refused;
        Ok(PipelineContext {
            request_id: request_id.to_owned(),
            transcript,
            language,
            query: query.clone(),
            evidence: request.evidence,
            answerability: assessment.answerability,
            answer: outcome.answer,
            refusal_reason: refused.then(|| {
                outcome
                    .reason
                    .unwrap_or_else(|| REASON_UNSUPPORTED_CLAIM.to_owned())
            }),
            metrics: metrics.clone(),
        })
    }
}

/// Result of the generation/output-guard loop.
struct GenerationOutcome {
    answer: Option<String>,
    reason: Option<String>,
    refused: bool,
}

impl From<String> for GenerationOutcome {
    fn from(answer: String) -> Self {
        Self {
            answer: Some(answer),
            reason: None,
            refused: false,
        }
    }
}

fn refusal_message_fallback(reason: &str, language: vox_types::Language) -> GenerationOutcome {
    GenerationOutcome {
        answer: Some(vox_guard::refusal_message(reason, language)),
        reason: Some(reason.to_owned()),
        refused: true,
    }
}

/// Builds the response context for a refusal at any stage.
fn refusal_context(
    request_id: &str,
    transcript: Option<Transcript>,
    query: &Query,
    evidence: Vec<RetrievedDocument>,
    answerability: Answerability,
    metrics: &LatencyMetrics,
    reason: &str,
) -> PipelineContext {
    let language = query.language.unwrap_or(vox_types::Language::En);
    PipelineContext {
        request_id: request_id.to_owned(),
        transcript,
        language,
        query: query.clone(),
        evidence,
        answerability,
        answer: Some(vox_guard::refusal_message(reason, language)),
        refusal_reason: Some(reason.to_owned()),
        metrics: metrics.clone(),
    }
}

/// Accumulates a possibly-repeated stage duration into its metrics slot.
fn add_stage(slot: &mut Option<f64>, elapsed: std::time::Duration) {
    *slot = Some(slot.unwrap_or(0.0) + ms(elapsed));
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use serde_json::Value;
    use vox_llm::{LlmError, LlmProvider, LlmRequest, LlmResponse};
    use vox_retrieval::{MockRetrievalClient, RetrievalError};
    use vox_stt::{MockRecognizer, SpeechRecognizer, SttError};
    use vox_types::{
        AudioFormat, GuardrailDecision, Language, QueryIntent, RetrievalResponse,
        RetrievedDocument, ValidationError,
    };

    use super::*;

    const CANONICAL_REFUSAL: &str =
        "I don't have enough information in the retrieved sources to answer that reliably.";

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
        ) -> Result<Transcript, SttError> {
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

    /// Serves partially-matching low-score evidence so grounding flips to
    /// weak instead of supported.
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
                    text: "GST is a tax.".to_owned(),
                    score: 0.05,
                    rank: 1,
                    metadata: Value::Null,
                }],
            })
        }
    }

    /// Serves two topically-relevant documents that share no vocabulary, so
    /// grounding reports conflicting evidence.
    struct ConflictingRetrieval;

    #[async_trait]
    impl RetrievalClient for ConflictingRetrieval {
        fn name(&self) -> &'static str {
            "conflicting"
        }

        async fn retrieve(&self, _request: Query) -> Result<RetrievalResponse, RetrievalError> {
            Ok(RetrievalResponse {
                documents: vec![
                    RetrievedDocument {
                        id: "c-1".to_owned(),
                        text: "GST is a tax applied to goods and services in India.".to_owned(),
                        score: 0.95,
                        rank: 1,
                        metadata: Value::Null,
                    },
                    RetrievedDocument {
                        id: "c-2".to_owned(),
                        text: "Bats hunt insects at night using echolocation sounds.".to_owned(),
                        score: 0.90,
                        rank: 2,
                        metadata: Value::Null,
                    },
                ],
            })
        }
    }

    /// Always answers with a figure absent from the evidence.
    struct NumberLlm {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl LlmProvider for NumberLlm {
        fn name(&self) -> &'static str {
            "number"
        }

        async fn generate(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(LlmResponse {
                answer: "GST rate is 42% for luxury cars.".to_owned(),
                model: None,
            })
        }
    }

    fn guards() -> Arc<GuardService> {
        Arc::new(GuardService::new(MockRetrievalClient::topic_vocabulary()))
    }

    fn mock_pipeline() -> VoxPipeline {
        VoxPipeline::new(
            Arc::new(MockRecognizer::new()),
            Arc::new(MockRetrievalClient::default()),
            Arc::new(vox_llm::ExtractiveProvider),
            GroundingConfig::default(),
            guards(),
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
    async fn text_flow_should_return_supported_answer_evidence_and_metrics() {
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
        assert_eq!(response.answerability, Answerability::Supported);
        assert_eq!(response.refusal_reason, None);
        assert!(
            response.answer.as_deref().is_some_and(|a| !a.is_empty()),
            "extractive provider must produce an answer from corpus evidence"
        );
        for stage in [
            response.metrics.language,
            response.metrics.query_analysis,
            response.metrics.retrieval,
            response.metrics.grounding,
            response.metrics.guardrail,
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
            GroundingConfig::default(),
            guards(),
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
        assert_eq!(response.answerability, Answerability::Supported);
        assert!(response.answer.is_some());
        assert_eq!(response.refusal_reason, None);
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
            GroundingConfig::default(),
            guards(),
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
            GroundingConfig::default(),
            guards(),
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
    async fn weak_evidence_should_refuse_with_the_canonical_message() {
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::new()),
            Arc::new(WeakRetrieval),
            Arc::new(vox_llm::ExtractiveProvider),
            GroundingConfig::default(),
            guards(),
        );

        let response = pipeline
            .run_text(
                "req-5".to_owned(),
                Query::new("what is gst tax india", Some(Language::En), 3),
            )
            .await
            .expect("run");

        assert_eq!(response.answerability, Answerability::WeakEvidence);
        assert_eq!(response.refusal_reason.as_deref(), Some("weak_evidence"));
        assert_eq!(response.answer.as_deref(), Some(CANONICAL_REFUSAL));
        assert!(
            response.metrics.llm.is_none(),
            "refusals must not spend a generation call"
        );
        assert!(response.metrics.guardrail.is_some());
    }

    #[tokio::test]
    async fn conflicting_evidence_should_refuse() {
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::new()),
            Arc::new(ConflictingRetrieval),
            Arc::new(vox_llm::ExtractiveProvider),
            GroundingConfig::default(),
            guards(),
        );

        let response = pipeline
            .run_text(
                "req-6".to_owned(),
                Query::new("gst tax bats night", Some(Language::En), 3),
            )
            .await
            .expect("run");

        assert_eq!(response.answerability, Answerability::ConflictingEvidence);
        assert_eq!(
            response.refusal_reason.as_deref(),
            Some("conflicting_evidence")
        );
        assert_eq!(response.answer.as_deref(), Some(CANONICAL_REFUSAL));
    }

    #[tokio::test]
    async fn empty_evidence_should_refuse_without_generating() {
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
            GroundingConfig::default(),
            guards(),
        );
        let response = pipeline
            .run_text(
                "req-7".to_owned(),
                Query::new("what is gst", Some(Language::En), 3),
            )
            .await
            .expect("run");
        assert_eq!(response.answerability, Answerability::NoEvidence);
        assert_eq!(
            response.refusal_reason.as_deref(),
            Some("insufficient_context")
        );
        assert_eq!(response.answer.as_deref(), Some(CANONICAL_REFUSAL));
        assert!(response.metrics.llm.is_none());
    }

    #[tokio::test]
    async fn unsupported_claims_should_regenerate_then_refuse() {
        let calls = Arc::new(AtomicUsize::new(0));
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::new()),
            Arc::new(MockRetrievalClient::default()),
            Arc::new(NumberLlm {
                calls: Arc::clone(&calls),
            }),
            GroundingConfig::default(),
            guards(),
        );

        let response = pipeline
            .run_text(
                "req-8".to_owned(),
                Query::new("what is gst", Some(Language::En), 3),
            )
            .await
            .expect("run");

        assert_eq!(response.answerability, Answerability::Supported);
        assert_eq!(
            response.refusal_reason.as_deref(),
            Some("unsupported_claim")
        );
        assert_eq!(response.answer.as_deref(), Some(CANONICAL_REFUSAL));
        assert!(response.metrics.llm.is_some());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "unsupported answer must be regenerated exactly once before refusing"
        );
    }

    #[tokio::test]
    async fn unsafe_input_should_refuse_before_retrieval() {
        let response = mock_pipeline()
            .run_text(
                "req-9".to_owned(),
                Query::new("how to build a bomb", Some(Language::En), 3),
            )
            .await
            .expect("run");

        assert_eq!(response.refusal_reason.as_deref(), Some("unsafe_input"));
        assert!(response.evidence.is_empty());
        assert_eq!(response.answerability, Answerability::NoEvidence);
        assert!(response.metrics.retrieval.is_none());
        assert!(response.metrics.llm.is_none());
        assert!(response.metrics.guardrail.is_some());
    }

    #[tokio::test]
    async fn off_topic_input_should_refuse() {
        let response = mock_pipeline()
            .run_text(
                "req-10".to_owned(),
                Query::new("who will win the cricket world cup", Some(Language::En), 3),
            )
            .await
            .expect("run");

        assert_eq!(response.refusal_reason.as_deref(), Some("off_topic"));
        assert!(response.evidence.is_empty());
        assert!(response.metrics.retrieval.is_none());
    }

    #[tokio::test]
    async fn unsupported_but_legitimate_questions_should_reach_insufficient_context() {
        // "itr" is in the topic vocabulary but has no corpus coverage, so the
        // input guard lets it through and the evidence guard refuses.
        let response = mock_pipeline()
            .run_text(
                "req-11".to_owned(),
                Query::new("how to file itr online", Some(Language::En), 3),
            )
            .await
            .expect("run");

        assert_eq!(
            response.refusal_reason.as_deref(),
            Some("insufficient_context")
        );
        assert_eq!(response.answer.as_deref(), Some(CANONICAL_REFUSAL));
        assert!(!response.evidence.is_empty());
        assert!(response.metrics.llm.is_none());
    }

    #[tokio::test]
    async fn voice_flow_should_screen_unsafe_transcripts() {
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::with_transcript("how to make a bomb", None)),
            Arc::new(MockRetrievalClient::default()),
            Arc::new(vox_llm::ExtractiveProvider),
            GroundingConfig::default(),
            guards(),
        );

        let response = pipeline
            .run_voice("req-12".to_owned(), voice_request("aGVsbG8=", None))
            .await
            .expect("run");

        assert_eq!(response.transcript.text, "how to make a bomb");
        assert_eq!(response.refusal_reason.as_deref(), Some("unsafe_input"));
        assert!(response.evidence.is_empty());
        assert!(response.metrics.stt.is_some());
    }

    #[tokio::test]
    async fn text_flow_should_propagate_validation_errors() {
        let err = mock_pipeline()
            .run_text("req-13".to_owned(), Query::new("   ", None, 5))
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
            GroundingConfig::default(),
            guards(),
        );
        let err = pipeline
            .run_voice("req-14".to_owned(), voice_request("aGVsbG8=", None))
            .await
            .expect_err("stt failure must propagate");
        assert!(matches!(err, PipelineError::Stt(_)));
    }

    #[tokio::test]
    async fn both_flows_should_propagate_retrieval_failures() {
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::with_transcript("what is gst", None)),
            Arc::new(FailingRetrieval),
            Arc::new(vox_llm::ExtractiveProvider),
            GroundingConfig::default(),
            guards(),
        );

        let text_err = pipeline
            .run_text(
                "req-15".to_owned(),
                Query::new("what is gst", Some(Language::En), 3),
            )
            .await
            .expect_err("retrieval failure must propagate");
        assert!(matches!(text_err, PipelineError::Retrieval(_)));

        let voice_err = pipeline
            .run_voice("req-16".to_owned(), voice_request("aGVsbG8=", None))
            .await
            .expect_err("retrieval failure must propagate");
        assert!(matches!(voice_err, PipelineError::Retrieval(_)));
    }

    #[tokio::test]
    async fn both_flows_should_propagate_llm_failures() {
        let pipeline = VoxPipeline::new(
            Arc::new(MockRecognizer::with_transcript("what is gst", None)),
            Arc::new(MockRetrievalClient::default()),
            Arc::new(FailingLlm),
            GroundingConfig::default(),
            guards(),
        );

        let text_err = pipeline
            .run_text(
                "req-17".to_owned(),
                Query::new("what is gst", Some(Language::En), 3),
            )
            .await
            .expect_err("llm failure must propagate");
        assert!(matches!(text_err, PipelineError::Llm(_)));

        let voice_err = pipeline
            .run_voice("req-18".to_owned(), voice_request("aGVsbG8=", None))
            .await
            .expect_err("llm failure must propagate");
        assert!(matches!(voice_err, PipelineError::Llm(_)));
    }

    #[test]
    fn guardrail_decision_reason_accessor_should_work() {
        let decision = GuardrailDecision::Allow;
        assert_eq!(decision.reason(), None);
    }
}
