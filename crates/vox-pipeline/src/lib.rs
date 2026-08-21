//! VOX pipeline crate: orchestrates language handling, query analysis,
//! retrieval, grounding, generation, and guardrails for one user query.

pub mod error;
pub mod guardrail;
pub mod llm;
pub mod pipeline;

pub use error::PipelineError;
pub use guardrail::{evaluate_answer, evaluate_grounding, GuardrailVerdict};
pub use llm::{ExtractiveLlmClient, LlmClient, LlmOutput};
pub use pipeline::{PipelineConfig, VoiceRagPipeline};
