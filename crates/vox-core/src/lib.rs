//! VOX core orchestration: runs one user query through language handling,
//! query analysis, retrieval, grounding, generation, and guardrails, with
//! per-stage latency measurement.
//!
//! Domain types live in `vox-types`; backend boundaries live in their own
//! crates (`vox-retrieval`, `vox-llm`, `vox-grounding`, `vox-guard`). This
//! crate owns only the sequencing of those stages.

pub mod error;
pub mod pipeline;

pub use error::PipelineError;
pub use pipeline::{PipelineConfig, VoiceRagPipeline};
