//! VOX core orchestration: runs one user request through STT, language
//! detection, query analysis, and retrieval, with per-stage latency
//! measurement.
//!
//! Domain types live in `vox-types`; backend boundaries live in their own
//! crates (`vox-stt`, `vox-retrieval`, `vox-ingest`). This crate owns the
//! sequencing of those stages and the analysis heuristics.

pub mod analysis;
pub mod context;
pub mod error;
pub mod language;
pub mod pipeline;

pub use context::PipelineContext;
pub use error::PipelineError;
pub use language::{detect_by_script, detect_language};
pub use pipeline::VoxPipeline;
