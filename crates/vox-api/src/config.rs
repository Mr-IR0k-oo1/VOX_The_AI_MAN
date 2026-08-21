//! Runtime configuration loaded from environment variables.

use std::time::Duration;

use thiserror::Error;
use vox_core::PipelineConfig;
use vox_retrieval::RetryPolicy;

/// How the retrieval backend is selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    /// Deterministic in-process mock (default until OREO's API is stable).
    Mock,
    /// Real retrieval service over HTTP.
    Http,
}

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-oriented single-line logs.
    Text,
    /// Structured JSON logs for ingestion.
    Json,
}

/// Retrieval backend settings.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Which backend implementation to build.
    pub mode: RetrievalMode,
    /// Base URL of the retrieval service (`/v1/retrieve` is appended).
    pub base_url: String,
    /// Per-attempt timeout.
    pub timeout: Duration,
    /// Bounded retry policy for transient failures.
    pub retry: RetryPolicy,
    /// Artificial latency injected by the mock backend.
    pub mock_delay: Duration,
}

/// Full runtime configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Bind host.
    pub host: String,
    /// Bind port.
    pub port: u16,
    /// Log output format.
    pub log_format: LogFormat,
    /// Retrieval backend settings.
    pub retrieval: RetrievalConfig,
    /// Pipeline tunables.
    pub pipeline: PipelineConfig,
}

/// Environment parsing failures.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// A required variable is missing.
    #[error("missing required environment variable: {0}")]
    MissingEnv(&'static str),
    /// A variable is present but cannot be parsed.
    #[error("invalid value for environment variable {name}: {value}")]
    InvalidEnv {
        /// Variable name.
        name: &'static str,
        /// Offending raw value.
        value: String,
    },
    /// The HTTP client for the retrieval backend could not be built.
    #[error("failed to build retrieval http client: {0}")]
    HttpClient(String),
}

impl Config {
    /// Loads configuration from the process environment.
    ///
    /// # Errors
    /// Returns [`ConfigError`] when a required variable is missing or a
    /// value fails to parse. Defaults are documented in the README.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_source(|name| std::env::var(name).ok())
    }

    /// Loads configuration from an arbitrary name→value source.
    ///
    /// Exists so configuration parsing is testable without mutating process
    /// state; [`Config::from_env`] delegates here.
    ///
    /// # Errors
    /// Same contract as [`Config::from_env`].
    pub fn from_source(source: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let host = source("VOX_HOST").unwrap_or_else(|| "0.0.0.0".to_owned());
        let port = parse_env(&source, "VOX_PORT", 8080)?;
        let log_format = match source("VOX_LOG_FORMAT").as_deref() {
            Some("json") => LogFormat::Json,
            Some("text") | None => LogFormat::Text,
            Some(other) => {
                return Err(ConfigError::InvalidEnv {
                    name: "VOX_LOG_FORMAT",
                    value: other.to_owned(),
                })
            }
        };

        let mode = match source("VOX_RETRIEVAL_MODE").as_deref() {
            Some("http") => RetrievalMode::Http,
            Some("mock") | None => RetrievalMode::Mock,
            Some(other) => {
                return Err(ConfigError::InvalidEnv {
                    name: "VOX_RETRIEVAL_MODE",
                    value: other.to_owned(),
                })
            }
        };
        let base_url = if mode == RetrievalMode::Http {
            Some(
                source("VOX_RETRIEVAL_BASE_URL")
                    .ok_or(ConfigError::MissingEnv("VOX_RETRIEVAL_BASE_URL"))?,
            )
        } else {
            None
        };

        Ok(Self {
            host,
            port,
            log_format,
            retrieval: RetrievalConfig {
                mode,
                base_url: base_url.unwrap_or_default(),
                timeout: Duration::from_millis(parse_env(
                    &source,
                    "VOX_RETRIEVAL_TIMEOUT_MS",
                    800,
                )?),
                retry: RetryPolicy {
                    max_retries: parse_env(&source, "VOX_RETRIEVAL_MAX_RETRIES", 2)?,
                    backoff: Duration::from_millis(parse_env(
                        &source,
                        "VOX_RETRIEVAL_BACKOFF_MS",
                        50,
                    )?),
                },
                mock_delay: Duration::from_millis(parse_env(&source, "VOX_MOCK_DELAY_MS", 0)?),
            },
            pipeline: PipelineConfig {
                grounding_min_score: parse_env(&source, "VOX_GROUNDING_MIN_SCORE", 0.30)?,
            },
        })
    }
}

fn parse_env<T: std::str::FromStr>(
    source: &impl Fn(&str) -> Option<String>,
    name: &'static str,
    default: T,
) -> Result<T, ConfigError> {
    match source(name) {
        None => Ok(default),
        Some(raw) => raw
            .trim()
            .parse::<T>()
            .map_err(|_| ConfigError::InvalidEnv { name, value: raw }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn source_from<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        let map: HashMap<&str, String> = pairs.iter().map(|(k, v)| (*k, (*v).to_owned())).collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn from_source_should_apply_defaults_when_unset() {
        let config = Config::from_source(|_| None).expect("defaults");
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.log_format, LogFormat::Text);
        assert_eq!(config.retrieval.mode, RetrievalMode::Mock);
        assert_eq!(config.retrieval.timeout, Duration::from_millis(800));
        assert!((config.pipeline.grounding_min_score - 0.30).abs() < f32::EPSILON);
    }

    #[test]
    fn from_source_should_parse_provided_values() {
        let source = source_from(&[
            ("VOX_PORT", "9000"),
            ("VOX_LOG_FORMAT", "json"),
            ("VOX_GROUNDING_MIN_SCORE", "0.5"),
        ]);
        let config = Config::from_source(source).expect("parsed");
        assert_eq!(config.port, 9000);
        assert_eq!(config.log_format, LogFormat::Json);
        assert!((config.pipeline.grounding_min_score - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn from_source_should_require_base_url_in_http_mode() {
        let source = source_from(&[("VOX_RETRIEVAL_MODE", "http")]);
        assert!(matches!(
            Config::from_source(source),
            Err(ConfigError::MissingEnv("VOX_RETRIEVAL_BASE_URL"))
        ));
    }

    #[test]
    fn from_source_should_reject_unknown_modes_and_formats() {
        let bad_mode = source_from(&[("VOX_RETRIEVAL_MODE", "quantum")]);
        assert!(matches!(
            Config::from_source(bad_mode),
            Err(ConfigError::InvalidEnv {
                name: "VOX_RETRIEVAL_MODE",
                ..
            })
        ));

        let bad_format = source_from(&[("VOX_LOG_FORMAT", "yaml")]);
        assert!(matches!(
            Config::from_source(bad_format),
            Err(ConfigError::InvalidEnv {
                name: "VOX_LOG_FORMAT",
                ..
            })
        ));
    }

    #[test]
    fn from_source_should_reject_unparsable_numbers() {
        let source = source_from(&[("VOX_PORT", "not-a-port")]);
        assert!(matches!(
            Config::from_source(source),
            Err(ConfigError::InvalidEnv {
                name: "VOX_PORT",
                ..
            })
        ));
    }
}
