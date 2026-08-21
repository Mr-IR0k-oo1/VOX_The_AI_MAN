//! Execution environment capture so benchmark results state the hardware
//! they were measured on.

use serde::Serialize;

/// The machine and process context a benchmark run executed in.
#[derive(Debug, Clone, Serialize)]
pub struct EnvironmentInfo {
    /// Operating system (`windows`, `linux`, ...).
    pub os: String,
    /// CPU architecture (`x86_64`, `aarch64`, ...).
    pub arch: String,
    /// CPU model string as reported by the platform.
    pub cpu_model: String,
    /// Logical parallelism available to the runtime.
    pub parallelism: usize,
    /// Build profile the benchmark binary was compiled with.
    pub profile: String,
    /// Measurement timestamp, seconds since the Unix epoch (UTC).
    pub measured_at_unix_secs: u64,
    /// Measurement timestamp rendered as `YYYY-MM-DD` (UTC).
    pub measured_at_date: String,
}

/// Detects the current execution environment.
#[must_use]
pub fn detect_environment() -> EnvironmentInfo {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    EnvironmentInfo {
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
        cpu_model: cpu_model().unwrap_or_else(|| "unknown".to_owned()),
        parallelism: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        profile: if cfg!(debug_assertions) {
            "debug".to_owned()
        } else {
            "release".to_owned()
        },
        measured_at_unix_secs: secs,
        measured_at_date: iso_utc_date(secs),
    }
}

/// Converts Unix seconds to a `YYYY-MM-DD` UTC date (proleptic Gregorian).
#[must_use]
pub fn iso_utc_date(secs: u64) -> String {
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Days since 1970-01-01 → `(year, month, day)`; Howard Hinnant's
/// `civil_from_days` algorithm, no external dependencies.
fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (
        if month <= 2 { year + 1 } else { year },
        month as u32,
        day as u32,
    )
}

#[cfg(windows)]
fn cpu_model() -> Option<String> {
    use winreg::enums::HKEY_LOCAL_MACHINE;
    use winreg::RegKey;
    let key = RegKey::predef(HKEY_LOCAL_MACHINE)
        .open_subkey(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        .ok()?;
    key.get_value::<String, _>("ProcessorNameString").ok()
}

#[cfg(not(windows))]
fn cpu_model() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_zero_should_render_as_1970_01_01() {
        assert_eq!(iso_utc_date(0), "1970-01-01");
    }

    #[test]
    fn known_timestamp_should_render_correctly() {
        // 2026-08-21T00:00:00Z
        assert_eq!(iso_utc_date(1_787_270_400), "2026-08-21");
    }
}
