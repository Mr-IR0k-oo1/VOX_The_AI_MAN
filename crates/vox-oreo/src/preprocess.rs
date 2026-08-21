//! Preprocessing pipeline: `raw → clean → normalize → language metadata →
//! documents`.
//!
//! * **clean** strips control characters, collapses whitespace, and drops
//!   records that are too short to be useful evidence.
//! * **normalize** applies Unicode NFC so equivalent Devanagari/Tamil
//!   encodings compare equal downstream.
//! * **language metadata** detects the script of the text (Devanagari →
//!   Hindi, Tamil block → Tamil, Latin → English) and keeps only documents
//!   in the configured supported set.
//! * **documents** emits the final [`PreprocessedDocument`]s handed to the
//!   chunker.

use unicode_normalization::UnicodeNormalization;
use vox_types::Language;

use crate::error::OreoError;
use crate::ingest::RawDocument;

/// Minimum cleaned length for a document to be worth indexing.
const MIN_DOCUMENT_CHARS: usize = 20;

/// A cleaned, normalized, language-tagged document ready for chunking.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedDocument {
    /// Stable identifier carried over from ingestion.
    pub id: String,
    /// Cleaned + NFC-normalized text.
    pub text: String,
    /// Script-detected language.
    pub language: Language,
    /// Provenance label carried over from ingestion.
    pub source: String,
}

/// Counts of what preprocessing kept and why it skipped records.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PreprocessStats {
    /// Records that passed every stage.
    pub kept: usize,
    /// Records dropped because cleaning left too little text.
    pub skipped_too_short: usize,
    /// Records dropped because their language is not in the supported set.
    pub skipped_unsupported_language: usize,
}

/// Detects a language from the Unicode script mix of `text`.
///
/// Devanagari presence wins over Latin (Hindi text frequently embeds Latin
/// loanwords), Tamil likewise. Text with no letters at all maps to `None`.
#[must_use]
pub fn detect_language(text: &str) -> Option<Language> {
    let mut devanagari = 0usize;
    let mut tamil = 0usize;
    let mut latin = 0usize;
    for ch in text.chars() {
        let code = ch as u32;
        if (0x0900..=0x097F).contains(&code) {
            devanagari += 1;
        } else if (0x0B80..=0x0BFF).contains(&code) {
            tamil += 1;
        } else if ch.is_ascii_alphabetic() {
            latin += 1;
        }
    }
    if devanagari >= tamil && devanagari >= latin && devanagari > 0 {
        Some(Language::Hi)
    } else if tamil > devanagari && tamil >= latin && tamil > 0 {
        Some(Language::Ta)
    } else if latin > 0 {
        Some(Language::En)
    } else {
        None
    }
}

/// Removes control characters and collapses all whitespace runs to single
/// spaces.
#[must_use]
pub fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous_was_space = false;
    for ch in text.chars() {
        if ch.is_control() {
            continue;
        }
        if ch.is_whitespace() {
            if !previous_was_space {
                out.push(' ');
            }
            previous_was_space = true;
        } else {
            out.push(ch);
            previous_was_space = false;
        }
    }
    out.trim().to_owned()
}

/// Normalizes text to Unicode NFC form.
#[must_use]
pub fn normalize(text: &str) -> String {
    text.chars().nfc().collect()
}

/// Runs the full pipeline over raw documents.
///
/// Documents whose detected language is not in `supported` are skipped, so
/// language support stays configurable (`VOX_OREO_LANGUAGES`).
///
/// # Errors
/// Never fails on individual records; returns [`OreoError`] only if a future
/// stage needs hard failure semantics (currently infallible beyond types).
pub fn preprocess(
    documents: &[RawDocument],
    supported: &[Language],
) -> Result<(Vec<PreprocessedDocument>, PreprocessStats), OreoError> {
    let mut processed = Vec::with_capacity(documents.len());
    let mut stats = PreprocessStats::default();
    for raw in documents {
        let cleaned = clean(&raw.text);
        if cleaned.chars().count() < MIN_DOCUMENT_CHARS {
            stats.skipped_too_short += 1;
            continue;
        }
        let normalized = normalize(&cleaned);
        let language = match detect_language(&normalized) {
            Some(language) if supported.contains(&language) => language,
            _ => {
                stats.skipped_unsupported_language += 1;
                continue;
            }
        };
        processed.push(PreprocessedDocument {
            id: raw.id.clone(),
            text: normalized,
            language,
            source: raw.source.clone(),
        });
    }
    stats.kept = processed.len();
    Ok((processed, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_should_strip_controls_and_collapse_whitespace() {
        assert_eq!(clean("a\u{0000}b\t c \n d"), "ab c d");
        assert_eq!(clean("  multiple   spaces  "), "multiple spaces");
    }

    #[test]
    fn normalize_should_apply_nfc() {
        // Decomposed Devanagari "नि" (na + vowel sign) composes under NFC.
        let decomposed = "\u{0928}\u{093F}";
        assert_eq!(
            normalize(decomposed),
            "\u{0928}\u{093F}".chars().nfc().collect::<String>()
        );
    }

    #[test]
    fn detection_should_cover_english_hindi_tamil() {
        assert_eq!(
            detect_language("Goods and Services Tax"),
            Some(Language::En)
        );
        assert_eq!(detect_language("जीएसटी एक कर है"), Some(Language::Hi));
        assert_eq!(detect_language("பொருள் சேவை வரி"), Some(Language::Ta));
        assert_eq!(detect_language("12345 !!!"), None);
    }

    #[test]
    fn hindi_with_latin_loanwords_should_stay_hindi() {
        assert_eq!(detect_language("GST पंजीकरण अनिवार्य है"), Some(Language::Hi));
    }

    #[test]
    fn preprocess_should_filter_by_configured_languages() {
        let docs = vec![
            RawDocument {
                id: "1".into(),
                text: "Goods and Services Tax explained fully here".into(),
                source: "t".into(),
            },
            RawDocument {
                id: "2".into(),
                text: "जीएसटी एक अप्रत्यक्ष कर है भारत में".into(),
                source: "t".into(),
            },
            RawDocument {
                id: "3".into(),
                text: "ஜிஎஸ்டி ஒரு மறைமுக வரி ஆகும்".into(),
                source: "t".into(),
            },
            // Script-based detection cannot tell French from English, so use
            // a script outside the supported set entirely (Arabic).
            RawDocument {
                id: "4".into(),
                text: "هذا نص عربي يستخدم للتجربة فقط".into(),
                source: "t".into(),
            },
            RawDocument {
                id: "5".into(),
                text: "short".into(),
                source: "t".into(),
            },
        ];
        let (docs, stats) = preprocess(&docs, &[Language::En, Language::Hi]).expect("preprocess");
        assert_eq!(stats.kept, 2);
        assert_eq!(stats.skipped_unsupported_language, 2);
        assert_eq!(stats.skipped_too_short, 1);
        assert_eq!(docs[0].language, Language::En);
        assert_eq!(docs[1].language, Language::Hi);
    }

    #[test]
    fn preprocess_should_preserve_ids_and_sources() {
        let docs = vec![RawDocument {
            id: "en-gst-001".into(),
            text: "Goods and Services Tax explained fully here".into(),
            source: "sample-docs".into(),
        }];
        let (docs, _) = preprocess(&docs, &[Language::En]).expect("preprocess");
        assert_eq!(docs[0].id, "en-gst-001");
        assert_eq!(docs[0].source, "sample-docs");
    }
}
