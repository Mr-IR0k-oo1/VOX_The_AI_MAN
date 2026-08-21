//! Language resolution for the pipeline.
//!
//! Priority: explicit hint → STT-provided language → local script fallback.
//! The fallback inspects Unicode block ranges only; no ML classifier.

use vox_types::Language;

/// Resolves the pipeline language for one request.
///
/// 1. `hint` — the caller's explicit language (request field).
/// 2. `stt` — the language the recognizer reported, when any.
/// 3. Script detection over `text`, defaulting to English for Latin or
///    unrecognized scripts.
#[must_use]
pub fn detect_language(hint: Option<Language>, stt: Option<Language>, text: &str) -> Language {
    hint.or(stt).unwrap_or_else(|| detect_by_script(text))
}

/// Detects a language from the first character in an Indic Unicode block.
///
/// Latin-script text (and text with no Indic characters at all) resolves to
/// English, the pipeline's working fallback.
#[must_use]
pub fn detect_by_script(text: &str) -> Language {
    for ch in text.chars() {
        let code = ch as u32;
        let lang = match code {
            0x0900..=0x097F => Language::Hi, // Devanagari
            0x0980..=0x09FF => Language::Bn, // Bengali
            0x0A00..=0x0A7F => Language::Pa, // Gurmukhi
            0x0A80..=0x0AFF => Language::Gu, // Gujarati
            0x0B00..=0x0B7F => Language::Or, // Odia
            0x0B80..=0x0BFF => Language::Ta, // Tamil
            0x0C00..=0x0C7F => Language::Te, // Telugu
            0x0C80..=0x0CFF => Language::Kn, // Kannada
            0x0D00..=0x0D7F => Language::Ml, // Malayalam
            _ => continue,
        };
        return lang;
    }
    Language::En
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_should_win_over_everything() {
        assert_eq!(
            detect_language(Some(Language::En), Some(Language::Ta), "தமிழ்"),
            Language::En
        );
    }

    #[test]
    fn stt_language_should_win_over_script_detection() {
        assert_eq!(
            detect_language(None, Some(Language::Hi), "தமிழ்"),
            Language::Hi
        );
    }

    #[test]
    fn tamil_text_should_be_detected_by_script() {
        assert_eq!(
            detect_language(None, None, "வணக்கம், தமிழ் எப்படி?"),
            Language::Ta
        );
    }

    #[test]
    fn hindi_text_should_be_detected_by_script() {
        assert_eq!(detect_language(None, None, "नमस्ते दुनिया"), Language::Hi);
    }

    #[test]
    fn english_text_should_be_the_latin_fallback() {
        assert_eq!(
            detect_language(None, None, "What is artificial intelligence?"),
            Language::En
        );
    }

    #[test]
    fn empty_text_should_fall_back_to_english() {
        assert_eq!(detect_language(None, None, ""), Language::En);
    }

    #[test]
    fn other_indic_scripts_should_be_recognized() {
        assert_eq!(detect_by_script("বাংলা"), Language::Bn);
        assert_eq!(detect_by_script("ਪੰਜਾਬੀ"), Language::Pa);
        assert_eq!(detect_by_script("తెలుగు"), Language::Te);
        assert_eq!(detect_by_script("ಕನ್ನಡ"), Language::Kn);
    }
}
