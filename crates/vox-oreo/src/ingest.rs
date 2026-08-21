//! Dataset ingestion: raw corpus files → [`RawDocument`]s.
//!
//! The primary corpus is AI4Bharat's MSMARCO-XI release, which ships JSONL
//! records whose field names vary between files (`docid`/`doc_id`/`id`,
//! `text`/`passage`/`content`, nested `passages` arrays, …). The loader here
//! accepts every variant observed in that family of files so swapping in the
//! full dataset requires no code change: point `--input` at a `.jsonl`/`.json`
//! file or a directory of them.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::OreoError;

/// One unprocessed corpus record.
#[derive(Debug, Clone, PartialEq)]
pub struct RawDocument {
    /// Stable identifier from the dataset.
    pub id: String,
    /// Raw passage text.
    pub text: String,
    /// Provenance label (usually the source file stem).
    pub source: String,
}

/// Candidate field names for the document id, in priority order.
const ID_FIELDS: [&str; 6] = ["docid", "doc_id", "document_id", "id", "_id", "qid"];
/// Candidate field names for the passage text, in priority order.
const TEXT_FIELDS: [&str; 7] = [
    "text", "passage", "content", "body", "document", "segment", "answer",
];
/// Nested containers that may hold passage lists.
const PASSAGE_LIST_FIELDS: [&str; 3] = ["positive_passages", "passages", "negative_passages"];

/// The bundled multilingual sample corpus (English, Hindi, Tamil), shipped as
/// JSONL in `data/sample-docs.jsonl` and compiled into the binary so demos,
/// tests, and benchmarks never depend on an external download.
pub const SAMPLE_CORPUS_JSONL: &str = include_str!("../data/sample-docs.jsonl");

/// Parses the bundled sample corpus into raw documents.
///
/// # Panics
/// Never in practice: the bundled file is compile-time validated by tests.
#[must_use]
pub fn sample_corpus() -> Vec<RawDocument> {
    parse_records(SAMPLE_CORPUS_JSONL, "sample-docs").expect("bundled sample corpus parses")
}

/// Candidate field names for an id embedded inside a passage object.
const PASSAGE_ID_FIELDS: [&str; 4] = ["docid", "doc_id", "document_id", "id"];

/// Extracts the id/text pairs from one JSONL record, tolerating schema
/// variants.
///
/// Records shaped like MSMARCO-XI query files (`qid` plus a
/// `positive_passages` list whose items carry their own `docid`) expand into
/// one document per passage; every other record yields at most one pair.
fn extract_records(record: &Value) -> Vec<(String, String)> {
    let Some(object) = record.as_object() else {
        return Vec::new();
    };

    for field in PASSAGE_LIST_FIELDS {
        if let Some(Value::Array(items)) = object.get(field) {
            let mut extracted = Vec::new();
            for item in items {
                let text = item
                    .as_str()
                    .map(str::to_owned)
                    .or_else(|| item.get("text").and_then(Value::as_str).map(str::to_owned));
                let Some(text) = text else { continue };
                if text.trim().is_empty() {
                    continue;
                }
                let id = PASSAGE_ID_FIELDS
                    .iter()
                    .find_map(|id_field| {
                        item.get(*id_field)
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                            .or_else(|| {
                                item.get(*id_field)
                                    .and_then(Value::as_i64)
                                    .map(|value| value.to_string())
                            })
                    })
                    .unwrap_or_default();
                extracted.push((id, text));
            }
            // Only expand when every passage is a standalone document (it
            // carries its own id); otherwise the array is just alternate
            // text fields belonging to the record itself.
            if !extracted.is_empty() && extracted.iter().all(|(id, _)| !id.is_empty()) {
                return extracted;
            }
        }
    }

    if let Some(pair) = extract_record(record) {
        return vec![pair];
    }
    Vec::new()
}

/// Extracts the single id/text pair from one JSONL record.
fn extract_record(record: &Value) -> Option<(String, String)> {
    let object = record.as_object()?;

    let id = ID_FIELDS.iter().find_map(|field| {
        object
            .get(*field)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                object
                    .get(*field)
                    .and_then(Value::as_i64)
                    .map(|value| value.to_string())
            })
    });

    let mut texts = Vec::new();
    for field in TEXT_FIELDS {
        match object.get(field) {
            Some(Value::String(text)) => texts.push(text.clone()),
            Some(Value::Array(items)) => {
                for item in items {
                    if let Some(text) = item.as_str() {
                        texts.push(text.to_owned());
                    } else if let Some(text) = item.get("text").and_then(Value::as_str) {
                        texts.push(text.to_owned());
                    }
                }
            }
            _ => {}
        }
        if !texts.is_empty() {
            break;
        }
    }

    if texts.is_empty() {
        for field in PASSAGE_LIST_FIELDS {
            if let Some(Value::Array(items)) = object.get(field) {
                for item in items {
                    let text = item
                        .as_str()
                        .map(str::to_owned)
                        .or_else(|| item.get("text").and_then(Value::as_str).map(str::to_owned));
                    if let Some(text) = text {
                        texts.push(text);
                    }
                }
            }
            if !texts.is_empty() {
                break;
            }
        }
    }

    let text = texts.join(" ");
    if text.trim().is_empty() {
        return None;
    }
    Some((id.unwrap_or_default(), text))
}

/// Reads every corpus file under `path` (a file or a directory).
///
/// `.jsonl` files are parsed line by line; `.json` files may either be a
/// single array of records or an object with a top-level `docs`/`data`
/// array (the MSMARCO collection layout).
///
/// # Errors
/// Returns [`OreoError::Io`] when the path cannot be read and
/// [`OreoError::Corpus`] when a line is not valid JSON.
pub fn load_corpus(path: &Path) -> Result<Vec<RawDocument>, OreoError> {
    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    if files.is_empty() {
        return Err(OreoError::Corpus {
            location: path.display().to_string(),
            message: "no .jsonl/.json corpus files found".to_owned(),
        });
    }
    files.sort();

    let mut documents = Vec::new();
    for file in &files {
        let source = file
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "corpus".to_owned());
        let raw = fs::read_to_string(file)?;
        documents.extend(parse_records(&raw, &source)?);
    }
    Ok(documents)
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), OreoError> {
    if path.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(path)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|entry| {
            entry
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext == "jsonl" || ext == "json")
        })
        .collect();
    files.append(&mut entries);
    Ok(())
}

fn parse_records(raw: &str, source: &str) -> Result<Vec<RawDocument>, OreoError> {
    let mut documents = Vec::new();
    let trimmed = raw.trim_start();
    if trimmed.starts_with('[') {
        let array: Value = serde_json::from_str(trimmed).map_err(|err| OreoError::Corpus {
            location: source.to_owned(),
            message: err.to_string(),
        })?;
        push_from_value(&array, source, &mut documents);
        return Ok(documents);
    }

    for (index, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: Value = serde_json::from_str(line).map_err(|err| OreoError::Corpus {
            location: format!("{source}:{}", index + 1),
            message: err.to_string(),
        })?;
        push_from_value(&record, source, &mut documents);
    }
    Ok(documents)
}

fn push_from_value(value: &Value, source: &str, documents: &mut Vec<RawDocument>) {
    // A bare array, or an object wrapping one under `docs`/`data`
    // (MSMARCO collection.json style), expands into many records.
    let items: Vec<&Value> = match value {
        Value::Array(array) => array.iter().collect(),
        Value::Object(object) => object
            .get("docs")
            .or_else(|| object.get("data"))
            .and_then(Value::as_array)
            .map(|array| array.iter().collect())
            .unwrap_or_else(|| vec![value]),
        _ => vec![value],
    };
    for item in items {
        for (id, text) in extract_records(item) {
            let id = if id.is_empty() {
                format!("{source}-{}", documents.len())
            } else {
                id
            };
            documents.push(RawDocument {
                id,
                text,
                source: source.to_owned(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, contents: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oreo-ingest-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(name);
        let mut file = fs::File::create(&path).expect("create file");
        file.write_all(contents.as_bytes()).expect("write file");
        path
    }

    #[test]
    fn extracts_docid_and_text_fields() {
        let docs = parse_records(
            r#"{"docid":"m-1","text":"Goods and Services Tax explained."}
{"doc_id":42,"passage":"जीएसटी एक अप्रत्यक्ष कर है।"}
"#,
            "msmarco",
        )
        .expect("parse");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].id, "m-1");
        assert_eq!(docs[0].text, "Goods and Services Tax explained.");
        assert_eq!(docs[1].id, "42");
        assert_eq!(docs[1].source, "msmarco");
    }

    #[test]
    fn supports_nested_positive_passages_and_collection_layout() {
        let jsonl = r#"{"qid":"q1","positive_passages":[{"docid":"d9","text":"Chennai metro water board issues tax cards."}]}"#;
        let docs = parse_records(jsonl, "queries").expect("parse");
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "d9");

        let collection =
            r#"{"version":"1.0","docs":[{"docid":"c1","text":"A"},{"docid":"c2","text":"B"}]}"#;
        let docs = parse_records(collection, "collection").expect("parse");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[1].id, "c2");
    }

    #[test]
    fn skips_records_without_usable_text() {
        let docs = parse_records(
            "{\"docid\":\"x\"}\n{\"docid\":\"y\",\"text\":\"real\"}",
            "f",
        )
        .expect("parse");
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "y");
    }

    #[test]
    fn loads_directories_of_jsonl_files() {
        write_temp("dir-a.jsonl", "{\"docid\":\"a\",\"text\":\"alpha\"}\n");
        write_temp("dir-b.jsonl", "{\"docid\":\"b\",\"text\":\"beta\"}\n");
        let dir = std::env::temp_dir().join(format!("oreo-ingest-{}", std::process::id()));
        let docs = load_corpus(&dir).expect("load");
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn missing_corpus_is_an_error() {
        let err = load_corpus(Path::new("definitely/not/here")).expect_err("missing corpus");
        assert!(matches!(err, OreoError::Corpus { .. } | OreoError::Io(_)));
    }
}
