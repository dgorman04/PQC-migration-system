//! Feature schema + request-payload encoding. Only `numeric` and
//! `categorical` kinds — PQC features come pre-computed from the Core API
//! (§7.4), unlike the reference `inference-feature-extractor`'s
//! `*_extracted` kinds which derive features from raw message text. See
//! README.md §7.6.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use crate::error::Error;

#[derive(Debug, Clone, Deserialize)]
pub struct FeatureSpec {
    pub name: String,
    /// `"numeric"` or `"categorical"`.
    pub kind: String,
    #[serde(default)]
    pub categories: Option<BTreeMap<String, f32>>,
}

/// Encodes one request instance's raw feature map into the ordered tensor
/// row `features` describes. Fails loudly (400) on a missing feature or an
/// unknown category rather than silently defaulting — same philosophy as
/// the reference adapter's `missing_strategy: error` default.
pub fn encode(features: &[FeatureSpec], raw: &BTreeMap<String, Value>) -> Result<Vec<f32>, Error> {
    features.iter().map(|spec| encode_one(spec, raw)).collect()
}

fn encode_one(spec: &FeatureSpec, raw: &BTreeMap<String, Value>) -> Result<f32, Error> {
    let value = raw
        .get(&spec.name)
        .ok_or_else(|| Error::BadRequest(format!("missing feature `{}`", spec.name)))?;

    match spec.kind.as_str() {
        "numeric" => value.as_f64().map(|v| v as f32).ok_or_else(|| {
            Error::BadRequest(format!(
                "feature `{}` must be numeric, got {value}",
                spec.name
            ))
        }),
        "categorical" => {
            let categories = spec.categories.as_ref().ok_or_else(|| {
                Error::Config(format!(
                    "feature `{}` is categorical but has no categories map",
                    spec.name
                ))
            })?;
            let raw_str = value.as_str().ok_or_else(|| {
                Error::BadRequest(format!(
                    "feature `{}` must be a string category, got {value}",
                    spec.name
                ))
            })?;
            categories.get(raw_str).copied().ok_or_else(|| {
                Error::BadRequest(format!(
                    "feature `{}`: unknown category `{raw_str}`",
                    spec.name
                ))
            })
        }
        other => Err(Error::Config(format!(
            "feature `{}`: unknown kind `{other}`",
            spec.name
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Vec<FeatureSpec> {
        vec![
            FeatureSpec {
                name: "key_size".to_string(),
                kind: "numeric".to_string(),
                categories: None,
            },
            FeatureSpec {
                name: "criticality".to_string(),
                kind: "categorical".to_string(),
                categories: Some([("LOW".to_string(), 0.0), ("CRITICAL".to_string(), 3.0)].into()),
            },
        ]
    }

    fn raw(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn encodes_numeric_and_categorical() {
        let row = encode(
            &schema(),
            &raw(&[
                ("key_size", serde_json::json!(2048)),
                ("criticality", serde_json::json!("CRITICAL")),
            ]),
        )
        .unwrap();
        assert_eq!(row, vec![2048.0, 3.0]);
    }

    #[test]
    fn missing_feature_is_a_bad_request() {
        let err = encode(&schema(), &raw(&[("key_size", serde_json::json!(2048))])).unwrap_err();
        assert!(matches!(err, Error::BadRequest(_)));
    }

    #[test]
    fn unknown_category_is_a_bad_request() {
        let err = encode(
            &schema(),
            &raw(&[
                ("key_size", serde_json::json!(2048)),
                ("criticality", serde_json::json!("NOT_A_LEVEL")),
            ]),
        )
        .unwrap_err();
        assert!(matches!(err, Error::BadRequest(_)));
    }
}
