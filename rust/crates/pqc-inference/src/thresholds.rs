//! Score → priority bucket mapping. Rust counterpart to
//! `python/src/pqc_riskmodel/thresholds.py` — same `risk_thresholds.yaml`
//! shape, read independently (no shared package between the two languages,
//! by design — README.md §5).

use std::path::Path;

use serde::Deserialize;

use crate::error::Error;

#[derive(Debug, Deserialize)]
struct Bucket {
    priority: String,
    min_score: f64,
}

#[derive(Debug, Deserialize)]
struct RawTable {
    buckets: Vec<Bucket>,
}

pub struct ThresholdTable {
    // ordered highest-priority first
    buckets: Vec<(String, f64)>,
}

impl ThresholdTable {
    pub fn from_yaml_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|source| Error::Config(format!("reading {}: {source}", path.display())))?;
        Self::from_yaml_str(&text)
            .map_err(|source| Error::Config(format!("parsing {}: {source}", path.display())))
    }

    fn from_yaml_str(text: &str) -> Result<Self, Error> {
        let raw: RawTable =
            serde_yaml::from_str(text).map_err(|source| Error::Config(source.to_string()))?;

        let buckets: Vec<(String, f64)> = raw
            .buckets
            .into_iter()
            .map(|b| (b.priority, b.min_score))
            .collect();
        let scores: Vec<f64> = buckets.iter().map(|(_, s)| *s).collect();
        if scores.windows(2).any(|w| w[0] < w[1]) {
            return Err(Error::Config(
                "buckets must be listed highest min_score first".to_string(),
            ));
        }

        Ok(Self { buckets })
    }

    pub fn priority_for(&self, score: f64) -> &str {
        for (priority, min_score) in &self.buckets {
            if score >= *min_score {
                return priority;
            }
        }
        self.buckets
            .last()
            .map(|(priority, _)| priority.as_str())
            .unwrap_or("UNKNOWN")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "buckets:\n  - priority: CRITICAL\n    min_score: 0.75\n  - priority: HIGH\n    min_score: 0.5\n  - priority: LOW\n    min_score: 0.0\n";

    #[test]
    fn picks_the_first_bucket_whose_floor_the_score_clears() {
        let table = ThresholdTable::from_yaml_str(SAMPLE).unwrap();
        assert_eq!(table.priority_for(0.9), "CRITICAL");
        assert_eq!(table.priority_for(0.6), "HIGH");
        assert_eq!(table.priority_for(0.1), "LOW");
    }

    #[test]
    fn rejects_out_of_order_buckets() {
        let bad = "buckets:\n  - priority: LOW\n    min_score: 0.0\n  - priority: CRITICAL\n    min_score: 0.75\n";
        assert!(ThresholdTable::from_yaml_str(bad).is_err());
    }
}
