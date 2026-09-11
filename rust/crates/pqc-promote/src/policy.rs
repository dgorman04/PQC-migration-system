//! Ported from the reference
//! `promotion-controller/crates/promotion-policy/src/policy.rs`
//! (README.md §5.4) — same shape, reads the same
//! `python/config/promotion_policy.yaml` this project already has.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub revision: String,
    pub environments: BTreeMap<String, EnvironmentPolicy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentPolicy {
    pub defaults: EnvironmentDefaults,
    pub deployments: BTreeMap<String, DeploymentPolicy>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentDefaults {
    pub max_eval_age_days: i64,
    pub min_promotion_interval_hours: i64,
    pub on_pass: OnPass,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentPolicy {
    pub primary_metric: String,
    #[serde(default)]
    pub min_primary_margin: f64,
    #[serde(default)]
    pub floors: BTreeMap<String, f64>,
    #[serde(default)]
    pub max_regressions: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnPass {
    Approve,
    Park,
}

/// Borrowed view of one deployment's rules, flattened with its environment's
/// defaults — what `gates.rs` actually gates against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rules<'a> {
    pub primary_metric: &'a str,
    pub min_primary_margin: f64,
    pub floors: &'a BTreeMap<String, f64>,
    pub max_regressions: &'a BTreeMap<String, f64>,
    pub max_eval_age_days: i64,
    pub min_promotion_interval_hours: i64,
    pub on_pass: OnPass,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NotInPolicy {
    #[error("the policy has no rules for `{environment}`, it only names {known}")]
    UnknownEnvironment { environment: String, known: String },

    #[error(
        "`{environment}` has no rules for `{deployment}`, it only names {known}. a model with no bars written for it can't be judged"
    )]
    UnknownDeployment {
        environment: String,
        deployment: String,
        known: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum InvalidPolicy {
    #[error("the policy could not be read from {path}: {source}")]
    Unreadable {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("the policy at {path} could not be parsed: {source}")]
    Malformed {
        path: String,
        #[source]
        source: serde_yaml::Error,
    },
}

impl Policy {
    pub fn from_yaml_path(path: impl AsRef<Path>) -> Result<Self, InvalidPolicy> {
        let path_ref = path.as_ref();
        let text =
            std::fs::read_to_string(path_ref).map_err(|source| InvalidPolicy::Unreadable {
                path: path_ref.display().to_string(),
                source,
            })?;
        serde_yaml::from_str(&text).map_err(|source| InvalidPolicy::Malformed {
            path: path_ref.display().to_string(),
            source,
        })
    }

    pub fn rules(&self, environment: &str, deployment: &str) -> Result<Rules<'_>, NotInPolicy> {
        let env =
            self.environments
                .get(environment)
                .ok_or_else(|| NotInPolicy::UnknownEnvironment {
                    environment: environment.to_string(),
                    known: join_keys(&self.environments),
                })?;

        let dep =
            env.deployments
                .get(deployment)
                .ok_or_else(|| NotInPolicy::UnknownDeployment {
                    environment: environment.to_string(),
                    deployment: deployment.to_string(),
                    known: join_keys(&env.deployments),
                })?;

        Ok(Rules {
            primary_metric: &dep.primary_metric,
            min_primary_margin: dep.min_primary_margin,
            floors: &dep.floors,
            max_regressions: &dep.max_regressions,
            max_eval_age_days: env.defaults.max_eval_age_days,
            min_promotion_interval_hours: env.defaults.min_promotion_interval_hours,
            on_pass: env.defaults.on_pass,
        })
    }
}

fn join_keys<V>(map: &BTreeMap<String, V>) -> String {
    map.keys().cloned().collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
revision: v1
environments:
  local:
    defaults:
      max_eval_age_days: 30
      min_promotion_interval_hours: 0
      on_pass: approve
    deployments:
      pqc-quantum-risk:
        primary_metric: neg_test_mae
        min_primary_margin: 0.0
        floors:
          bucket_macro_f1: 0.55
        max_regressions:
          bucket_macro_f1: 0.02
"#;

    #[test]
    fn parses_the_sample_policy() {
        let policy: Policy = serde_yaml::from_str(SAMPLE).unwrap();
        let rules = policy.rules("local", "pqc-quantum-risk").unwrap();
        assert_eq!(rules.primary_metric, "neg_test_mae");
        assert_eq!(rules.floors.get("bucket_macro_f1"), Some(&0.55));
    }

    #[test]
    fn unknown_environment_names_what_it_does_know() {
        let policy: Policy = serde_yaml::from_str(SAMPLE).unwrap();
        let err = policy.rules("production", "pqc-quantum-risk").unwrap_err();
        assert!(matches!(err, NotInPolicy::UnknownEnvironment { .. }));
    }

    #[test]
    fn unknown_deployment_names_what_it_does_know() {
        let policy: Policy = serde_yaml::from_str(SAMPLE).unwrap();
        let err = policy.rules("local", "some-other-model").unwrap_err();
        assert!(matches!(err, NotInPolicy::UnknownDeployment { .. }));
    }
}
