//! Ported from the reference
//! `promotion-controller/crates/promotion-policy/src/evaluate.rs`
//! (README.md §5.4) — the single entry point the rest of the crate exists to
//! support: compare, then run every gate in a fixed order, stopping at the
//! first failure.

use chrono::{DateTime, Utc};

use crate::compare::{compare, Candidate, Champion, Comparison};
use crate::decision::{Decision, Outcome, Parked};
use crate::gates::{age, environment_rules, floors, improvement, regressions};
use crate::policy::Rules;

pub fn evaluate(
    rules: &Rules<'_>,
    environment: &str,
    champion: Option<Champion<'_>>,
    candidate: Candidate<'_>,
    at: DateTime<Utc>,
    last_promoted_at: Option<DateTime<Utc>>,
) -> Decision {
    let comparison = match compare(champion, candidate) {
        Ok(comparison) => comparison,
        Err(refusal) => {
            return Decision {
                outcome: Outcome::Park(Parked::NotComparable(refusal)),
                comparison: None,
            };
        }
    };

    let outcome = match gates(rules, environment, &comparison, at, last_promoted_at) {
        Ok(()) => Outcome::Approve,
        Err(outcome) => outcome,
    };

    Decision {
        outcome,
        comparison: Some(comparison),
    }
}

fn gates(
    rules: &Rules<'_>,
    environment: &str,
    comparison: &Comparison,
    at: DateTime<Utc>,
    last_promoted_at: Option<DateTime<Utc>>,
) -> Result<(), Outcome> {
    floors(rules, comparison).map_err(Outcome::Reject)?;
    improvement(rules, comparison).map_err(Outcome::Reject)?;
    regressions(rules, comparison).map_err(Outcome::Reject)?;
    age(rules, comparison, at).map_err(Outcome::Park)?;
    environment_rules(rules, environment, at, last_promoted_at).map_err(Outcome::Park)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::Eval;
    use crate::policy::OnPass;
    use std::collections::BTreeMap;

    fn rules<'a>(
        floors: &'a BTreeMap<String, f64>,
        max_regressions: &'a BTreeMap<String, f64>,
    ) -> Rules<'a> {
        Rules {
            primary_metric: "neg_test_mae",
            min_primary_margin: 0.0,
            floors,
            max_regressions,
            max_eval_age_days: 30,
            min_promotion_interval_hours: 0,
            on_pass: OnPass::Approve,
        }
    }

    fn eval(metrics: &[(&str, f64)]) -> Eval {
        Eval {
            dataset_id: "ds-1".to_string(),
            evaluated_at: Utc::now(),
            metrics: metrics.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    #[test]
    fn approves_when_every_gate_passes() {
        let floors: BTreeMap<String, f64> = [("bucket_macro_f1".to_string(), 0.55)].into();
        let max_regressions: BTreeMap<String, f64> = [("bucket_macro_f1".to_string(), 0.02)].into();

        let champion = eval(&[("neg_test_mae", -0.10), ("bucket_macro_f1", 0.60)]);
        let candidate = eval(&[("neg_test_mae", -0.08), ("bucket_macro_f1", 0.62)]);

        let decision = evaluate(
            &rules(&floors, &max_regressions),
            "local",
            Some(Champion(&champion)),
            Candidate(&candidate),
            Utc::now(),
            None,
        );

        assert!(decision.approved());
    }

    #[test]
    fn parks_when_there_is_no_champion_yet() {
        let floors: BTreeMap<String, f64> = [("bucket_macro_f1".to_string(), 0.55)].into();
        let max_regressions: BTreeMap<String, f64> = [("bucket_macro_f1".to_string(), 0.02)].into();

        let candidate = eval(&[("neg_test_mae", -0.08), ("bucket_macro_f1", 0.62)]);

        let decision = evaluate(
            &rules(&floors, &max_regressions),
            "local",
            None,
            Candidate(&candidate),
            Utc::now(),
            None,
        );

        assert!(!decision.approved());
        assert!(matches!(decision.outcome, Outcome::Park(_)));
    }
}
