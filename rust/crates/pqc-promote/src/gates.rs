//! Ported from the reference
//! `promotion-controller/crates/promotion-policy/src/gates.rs`
//! (README.md §5.4) — unchanged logic.
//!
//! Convention (documented in policy.yaml too): every metric named in a
//! policy is assumed **higher is better**. `test_mae` is lower-is-better, so
//! the training pipeline also logs `neg_test_mae`, and the policy gates on
//! that name instead — see `python/config/promotion_policy.yaml`.

use chrono::{DateTime, Duration, Utc};

use crate::compare::{Comparison, MetricPair};
use crate::decision::{Gate, Parked, Rejected};
use crate::policy::Rules;

fn pair<'a>(
    comparison: &'a Comparison,
    metric: &str,
    gate: Gate,
) -> Result<&'a MetricPair, Rejected> {
    comparison
        .metric(metric)
        .ok_or_else(|| Rejected::MetricNotEvaluated {
            metric: metric.to_string(),
            gate,
        })
}

pub(crate) fn floors(rules: &Rules<'_>, comparison: &Comparison) -> Result<(), Rejected> {
    for (metric, floor) in rules.floors {
        let observed = pair(comparison, metric, Gate::AbsoluteFloor)?;

        if observed.candidate < *floor {
            return Err(Rejected::BelowFloor {
                metric: metric.clone(),
                observed: observed.candidate,
                floor: *floor,
            });
        }
    }

    Ok(())
}

pub(crate) fn improvement(rules: &Rules<'_>, comparison: &Comparison) -> Result<(), Rejected> {
    let observed = pair(comparison, rules.primary_metric, Gate::RelativeImprovement)?;
    let gap = observed.gap();

    if gap < rules.min_primary_margin {
        return Err(Rejected::NotEnoughBetter {
            metric: rules.primary_metric.to_string(),
            gap,
            required: rules.min_primary_margin,
        });
    }

    Ok(())
}

pub(crate) fn regressions(rules: &Rules<'_>, comparison: &Comparison) -> Result<(), Rejected> {
    for (metric, allowed) in rules.max_regressions {
        let observed = pair(comparison, metric, Gate::Regression)?;
        let lost = -observed.gap();

        if lost > *allowed {
            return Err(Rejected::GuardedRegression {
                metric: metric.clone(),
                regression: lost,
                allowed: *allowed,
            });
        }
    }

    Ok(())
}

pub(crate) fn age(
    rules: &Rules<'_>,
    comparison: &Comparison,
    at: DateTime<Utc>,
) -> Result<(), Parked> {
    let age_days = (at - comparison.candidate_evaluated_at).num_days();

    if age_days > rules.max_eval_age_days {
        return Err(Parked::EvaluationTooOld {
            age_days,
            max_days: rules.max_eval_age_days,
        });
    }

    Ok(())
}

pub(crate) fn environment_rules(
    rules: &Rules<'_>,
    environment: &str,
    at: DateTime<Utc>,
    last_promoted_at: Option<DateTime<Utc>>,
) -> Result<(), Parked> {
    if let Some(last) = last_promoted_at {
        let until = last + Duration::hours(rules.min_promotion_interval_hours);

        if at < until {
            return Err(Parked::RateLimited {
                environment: environment.to_string(),
                interval_hours: rules.min_promotion_interval_hours,
                until,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::compare::{compare, Candidate, Champion, Eval};
    use crate::policy::OnPass;

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
    fn floors_rejects_below_floor() {
        let champion = eval(&[("f1", 0.60)]);
        let candidate = eval(&[("f1", 0.50)]);
        let comparison = compare(Some(Champion(&champion)), Candidate(&candidate)).unwrap();

        let floor_map: BTreeMap<String, f64> = [("f1".to_string(), 0.55)].into();
        let empty = BTreeMap::new();
        let err = floors(&rules(&floor_map, &empty), &comparison).unwrap_err();
        assert!(matches!(err, Rejected::BelowFloor { .. }));
    }

    #[test]
    fn regressions_allows_small_losses_but_not_large_ones() {
        let champion = eval(&[("f1", 0.70)]);
        let candidate = eval(&[("f1", 0.69)]);
        let comparison = compare(Some(Champion(&champion)), Candidate(&candidate)).unwrap();

        let empty = BTreeMap::new();
        let allowed: BTreeMap<String, f64> = [("f1".to_string(), 0.02)].into();
        assert!(regressions(&rules(&empty, &allowed), &comparison).is_ok());

        let too_strict: BTreeMap<String, f64> = [("f1".to_string(), 0.005)].into();
        assert!(regressions(&rules(&empty, &too_strict), &comparison).is_err());
    }

    #[test]
    fn age_parks_stale_evaluations() {
        let champion = eval(&[("f1", 0.70)]);
        let mut candidate = eval(&[("f1", 0.71)]);
        candidate.evaluated_at = Utc::now() - Duration::days(45);
        let comparison = compare(Some(Champion(&champion)), Candidate(&candidate)).unwrap();

        let empty = BTreeMap::new();
        let err = age(&rules(&empty, &empty), &comparison, Utc::now()).unwrap_err();
        assert!(matches!(err, Parked::EvaluationTooOld { .. }));
    }
}
