//! CLI entry point for a promotion check.
//!
//!     pqc-promote --environment local --deployment pqc-quantum-risk
//!
//! Fetches the `champion` and `candidate` MLflow model versions for the
//! given registered model, pulls their metrics off the runs that produced
//! them, runs the gate (`pqc_promote::evaluate`), and on approval moves the
//! `champion` alias to the candidate's version. On reject/park it prints
//! every reason and exits non-zero — nothing is promoted silently. See
//! README.md §7.7.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use chrono::Utc;

use pqc_promote::{
    evaluate, Candidate, Champion, Decision, Eval, MlflowClient, ModelVersionRef, Outcome, Policy,
};

const CANDIDATE_ALIAS: &str = "candidate";
const CHAMPION_ALIAS: &str = "champion";
const DEFAULT_POLICY_PATH: &str = "../python/config/promotion_policy.yaml";

struct Args {
    environment: String,
    deployment: String,
    policy_path: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut environment = None;
    let mut deployment = None;
    let mut policy_path = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--environment" => environment = args.next(),
            "--deployment" => deployment = args.next(),
            "--policy" => policy_path = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        environment: environment.ok_or("--environment is required")?,
        deployment: deployment.ok_or("--deployment is required")?,
        policy_path: policy_path.unwrap_or_else(|| PathBuf::from(DEFAULT_POLICY_PATH)),
    })
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!(
                "usage: pqc-promote --environment <env> --deployment <mlflow-model-name> [--policy <path>]"
            );
            return ExitCode::FAILURE;
        }
    };

    match run(args).await {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

async fn run(args: Args) -> Result<bool, String> {
    let tracking_uri =
        env::var("MLFLOW_TRACKING_URI").unwrap_or_else(|_| "http://localhost:5000".to_string());
    let client = MlflowClient::new(tracking_uri);

    let policy = Policy::from_yaml_path(&args.policy_path).map_err(|error| error.to_string())?;
    let rules = policy
        .rules(&args.environment, &args.deployment)
        .map_err(|error| error.to_string())?;

    let candidate_version = client
        .model_version_by_alias(&args.deployment, CANDIDATE_ALIAS)
        .await
        .map_err(|error| error.to_string())?
        .ok_or_else(|| {
            format!(
                "no `{CANDIDATE_ALIAS}` alias set on `{}` — nothing to promote. \
                 did the training pipeline finish and register a version?",
                args.deployment
            )
        })?;

    let candidate_eval = resolve_eval(&client, &candidate_version).await?;

    let champion_version = client
        .model_version_by_alias(&args.deployment, CHAMPION_ALIAS)
        .await
        .map_err(|error| error.to_string())?;

    let decision = match champion_version {
        None => {
            // Bootstrap case: nothing has ever been promoted for this
            // deployment. There is nothing to compare against, so the first
            // candidate is approved automatically rather than parked
            // forever by `NotComparable::MissingEval` — see compare.rs.
            println!(
                "no `{CHAMPION_ALIAS}` alias exists yet for `{}` — treating this as the first promotion.",
                args.deployment
            );
            Decision {
                outcome: Outcome::Approve,
                comparison: None,
            }
        }
        Some(champion_version) => {
            let champion_eval = resolve_eval(&client, &champion_version).await?;
            evaluate(
                &rules,
                &args.environment,
                Some(Champion(&champion_eval)),
                Candidate(&candidate_eval),
                Utc::now(),
                None, // TODO: track last_promoted_at once something persists promotion history
            )
        }
    };

    print_decision(&decision);

    if !decision.approved() {
        return Ok(false);
    }

    client
        .set_registered_model_alias(&args.deployment, CHAMPION_ALIAS, &candidate_version.version)
        .await
        .map_err(|error| error.to_string())?;

    println!(
        "promoted `{}` version {} to `{CHAMPION_ALIAS}`",
        args.deployment, candidate_version.version
    );
    println!(
        "remember to restart/reload pqc-inference so it picks up the new alias (README §7.7)."
    );
    Ok(true)
}

async fn resolve_eval(client: &MlflowClient, version: &ModelVersionRef) -> Result<Eval, String> {
    let run_id = version
        .run_id
        .clone()
        .ok_or_else(|| format!("mlflow model version {} has no run_id", version.version))?;

    let run = client
        .get_run(&run_id)
        .await
        .map_err(|error| error.to_string())?;

    Ok(Eval {
        dataset_id: run.dataset_id().unwrap_or("unknown").to_string(),
        evaluated_at: run.finished_at.unwrap_or_else(Utc::now),
        metrics: run.metrics,
    })
}

fn print_decision(decision: &Decision) {
    match &decision.outcome {
        Outcome::Approve => println!("decision: approve"),
        Outcome::Reject(reason) => println!("decision: reject\n  - {reason}"),
        Outcome::Park(reason) => println!("decision: park\n  - {reason}"),
    }

    if let Some(comparison) = &decision.comparison {
        println!("metrics compared:");
        for (name, pair) in comparison.metrics() {
            println!(
                "  {name}: champion={:.4} candidate={:.4} gap={:+.4}",
                pair.champion,
                pair.candidate,
                pair.gap()
            );
        }
    }
}
