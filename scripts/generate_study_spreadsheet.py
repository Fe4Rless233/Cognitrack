"""Generate and export large-scale multi-participant experimental cohort dataset to spreadsheet."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from cognitrack.constants import ALERT_THRESHOLD_PERCENT
from cognitrack.dataset import apply_ground_truth_labels, generate_trial2_cohort, secondary_verification_score
from cognitrack.features import build_feature_frame
from cognitrack.model import BurnoutRiskGauge, CogniTrackTrainer


def build_full_study_spreadsheet(
    n_patients: int = 15,
    output_dir: str | Path = "data",
    fast_mode: bool = False,
    random_state: int = 42,
) -> tuple[Path, Path]:
    """Generate extensive multi-patient repeated-measures study telemetry and export to CSV spreadsheets.

    Outputs:
    1. data/cognitrack_full_study_epochs.csv - ~22,500 rows of 1-second epoch telemetry
    2. data/cognitrack_participant_summary.csv - participant and stage-aggregated metrics
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[CogniTrack] Generating realistic Trial II study cohort with N={n_patients} participants...")

    if fast_mode:
        # Compact durations for rapid testing
        durations = {
            "rest_eyes_closed": 15,
            "rest_eyes_open": 15,
            "practice_n_back": 15,
            "1-back": 30,
            "3-back": 30,
            "mental_math": 20,
            "recovery": 25,
            "final_baseline": 15,
        }
    else:
        # Full experimental protocol matching STS paper draft
        durations = {
            "rest_eyes_closed": 120,       # 2 min
            "rest_eyes_open": 120,         # 2 min
            "practice_n_back": 120,        # 2 min
            "1-back": 300,                 # 5 min
            "3-back": 300,                 # 5 min
            "mental_math": 180,            # 3 min
            "recovery": 240,               # 4 min
            "final_baseline": 120,         # 2 min
        }

    raw_cohort = generate_trial2_cohort(
        n_patients=n_patients,
        stage_durations_sec=durations,
        random_state=random_state,
    )
    print(f"[CogniTrack] Generated {len(raw_cohort):,} raw epoch telemetry observations.")

    # 1. Apply ground truth labels
    labeled_cohort = apply_ground_truth_labels(raw_cohort, task_column="task")

    # 2. Compute secondary verification score (accuracy, RT, TLX, KSS)
    labeled_cohort["secondary_verification_score"] = secondary_verification_score(labeled_cohort)

    # 3. Feature engineering & within-subject baseline z-score normalization
    features = build_feature_frame(
        labeled_cohort,
        patient_column="patient_id",
        task_column="task",
    )

    # Combine data
    merged = labeled_cohort.copy()
    merged["fatigue_index_theta_alpha"] = features["theta_alpha"]
    merged["attentional_deficit_theta_beta"] = features["theta_beta"]
    merged["engagement_index"] = features["engagement"]
    merged["focus_score"] = features["focus"]
    merged["z_theta_alpha"] = features["z_theta_alpha"]
    merged["z_theta_beta"] = features["z_theta_beta"]
    merged["z_focus"] = features["z_focus"]
    merged["z_engagement"] = features["z_engagement"]

    # 4. Train pipeline and compute Burnout Risk Gauge probabilities
    print("[CogniTrack] Fitting model to compute out-of-fold and calibrated risk probabilities...")
    trainer = CogniTrackTrainer(random_state=random_state)
    trained = trainer.train(
        features=features,
        labels=merged["label"],
        patients=merged["patient_id"],
        cv_splits=5,
        enable_nested_cv=False,  # Train final calibrated risk model for spreadsheet
    )

    risk_percentages = BurnoutRiskGauge.risk_percent(trained.logistic_model, features)
    merged["burnout_risk_percent"] = np.round(risk_percentages, 2)
    merged["critical_alert_triggered"] = merged["burnout_risk_percent"] >= ALERT_THRESHOLD_PERCENT
    merged["predicted_tier"] = [BurnoutRiskGauge.classify_operational_tier(r) for r in risk_percentages]
    merged["prescribed_intervention"] = [BurnoutRiskGauge.prescribe_intervention(r) for r in risk_percentages]

    # Signal quality and simulated hardware flags
    rng = np.random.default_rng(random_state)
    # 98% clean telemetry, 2% transient artifact spikes
    artifacts = rng.binomial(1, 0.02, size=len(merged))
    merged["poor_signal_flag"] = artifacts
    merged["artifact_detected"] = artifacts == 1

    # Rename label column for publication clarity
    merged["ground_truth_tier"] = merged["label"]
    merged["is_critical_fatigue"] = (merged["ground_truth_tier"] == 2).astype(int)

    # Column ordering for spreadsheet clarity
    primary_cols = [
        "patient_id",
        "age",
        "gender",
        "grade",
        "condition_group",
        "task",
        "ground_truth_tier",
        "is_critical_fatigue",
        "session_elapsed_sec",
        "stage_elapsed_sec",
        "theta_power",
        "alpha_power",
        "beta_power",
        "fatigue_index_theta_alpha",
        "attentional_deficit_theta_beta",
        "engagement_index",
        "focus_score",
        "z_theta_alpha",
        "z_theta_beta",
        "z_focus",
        "z_engagement",
        "accuracy",
        "reaction_time_ms",
        "nasa_tlx",
        "karolinska_sleepiness_scale",
        "secondary_verification_score",
        "burnout_risk_percent",
        "critical_alert_triggered",
        "predicted_tier",
        "prescribed_intervention",
        "poor_signal_flag",
    ]
    remaining = [c for c in merged.columns if c not in primary_cols and c != "label"]
    final_df = merged[primary_cols + remaining]

    epoch_csv_path = out_path / "cognitrack_full_study_epochs.csv"
    final_df.to_csv(epoch_csv_path, index=False)
    print(f"[CogniTrack] Saved full epoch telemetry to: {epoch_csv_path} ({len(final_df):,} rows)")

    # 5. Build participant & stage summary spreadsheet
    summary_df = final_df.groupby(["patient_id", "condition_group", "task"]).agg(
        epoch_count=("session_elapsed_sec", "count"),
        mean_theta=("theta_power", "mean"),
        mean_alpha=("alpha_power", "mean"),
        mean_beta=("beta_power", "mean"),
        mean_fatigue_index=("fatigue_index_theta_alpha", "mean"),
        mean_z_theta_alpha=("z_theta_alpha", "mean"),
        mean_focus_score=("focus_score", "mean"),
        mean_accuracy=("accuracy", "mean"),
        mean_reaction_time_ms=("reaction_time_ms", "mean"),
        mean_nasa_tlx=("nasa_tlx", "mean"),
        mean_kss=("karolinska_sleepiness_scale", "mean"),
        mean_burnout_risk=("burnout_risk_percent", "mean"),
        critical_alert_rate=("critical_alert_triggered", "mean"),
    ).reset_index()

    summary_csv_path = out_path / "cognitrack_participant_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[CogniTrack] Saved participant summary to: {summary_csv_path} ({len(summary_df):,} rows)")

    return epoch_csv_path, summary_csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CogniTrack study spreadsheet")
    parser.add_argument("--patients", type=int, default=15, help="Number of participants (default 15)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for spreadsheets")
    parser.add_argument("--fast", action="store_true", help="Fast mode with reduced duration for testing")
    args = parser.parse_args()

    build_full_study_spreadsheet(
        n_patients=args.patients,
        output_dir=args.output_dir,
        fast_mode=args.fast,
    )
