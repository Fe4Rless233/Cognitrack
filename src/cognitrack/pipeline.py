"""End-to-end CogniTrack pipeline orchestrating DSP features, modeling, and risk scoring."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .constants import ALERT_THRESHOLD_PERCENT
from .dataset import apply_ground_truth_labels, secondary_verification_score
from .features import build_feature_frame
from .model import BurnoutRiskGauge, CogniTrackTrainer, TrainedModels


@dataclass
class PipelineArtifacts:
    """Artifacts output by CogniTrack end-to-end training."""
    labeled_data: pd.DataFrame
    features: pd.DataFrame
    models: TrainedModels


class CogniTrackPipeline:
    """Run the complete EEG cognitive fatigue monitoring workflow.

    Supports:
    - Multi-participant cohort studies with patient-grouped nested cross validation
    - Within-subject baseline z-score normalization
    - Probabilistic Burnout Risk Gauge with validated 81% critical cutoff
    - Data-driven closed-loop recovery interventions
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.trainer = CogniTrackTrainer(random_state=random_state)

    def train(
        self,
        epochs: pd.DataFrame,
        cv_splits: int = 5,
        enable_nested_cv: bool = True,
    ) -> PipelineArtifacts:
        """Fit models on epoch telemetry and optionally run patient-grouped nested CV."""
        labeled = apply_ground_truth_labels(epochs)

        if {"accuracy", "reaction_time_ms", "nasa_tlx"}.issubset(labeled.columns):
            labeled["secondary_verification"] = secondary_verification_score(labeled)

        # Build spectral ratio features and apply baseline normalization if patient_id exists
        features = build_feature_frame(labeled)

        # Extract patient identifiers if present
        patients = labeled["patient_id"] if "patient_id" in labeled.columns else None

        models = self.trainer.train(
            features=features,
            labels=labeled["label"],
            patients=patients,
            cv_splits=cv_splits,
            enable_nested_cv=enable_nested_cv,
        )

        return PipelineArtifacts(labeled_data=labeled, features=features, models=models)

    @staticmethod
    def score_fatigue_risk(models: TrainedModels, feature_rows: pd.DataFrame) -> pd.DataFrame:
        """Score incoming epochs with Burnout Risk Gauge and prescribe recovery protocols."""
        risk = BurnoutRiskGauge.risk_percent(models.logistic_model, feature_rows)
        result = pd.DataFrame({"risk_percent": risk}, index=feature_rows.index)
        result["critical_alert"] = result["risk_percent"] >= ALERT_THRESHOLD_PERCENT
        result["predicted_tier"] = [BurnoutRiskGauge.classify_operational_tier(r) for r in risk]
        result["prescribed_intervention"] = [BurnoutRiskGauge.prescribe_intervention(r) for r in risk]
        return result
