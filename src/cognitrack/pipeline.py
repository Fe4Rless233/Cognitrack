"""End-to-end CogniTrack pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .constants import ALERT_THRESHOLD_PERCENT
from .dataset import apply_ground_truth_labels, secondary_verification_score
from .features import build_feature_frame
from .model import BurnoutRiskGauge, CogniTrackTrainer, TrainedModels


@dataclass
class PipelineArtifacts:
    labeled_data: pd.DataFrame
    features: pd.DataFrame
    models: TrainedModels


class CogniTrackPipeline:
    """Run the full raw-EEG to burnout-risk workflow."""

    def __init__(self, random_state: int = 42):
        self.trainer = CogniTrackTrainer(random_state=random_state)

    def train(self, epochs: pd.DataFrame) -> PipelineArtifacts:
        labeled = apply_ground_truth_labels(epochs)
        if {"accuracy", "reaction_time_ms", "nasa_tlx"}.issubset(labeled.columns):
            labeled["secondary_verification"] = secondary_verification_score(labeled)

        features = build_feature_frame(labeled)
        models = self.trainer.train(features=features, labels=labeled["label"])
        return PipelineArtifacts(labeled_data=labeled, features=features, models=models)

    @staticmethod
    def score_fatigue_risk(models: TrainedModels, feature_rows: pd.DataFrame) -> pd.DataFrame:
        risk = BurnoutRiskGauge.risk_percent(models.logistic_model, feature_rows)
        result = pd.DataFrame({"risk_percent": risk})
        result["critical_alert"] = result["risk_percent"] >= ALERT_THRESHOLD_PERCENT
        return result
