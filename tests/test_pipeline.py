import pandas as pd

from cognitrack.constants import ALERT_THRESHOLD_PERCENT, CRITICAL_FATIGUE, LOW_LOAD_REST, MODERATE_WORKLOAD
from cognitrack.dataset import apply_ground_truth_labels
from cognitrack.features import build_feature_frame
from cognitrack.model import BurnoutRiskGauge
from cognitrack.pipeline import CogniTrackPipeline


def _sample_epochs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task": [
                "rest_eyes_closed",
                "rest_eyes_open",
                "1-back",
                "1-back",
                "3-back",
                "mental_math",
            ],
            "theta_power": [2.0, 2.2, 3.0, 3.2, 5.0, 5.2],
            "alpha_power": [3.0, 3.1, 2.7, 2.6, 2.0, 1.8],
            "beta_power": [1.7, 1.8, 2.0, 2.1, 3.4, 3.5],
            "accuracy": [1.0, 0.98, 0.9, 0.87, 0.65, 0.6],
            "reaction_time_ms": [400, 410, 480, 500, 680, 700],
            "nasa_tlx": [15, 20, 45, 50, 85, 90],
        }
    )


def test_ground_truth_task_mapping():
    labeled = apply_ground_truth_labels(_sample_epochs())
    assert labeled["label"].tolist() == [
        LOW_LOAD_REST,
        LOW_LOAD_REST,
        MODERATE_WORKLOAD,
        MODERATE_WORKLOAD,
        CRITICAL_FATIGUE,
        CRITICAL_FATIGUE,
    ]


def test_end_to_end_training_and_grid_contains_required_hyperparameters():
    epochs = pd.concat([_sample_epochs()] * 3, ignore_index=True)
    pipeline = CogniTrackPipeline(random_state=7)
    artifacts = pipeline.train(epochs)

    best = artifacts.models.svm_search.best_params_
    assert best["kernel"] in {"rbf", "linear"}
    assert best["C"] in {0.1, 1.0, 10.0, 100.0}
    assert best["gamma"] in {"scale", "auto", 0.01, 0.1}
    assert best["class_weight"] == "balanced"


def test_risk_gauge_threshold_behavior():
    epochs = pd.concat([_sample_epochs()] * 4, ignore_index=True)
    pipeline = CogniTrackPipeline(random_state=0)
    artifacts = pipeline.train(epochs)

    features = build_feature_frame(_sample_epochs())
    risk = BurnoutRiskGauge.risk_percent(artifacts.models.logistic_model, features)

    assert ((risk >= 0.0) & (risk <= 100.0)).all()
    assert BurnoutRiskGauge.is_critical_alert(ALERT_THRESHOLD_PERCENT)
    assert BurnoutRiskGauge.is_critical_alert(ALERT_THRESHOLD_PERCENT - 0.1) is False
