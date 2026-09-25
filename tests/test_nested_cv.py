"""Unit tests for patient-grouped nested cross-validation in CogniTrack."""

import numpy as np
import pandas as pd
import pytest

from cognitrack.constants import ALERT_THRESHOLD_PERCENT
from cognitrack.dataset import apply_ground_truth_labels, generate_trial2_cohort
from cognitrack.features import build_feature_frame
from cognitrack.model import BurnoutRiskGauge, CogniTrackTrainer
from cognitrack.pipeline import CogniTrackPipeline


def test_patient_grouping_invariant_in_nested_cv():
    """Verify that all data from one patient stays strictly together in train or test fold."""
    # Generate compact cohort with 6 participants
    durations = {
        "rest_eyes_closed": 10,
        "rest_eyes_open": 10,
        "1-back": 15,
        "3-back": 15,
        "mental_math": 10,
    }
    raw = generate_trial2_cohort(n_patients=6, stage_durations_sec=durations, random_state=42)
    labeled = apply_ground_truth_labels(raw)
    features = build_feature_frame(labeled, patient_column="patient_id", task_column="task")

    trainer = CogniTrackTrainer(random_state=42)
    report = trainer.run_nested_group_cv(
        features=features,
        labels=labeled["label"],
        patients=labeled["patient_id"],
        outer_splits=3,
        inner_splits=2,
        model_type="svm",
    )

    assert report.n_outer_folds == 3
    assert report.n_patients == 6

    # Verify patient grouping invariant for every outer fold
    for fold in report.fold_results:
        train_set = set(fold.train_patients)
        test_set = set(fold.test_patients)

        # Invariant 1: Disjoint sets of patients
        assert train_set.isdisjoint(test_set), f"Data leakage detected! Overlapping patients: {train_set & test_set}"
        # Invariant 2: Union of patients covers full cohort
        assert train_set | test_set == set(labeled["patient_id"].unique())
        # Invariant 3: Performance metrics are properly bounded
        assert 0.0 <= fold.accuracy <= 1.0
        assert 0.0 <= fold.balanced_accuracy <= 1.0
        assert 0.0 <= fold.f1 <= 1.0
        assert 0.0 <= fold.roc_auc <= 1.0

    # Aggregate metrics
    assert 0.0 <= report.mean_accuracy <= 1.0
    assert 0.0 <= report.mean_f1 <= 1.0
    assert report.confusion_matrix.shape == (2, 2)


def test_pipeline_with_grouped_nested_cv_and_risk_gauge():
    """Verify end-to-end pipeline training with patient grouping and risk scoring."""
    durations = {
        "rest_eyes_closed": 10,
        "1-back": 10,
        "3-back": 10,
    }
    raw = generate_trial2_cohort(n_patients=4, stage_durations_sec=durations, random_state=123)
    pipeline = CogniTrackPipeline(random_state=123)
    artifacts = pipeline.train(raw, cv_splits=2, enable_nested_cv=True)

    assert artifacts.models.nested_cv_report is not None
    assert artifacts.models.nested_cv_report.n_outer_folds == 2

    # Score fatigue risk
    scored = pipeline.score_fatigue_risk(artifacts.models, artifacts.features)
    assert "risk_percent" in scored.columns
    assert "critical_alert" in scored.columns
    assert "predicted_tier" in scored.columns
    assert "prescribed_intervention" in scored.columns
    assert ((scored["risk_percent"] >= 0.0) & (scored["risk_percent"] <= 100.0)).all()
