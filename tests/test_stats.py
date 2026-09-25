"""Unit tests for CogniTrack statistical evaluation pipeline."""

import numpy as np
import pandas as pd
import pytest

from cognitrack.dataset import generate_trial2_cohort
from cognitrack.stats import (
    bonferroni_post_hoc,
    compute_roc_youden_threshold,
    fit_exponential_recovery,
    repeated_measures_anova,
)


def test_repeated_measures_anova_and_greenhouse_geisser():
    # Generate small cohort
    cohort = generate_trial2_cohort(
        n_patients=6,
        stage_durations_sec={
            "rest_eyes_closed": 10,
            "1-back": 10,
            "3-back": 10,
            "mental_math": 10,
        },
        random_state=42,
    )
    cohort["theta_alpha"] = cohort["theta_power"] / cohort["alpha_power"]

    res = repeated_measures_anova(
        cohort,
        patient_col="patient_id",
        condition_col="task",
        value_col="theta_alpha",
        conditions=["rest_eyes_closed", "1-back", "3-back", "mental_math"],
    )

    assert res.f_statistic > 0.0
    assert 0.0 <= res.p_value <= 1.0
    assert 0.0 <= res.greenhouse_geisser_epsilon <= 1.0
    assert 0.0 <= res.corrected_p_value <= 1.0
    assert res.eta_squared > 0.0


def test_bonferroni_post_hoc_adjustments():
    cohort = generate_trial2_cohort(
        n_patients=6,
        stage_durations_sec={
            "rest_eyes_closed": 10,
            "1-back": 10,
            "3-back": 10,
        },
        random_state=42,
    )
    cohort["theta_alpha"] = cohort["theta_power"] / cohort["alpha_power"]

    comparisons = [
        ("rest_eyes_closed", "1-back"),
        ("1-back", "3-back"),
    ]
    post_hocs = bonferroni_post_hoc(
        cohort,
        patient_col="patient_id",
        condition_col="task",
        value_col="theta_alpha",
        comparisons=comparisons,
    )

    assert len(post_hocs) == 2
    for ph in post_hocs:
        assert ph.p_bonferroni >= ph.p_raw
        assert ph.p_bonferroni <= 1.0


def test_exponential_recovery_curve_fitting():
    # Synthesize clean exponential decay: P(t) = 10 * exp(-0.045 * t)
    t = np.linspace(0, 100, 101)
    y = 10.0 * np.exp(-0.045 * t) + np.random.normal(0, 0.05, 101)

    fit = fit_exponential_recovery(t, y, condition_label="Guided Box Breathing")

    assert np.isclose(fit.p0, 10.0, atol=1.0)
    assert np.isclose(fit.k, 0.045, atol=0.015)
    assert fit.half_life_sec > 0.0
    assert fit.r_squared > 0.8


def test_roc_and_youden_j_threshold_optimization():
    y_true = np.array([0] * 50 + [1] * 50)
    # Probabilities well separated with some overlap
    probs = np.array([0.1 + 0.3 * np.random.rand() for _ in range(50)] + [0.7 + 0.25 * np.random.rand() for _ in range(50)])

    res = compute_roc_youden_threshold(y_true, probs)
    assert res["auc"] > 0.8
    assert res["max_youden_j"] > 0.0
    assert 0.0 <= res["optimal_threshold_percent"] <= 100.0
