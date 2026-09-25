"""Spectral feature engineering and within-subject baseline normalization for CogniTrack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import STAGE_REST_EYES_CLOSED, STAGE_REST_EYES_OPEN


def compute_spectral_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute standard CogniTrack cognitive workload quotients from band powers.

    Formulas:
    1. Fatigue Index = theta / alpha
    2. Attentional Deficit Quotient = theta / beta
    3. Engagement Index = beta / (alpha + theta)
    4. Focus Score = beta / (theta + alpha)
    """
    required = ["theta_power", "alpha_power", "beta_power"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing spectral columns: {missing}")

    theta = frame["theta_power"].astype(float)
    alpha = frame["alpha_power"].astype(float)
    beta = frame["beta_power"].astype(float)

    eps = 1e-9
    ratios = pd.DataFrame(
        {
            "theta_alpha": theta / (alpha + eps),
            "theta_beta": theta / (beta + eps),
            "engagement": beta / (alpha + theta + eps),
            "focus": beta / (theta + alpha + eps),
        },
        index=frame.index,
    )
    return ratios


def normalize_within_patient_baseline(
    ratios: pd.DataFrame,
    frame: pd.DataFrame,
    patient_col: str = "patient_id",
    task_col: str = "task",
) -> pd.DataFrame:
    """Standardize feature ratios per participant relative to their initial 4-minute baseline rest.

    Formula: z = (ratio - mean_baseline) / std_baseline
    """
    z_df = pd.DataFrame(index=ratios.index)
    patients = frame[patient_col].unique()

    for pid in patients:
        p_mask = frame[patient_col] == pid
        p_ratios = ratios.loc[p_mask]

        # Identify baseline epochs (eyes closed & eyes open rest)
        if task_col in frame.columns:
            base_mask = p_mask & frame[task_col].isin([STAGE_REST_EYES_CLOSED, STAGE_REST_EYES_OPEN])
        else:
            base_mask = pd.Series(False, index=frame.index)

        if base_mask.sum() >= 2:
            base_ratios = ratios.loc[base_mask]
            mean = base_ratios.mean()
            std = base_ratios.std(ddof=0).replace(0, np.nan)
        else:
            # Fallback to subject overall mean/std
            mean = p_ratios.mean()
            std = p_ratios.std(ddof=0).replace(0, np.nan)

        p_z = (p_ratios - mean) / std
        p_z = p_z.fillna(0.0)
        z_df.loc[p_mask, ["z_theta_alpha", "z_theta_beta", "z_engagement", "z_focus"]] = p_z[
            ["theta_alpha", "theta_beta", "engagement", "focus"]
        ].values

    return z_df


def build_feature_frame(
    frame: pd.DataFrame,
    patient_column: str = "patient_id",
    task_column: str = "task",
) -> pd.DataFrame:
    """Build full CogniTrack feature vector for each 1-second epoch.

    Required columns: theta_power, alpha_power, beta_power.
    If patient_id is present, uses within-subject baseline normalization.
    Returns DataFrame with canonical features: z_theta_alpha, z_theta_beta, z_focus, z_engagement.
    """
    ratios = compute_spectral_ratios(frame)

    if patient_column in frame.columns and frame[patient_column].nunique() > 0:
        z_features = normalize_within_patient_baseline(
            ratios=ratios,
            frame=frame,
            patient_col=patient_column,
            task_col=task_column,
        )
    else:
        # Standard cohort or single-subject global normalization
        mean = ratios.mean()
        std = ratios.std(ddof=0).replace(0, np.nan)
        zscored = (ratios - mean) / std
        zscored = zscored.fillna(0.0)
        z_features = zscored.rename(
            columns={
                "theta_alpha": "z_theta_alpha",
                "theta_beta": "z_theta_beta",
                "focus": "z_focus",
                "engagement": "z_engagement",
            }
        )

    # Attach raw ratios for reference and analysis
    result = z_features.copy()
    result["theta_alpha"] = ratios["theta_alpha"]
    result["theta_beta"] = ratios["theta_beta"]
    result["engagement"] = ratios["engagement"]
    result["focus"] = ratios["focus"]

    return result
