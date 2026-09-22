"""Spectral feature engineering for CogniTrack."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build CogniTrack feature vector for each 1-second epoch.

    Required columns: theta_power, alpha_power, beta_power.
    Produces z(theta/alpha), z(theta/beta), z(beta/(theta+alpha)).
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
            "focus": beta / (theta + alpha + eps),
        },
        index=frame.index,
    )

    mean = ratios.mean()
    std = ratios.std(ddof=0).replace(0, np.nan)
    zscored = (ratios - mean) / std
    zscored = zscored.fillna(0.0)

    return zscored.rename(
        columns={
            "theta_alpha": "z_theta_alpha",
            "theta_beta": "z_theta_beta",
            "focus": "z_focus",
        }
    )
