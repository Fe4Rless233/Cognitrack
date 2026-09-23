"""Dataset preparation and task-ground-truth labeling for CogniTrack."""

from __future__ import annotations

import pandas as pd

from .constants import CRITICAL_FATIGUE, LOW_LOAD_REST, MODERATE_WORKLOAD

_TASK_TO_LABEL = {
    "rest_eyes_closed": LOW_LOAD_REST,
    "rest_eyes_open": LOW_LOAD_REST,
    "1-back": MODERATE_WORKLOAD,
    "3-back": CRITICAL_FATIGUE,
    "mental_math": CRITICAL_FATIGUE,
}


def apply_ground_truth_labels(frame: pd.DataFrame, task_column: str = "task") -> pd.DataFrame:
    """Map task names to CogniTrack workload labels.

    Expects one row per 1-second EEG epoch.
    """
    if task_column not in frame.columns:
        raise ValueError(f"Missing required task column: {task_column}")

    labeled = frame.copy()
    labeled["label"] = labeled[task_column].map(_TASK_TO_LABEL)
    if labeled["label"].isna().any():
        unknown = sorted(labeled.loc[labeled["label"].isna(), task_column].unique())
        raise ValueError(f"Unknown task values for label mapping: {unknown}")

    labeled["label"] = labeled["label"].astype(int)
    return labeled


def secondary_verification_score(frame: pd.DataFrame) -> pd.Series:
    """Compute a normalized corroboration score from behavior and NASA-TLX.

    Higher values indicate stronger evidence of fatigue.
    """
    required = ["accuracy", "reaction_time_ms", "nasa_tlx"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing secondary verification columns: {missing}")

    # fatigue evidence: lower accuracy, slower reaction time, higher TLX
    acc_component = 1.0 - frame["accuracy"].clip(0.0, 1.0)
    rt_component = (frame["reaction_time_ms"] - frame["reaction_time_ms"].min())
    rt_range = (frame["reaction_time_ms"].max() - frame["reaction_time_ms"].min())
    if rt_range == 0:
        rt_component = rt_component * 0.0
    else:
        rt_component = rt_component / rt_range
    tlx_component = frame["nasa_tlx"].clip(0.0, 100.0) / 100.0

    return (acc_component + rt_component + tlx_component) / 3.0
