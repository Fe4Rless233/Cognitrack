"""Dataset preparation, task ground-truth labeling, and multi-patient cohort generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    ALL_STAGES,
    CRITICAL_FATIGUE,
    LOW_LOAD_REST,
    MODERATE_WORKLOAD,
    RECOVERY_K_BOX_BREATHING,
    RECOVERY_K_PASSIVE,
    STAGE_1_BACK,
    STAGE_3_BACK,
    STAGE_FINAL_BASELINE,
    STAGE_MENTAL_MATH,
    STAGE_PRACTICE_N_BACK,
    STAGE_RECOVERY_BOX_BREATHING,
    STAGE_RECOVERY_PASSIVE,
    STAGE_REST_EYES_CLOSED,
    STAGE_REST_EYES_OPEN,
)

_TASK_TO_LABEL: dict[str, int] = {
    STAGE_REST_EYES_CLOSED: LOW_LOAD_REST,
    STAGE_REST_EYES_OPEN: LOW_LOAD_REST,
    STAGE_PRACTICE_N_BACK: MODERATE_WORKLOAD,
    STAGE_1_BACK: MODERATE_WORKLOAD,
    STAGE_3_BACK: CRITICAL_FATIGUE,
    STAGE_MENTAL_MATH: CRITICAL_FATIGUE,
    STAGE_RECOVERY_BOX_BREATHING: LOW_LOAD_REST,
    STAGE_RECOVERY_PASSIVE: LOW_LOAD_REST,
    STAGE_FINAL_BASELINE: LOW_LOAD_REST,
}


def apply_ground_truth_labels(frame: pd.DataFrame, task_column: str = "task") -> pd.DataFrame:
    """Map task names to CogniTrack workload labels.

    Expects one row per 1-second EEG epoch.
    Tiers:
    0 = Low Load / Rest
    1 = Moderate Workload
    2 = Critical Fatigue / High Load Stress
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
    """Compute a normalized corroboration score from behavior, NASA-TLX, and KSS.

    Higher values (0.0 to 1.0) indicate stronger convergent evidence of fatigue.
    """
    required = ["accuracy", "reaction_time_ms", "nasa_tlx"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing secondary verification columns: {missing}")

    # Fatigue evidence: lower accuracy, slower reaction time, higher TLX
    acc_component = 1.0 - frame["accuracy"].clip(0.0, 1.0)

    rt_min = frame["reaction_time_ms"].min()
    rt_max = frame["reaction_time_ms"].max()
    rt_range = rt_max - rt_min
    if rt_range == 0:
        rt_component = pd.Series(0.0, index=frame.index)
    else:
        rt_component = (frame["reaction_time_ms"] - rt_min) / rt_range

    tlx_component = frame["nasa_tlx"].clip(0.0, 100.0) / 100.0

    components = [acc_component, rt_component, tlx_component]

    # Optional Karolinska Sleepiness Scale (1-9)
    if "karolinska_sleepiness_scale" in frame.columns:
        kss_component = (frame["karolinska_sleepiness_scale"].clip(1.0, 9.0) - 1.0) / 8.0
        components.append(kss_component)

    score = pd.concat(components, axis=1).mean(axis=1)
    return score.clip(0.0, 1.0)


def generate_trial2_cohort(
    n_patients: int = 15,
    stage_durations_sec: dict[str, int] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic repeated-measures experimental cohort dataset (Trial II).

    Adheres to the STS experimental design:
    - N participants (e.g. 15 high-school students, age 16-18)
    - 6 standardized stages:
      1. Baseline: 2 min eyes-closed + 2 min eyes-open
      2. Practice: 2 min n-back
      3. Load manipulation: 5 min 1-back + 5 min 3-back
      4. Stressor: mental arithmetic with visual timer & error prompts
      5. Recovery: randomized to either guided box breathing (4-4-4-4) or passive rest
      6. Final baseline: resting fixation
    - Realistic microvolt spectral band powers showing:
      * Elevated Theta & depressed Alpha in 3-back and math stressor
      * Exponential decay of post-stressor Theta during recovery
        (guided box breathing k ~ 0.045 s^-1 vs passive rest k ~ 0.012 s^-1)
      * Secondary behavioral & subjective markers: accuracy, reaction time, NASA-TLX, KSS
    """
    rng = np.random.default_rng(random_state)

    if stage_durations_sec is None:
        # Default full research paper protocol (seconds per stage)
        stage_durations_sec = {
            STAGE_REST_EYES_CLOSED: 120,       # 2 minutes
            STAGE_REST_EYES_OPEN: 120,         # 2 minutes
            STAGE_PRACTICE_N_BACK: 120,        # 2 minutes
            STAGE_1_BACK: 300,                 # 5 minutes
            STAGE_3_BACK: 300,                 # 5 minutes
            STAGE_MENTAL_MATH: 180,            # 3 minutes
            "recovery": 240,                   # 4 minutes
            STAGE_FINAL_BASELINE: 120,         # 2 minutes
        }

    cohort_records: list[dict] = []

    # Assign participant demographics
    for p_idx in range(1, n_patients + 1):
        patient_id = f"P{p_idx:02d}"
        age = int(rng.choice([16, 17, 18]))
        gender = str(rng.choice(["Female", "Male"]))
        grade = f"{int(rng.choice([11, 12]))}th"

        # Half assigned to guided box breathing, half to passive rest
        recovery_condition = (
            STAGE_RECOVERY_BOX_BREATHING if p_idx % 2 == 1 else STAGE_RECOVERY_PASSIVE
        )
        k_recovery = (
            RECOVERY_K_BOX_BREATHING if recovery_condition == STAGE_RECOVERY_BOX_BREATHING
            else RECOVERY_K_PASSIVE
        )
        # Participant-specific physiological baseline offsets
        p_theta_base = rng.normal(2.5, 0.4)
        p_alpha_base = rng.normal(3.8, 0.5)
        p_beta_base = rng.normal(1.9, 0.3)
        p_rt_base = rng.normal(390, 30)

        elapsed_session_sec = 0

        # Run through protocol stages
        for stage_name, duration_sec in stage_durations_sec.items():
            current_stage = recovery_condition if stage_name == "recovery" else stage_name

            for sec in range(duration_sec):
                t_stage = float(sec)
                # Compute stage dynamics
                if current_stage == STAGE_REST_EYES_CLOSED:
                    theta = p_theta_base * rng.normal(0.95, 0.08)
                    alpha = p_alpha_base * 1.35 * rng.normal(1.0, 0.07)  # high posterior alpha
                    beta = p_beta_base * 0.85 * rng.normal(1.0, 0.08)
                    acc = 1.0
                    rt = p_rt_base
                    tlx = rng.normal(15.0, 3.0)
                    kss = int(np.clip(rng.normal(2.5, 0.5), 1, 9))

                elif current_stage == STAGE_REST_EYES_OPEN:
                    theta = p_theta_base * rng.normal(1.0, 0.08)
                    alpha = p_alpha_base * 1.10 * rng.normal(1.0, 0.07)
                    beta = p_beta_base * 0.95 * rng.normal(1.0, 0.08)
                    acc = 1.0
                    rt = p_rt_base
                    tlx = rng.normal(20.0, 3.0)
                    kss = int(np.clip(rng.normal(3.0, 0.5), 1, 9))

                elif current_stage == STAGE_PRACTICE_N_BACK:
                    theta = p_theta_base * rng.normal(1.25, 0.1)
                    alpha = p_alpha_base * 0.90 * rng.normal(1.0, 0.08)
                    beta = p_beta_base * 1.15 * rng.normal(1.0, 0.09)
                    acc = float(np.clip(rng.normal(0.92, 0.04), 0.5, 1.0))
                    rt = float(p_rt_base * 1.15 + rng.normal(0, 20))
                    tlx = rng.normal(35.0, 5.0)
                    kss = int(np.clip(rng.normal(3.5, 0.5), 1, 9))

                elif current_stage == STAGE_1_BACK:
                    # Moderate cognitive workload
                    theta = p_theta_base * rng.normal(1.40, 0.12)
                    alpha = p_alpha_base * 0.82 * rng.normal(1.0, 0.08)
                    beta = p_beta_base * 1.30 * rng.normal(1.0, 0.10)
                    acc = float(np.clip(rng.normal(0.94, 0.03), 0.7, 1.0))
                    rt = float(p_rt_base * 1.20 + rng.normal(0, 25))
                    tlx = rng.normal(48.0, 6.0)
                    kss = int(np.clip(rng.normal(4.2, 0.6), 1, 9))

                elif current_stage == STAGE_3_BACK:
                    # High working memory load & cognitive fatigue
                    # Gradual accumulation of fatigue across the 5 minutes
                    fatigue_drift = 1.0 + 0.4 * (sec / duration_sec)
                    theta = p_theta_base * 2.25 * fatigue_drift * rng.normal(1.0, 0.14)
                    alpha = p_alpha_base * 0.55 * rng.normal(1.0, 0.09)
                    beta = p_beta_base * 1.70 * rng.normal(1.0, 0.12)
                    acc = float(np.clip(rng.normal(0.72 - 0.1 * (sec / duration_sec), 0.06), 0.4, 0.95))
                    rt = float(p_rt_base * 1.65 + 40 * (sec / duration_sec) + rng.normal(0, 35))
                    tlx = rng.normal(78.0, 6.0)
                    kss = int(np.clip(rng.normal(7.2, 0.7), 1, 9))

                elif current_stage == STAGE_MENTAL_MATH:
                    # Acute anxiety & high stressor (countdown timer + beeps)
                    theta = p_theta_base * 2.60 * rng.normal(1.0, 0.15)
                    alpha = p_alpha_base * 0.48 * rng.normal(1.0, 0.09)
                    beta = p_beta_base * 2.10 * rng.normal(1.0, 0.14)  # High beta from anxiety
                    acc = float(np.clip(rng.normal(0.64, 0.07), 0.3, 0.9))
                    rt = float(p_rt_base * 1.85 + rng.normal(0, 45))
                    tlx = rng.normal(88.0, 5.0)
                    kss = int(np.clip(rng.normal(8.0, 0.6), 1, 9))

                elif current_stage in (STAGE_RECOVERY_BOX_BREATHING, STAGE_RECOVERY_PASSIVE):
                    # Exponential attenuation: P(t) = P0 * exp(-k * t) + P_baseline
                    decay = np.exp(-k_recovery * t_stage)
                    theta_elev = (p_theta_base * 2.60 - p_theta_base) * decay
                    theta = (p_theta_base + theta_elev) * rng.normal(1.0, 0.09)
                    alpha = (p_alpha_base * (1.15 - 0.55 * decay)) * rng.normal(1.0, 0.07)
                    beta = (p_beta_base * (1.0 + 0.9 * decay)) * rng.normal(1.0, 0.08)
                    acc = 1.0
                    rt = p_rt_base
                    tlx = 25.0 + 55.0 * decay
                    kss = int(np.clip(3.5 + 4.0 * decay, 1, 9))

                elif current_stage == STAGE_FINAL_BASELINE:
                    theta = p_theta_base * rng.normal(1.05, 0.08)
                    alpha = p_alpha_base * 1.05 * rng.normal(1.0, 0.07)
                    beta = p_beta_base * 0.98 * rng.normal(1.0, 0.08)
                    acc = 1.0
                    rt = p_rt_base
                    tlx = rng.normal(22.0, 4.0)
                    kss = int(np.clip(rng.normal(3.2, 0.5), 1, 9))

                else:
                    theta, alpha, beta = p_theta_base, p_alpha_base, p_beta_base
                    acc, rt, tlx, kss = 1.0, p_rt_base, 30.0, 3

                # Ensure positive band powers
                theta = max(0.1, float(theta))
                alpha = max(0.1, float(alpha))
                beta = max(0.1, float(beta))

                cohort_records.append(
                    {
                        "patient_id": patient_id,
                        "age": age,
                        "gender": gender,
                        "grade": grade,
                        "task": current_stage,
                        "condition_group": "guided_box_breathing"
                        if recovery_condition == STAGE_RECOVERY_BOX_BREATHING
                        else "passive_rest",
                        "session_elapsed_sec": elapsed_session_sec,
                        "stage_elapsed_sec": int(sec),
                        "theta_power": round(theta, 4),
                        "alpha_power": round(alpha, 4),
                        "beta_power": round(beta, 4),
                        "accuracy": round(acc, 3),
                        "reaction_time_ms": round(rt, 1),
                        "nasa_tlx": round(float(np.clip(tlx, 0.0, 100.0)), 1),
                        "karolinska_sleepiness_scale": int(kss),
                    }
                )
                elapsed_session_sec += 1

    return pd.DataFrame(cohort_records)
