"""Label, stage, hardware, and algorithmic constants used across CogniTrack."""

from __future__ import annotations

# Operational Tiers / Ground Truth Labels
LOW_LOAD_REST = 0
MODERATE_WORKLOAD = 1
CRITICAL_FATIGUE = 2

# Burnout Risk Gauge Threshold (derived via ROC & Youden's J statistic)
ALERT_THRESHOLD_PERCENT = 81.0

# Canonical EEG Frequency Bands (Hz)
THETA_BAND = (4.0, 8.0)    # Drowsiness and cognitive fatigue
ALPHA_BAND = (8.0, 13.0)   # Relaxed wakefulness and attentional gating
BETA_BAND = (13.0, 30.0)   # Active mental processing and engagement

# Hardware & Sampling Telemetry
SAMPLING_RATE_HZ = 512
ADC_RESOLUTION_BITS = 12
UART_BAUD_RATE = 57600
MIN_VOLTAGE_UV = 0.5
MAX_VOLTAGE_UV = 100.0

# Digital Signal Processing (DSP) & Artifact Filtering
BANDPASS_LOW_CUT_HZ = 0.5
BANDPASS_HIGH_CUT_HZ = 30.0
BANDPASS_ORDER = 4
NOTCH_FREQ_HZ = 60.0
NOTCH_Q_FACTOR = 30.0
ARTIFACT_THRESHOLD_UV = 100.0   # Epochs with absolute voltage > 100 uV dropped

# Sliding Window Parameters
EPOCH_WINDOW_SEC = 1.0
EPOCH_STEP_SEC = 0.5            # 50% overlap
HAMMING_ALPHA = 0.54

# Standardized Experimental Protocol Stages (Trial II)
STAGE_REST_EYES_CLOSED = "rest_eyes_closed"
STAGE_REST_EYES_OPEN = "rest_eyes_open"
STAGE_PRACTICE_N_BACK = "practice_n_back"
STAGE_1_BACK = "1-back"
STAGE_3_BACK = "3-back"
STAGE_MENTAL_MATH = "mental_math"
STAGE_RECOVERY_BOX_BREATHING = "recovery_box_breathing"
STAGE_RECOVERY_PASSIVE = "recovery_passive"
STAGE_FINAL_BASELINE = "final_baseline"

ALL_STAGES = [
    STAGE_REST_EYES_CLOSED,
    STAGE_REST_EYES_OPEN,
    STAGE_PRACTICE_N_BACK,
    STAGE_1_BACK,
    STAGE_3_BACK,
    STAGE_MENTAL_MATH,
    STAGE_RECOVERY_BOX_BREATHING,
    STAGE_RECOVERY_PASSIVE,
    STAGE_FINAL_BASELINE,
]

# Exponential Recovery Decay Constants (s^-1): P(t) = P0 * exp(-k * t)
RECOVERY_K_BOX_BREATHING = 0.045   # Rapid attenuation under guided 4-4-4-4 pacing
RECOVERY_K_PASSIVE = 0.012         # Slower decay under passive sitting rest

# Intervention Prescriptions
INTERVENTION_NONE = "Optimal / Baseline — No action required"
INTERVENTION_MODERATE = "Moderate Cognitive Dip — Prompt hydration break, posture check, and ambient lighting adjustment"
INTERVENTION_CRITICAL = "Critical Fatigue Overload (>=81%) — Mandate immediate 4-4-4-4 Box Breathing reset and 15-min nature walk"

