"""CogniTrack: EEG cognitive fatigue modeling, signal processing, and risk monitoring."""

from .constants import (
    ALERT_THRESHOLD_PERCENT,
    CRITICAL_FATIGUE,
    LOW_LOAD_REST,
    MODERATE_WORKLOAD,
)
from .dataset import (
    apply_ground_truth_labels,
    generate_trial2_cohort,
    secondary_verification_score,
)
from .dsp import CogniTrackDSP, simulate_raw_eeg
from .features import build_feature_frame, compute_spectral_ratios
from .model import (
    BurnoutRiskGauge,
    CogniTrackTrainer,
    NestedCVReport,
    TrainedModels,
)
from .pipeline import CogniTrackPipeline, PipelineArtifacts
from .stats import (
    bonferroni_post_hoc,
    fit_exponential_recovery,
    repeated_measures_anova,
)

__all__ = [
    "CogniTrackPipeline",
    "PipelineArtifacts",
    "CogniTrackTrainer",
    "BurnoutRiskGauge",
    "NestedCVReport",
    "TrainedModels",
    "CogniTrackDSP",
    "simulate_raw_eeg",
    "build_feature_frame",
    "compute_spectral_ratios",
    "apply_ground_truth_labels",
    "generate_trial2_cohort",
    "secondary_verification_score",
    "repeated_measures_anova",
    "bonferroni_post_hoc",
    "fit_exponential_recovery",
    "LOW_LOAD_REST",
    "MODERATE_WORKLOAD",
    "CRITICAL_FATIGUE",
    "ALERT_THRESHOLD_PERCENT",
]
