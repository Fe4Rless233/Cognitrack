"""Unit tests for CogniTrack digital signal processing (DSP) pipeline."""

import numpy as np
import pytest

from cognitrack.constants import (
    ALPHA_BAND,
    ARTIFACT_THRESHOLD_UV,
    BETA_BAND,
    SAMPLING_RATE_HZ,
    THETA_BAND,
)
from cognitrack.dsp import CogniTrackDSP, simulate_raw_eeg


def test_simulate_raw_eeg_shape_and_range():
    fs = 512
    duration = 2.0
    signal_uv = simulate_raw_eeg(duration_sec=duration, sampling_rate=fs, state="baseline", random_state=42)
    assert len(signal_uv) == int(duration * fs)
    assert np.all(np.isfinite(signal_uv))
    # Signals without injected blink artifacts should stay well within +/- 100 uV
    assert np.max(np.abs(signal_uv)) < ARTIFACT_THRESHOLD_UV


def test_artifact_spike_detection():
    dsp = CogniTrackDSP(sampling_rate=512)
    # Clean epoch
    clean_epoch = np.sin(np.linspace(0, 10, 512)) * 15.0
    feat_clean = dsp.process_epoch(clean_epoch)
    assert feat_clean.is_artifact is False

    # Artifact epoch with > 100 uV spike
    artifact_epoch = clean_epoch.copy()
    artifact_epoch[200] = 135.0
    feat_art = dsp.process_epoch(artifact_epoch)
    assert feat_art.is_artifact is True
    assert feat_art.peak_uv == 135.0


def test_notch_and_bandpass_filtering():
    dsp = CogniTrackDSP(sampling_rate=512)
    t = np.linspace(0, 1.0, 512, endpoint=False)

    # 60 Hz hum (should be significantly attenuated by notch filter)
    hum_60hz = 20.0 * np.sin(2 * np.pi * 60.0 * t)
    # 10 Hz alpha wave (should be preserved by bandpass filter)
    alpha_10hz = 15.0 * np.sin(2 * np.pi * 10.0 * t)

    combined = hum_60hz + alpha_10hz
    filtered = dsp.filter_signal(combined)

    # Compute FFT power at 60 Hz before and after
    freqs_raw, psd_raw = dsp.compute_psd(combined)
    freqs_filt, psd_filt = dsp.compute_psd(filtered)

    idx_60_raw = np.argmin(np.abs(freqs_raw - 60.0))
    idx_60_filt = np.argmin(np.abs(freqs_filt - 60.0))

    # Power at 60 Hz should drop substantially
    assert psd_filt[idx_60_filt] < psd_raw[idx_60_raw] * 0.15


def test_canonical_band_power_extraction():
    dsp = CogniTrackDSP(sampling_rate=512)
    t = np.linspace(0, 2.0, 1024, endpoint=False)

    # Create signal dominant in theta band (6 Hz)
    theta_dominant = 20.0 * np.sin(2 * np.pi * 6.0 * t)
    feat = dsp.process_epoch(theta_dominant[:512])

    assert feat.theta_power > feat.alpha_power
    assert feat.theta_power > feat.beta_power
