"""Digital Signal Processing (DSP) pipeline for CogniTrack EEG telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import signal

from .constants import (
    ALPHA_BAND,
    ARTIFACT_THRESHOLD_UV,
    BANDPASS_HIGH_CUT_HZ,
    BANDPASS_LOW_CUT_HZ,
    BANDPASS_ORDER,
    BETA_BAND,
    EPOCH_STEP_SEC,
    EPOCH_WINDOW_SEC,
    NOTCH_FREQ_HZ,
    NOTCH_Q_FACTOR,
    SAMPLING_RATE_HZ,
    THETA_BAND,
)


@dataclass
class EpochSpectralFeatures:
    """Spectral analysis output for a single EEG epoch."""
    epoch_index: int
    timestamp_sec: float
    is_artifact: bool
    theta_power: float
    alpha_power: float
    beta_power: float
    peak_uv: float
    mean_uv: float


class CogniTrackDSP:
    """Implements multi-stage hardware-matched digital signal processing.

    Pipeline:
    1. 4th-order Butterworth bandpass (0.5 - 30 Hz)
    2. 60 Hz IIR digital notch filter
    3. Peak artifact rejection (> +/- 100 uV)
    4. Hamming windowed sliding Fourier Transform (FFT)
    5. Power Spectral Density (PSD) integration over Theta, Alpha, and Beta bands
    """

    def __init__(self, sampling_rate: int = SAMPLING_RATE_HZ) -> None:
        self.fs = sampling_rate
        # 4th-order Butterworth bandpass filter
        self.bp_sos = signal.butter(
            N=BANDPASS_ORDER,
            Wn=[BANDPASS_LOW_CUT_HZ, BANDPASS_HIGH_CUT_HZ],
            btype="bandpass",
            fs=self.fs,
            output="sos",
        )
        # 60 Hz Notch filter
        self.notch_b, self.notch_a = signal.iirnotch(
            w0=NOTCH_FREQ_HZ,
            Q=NOTCH_Q_FACTOR,
            fs=self.fs,
        )

    def filter_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """Apply Butterworth bandpass and 60 Hz notch filter to time-domain signal."""
        if len(raw_signal) == 0:
            return np.array([], dtype=float)
        # Apply bandpass filter
        filtered = signal.sosfiltfilt(self.bp_sos, raw_signal)
        # Apply notch filter
        filtered = signal.filtfilt(self.notch_b, self.notch_a, filtered)
        return filtered

    def compute_psd(
        self,
        epoch_signal: np.ndarray,
        apply_hamming: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute single-sided PSD using discrete FFT with Hamming window.

        Formula: X[k] = sum_{n=0}^{N-1} x[n] * w[n] * exp(-j * 2 * pi * k * n / N)
        """
        n = len(epoch_signal)
        if n == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        if apply_hamming:
            window = np.hamming(n)
            windowed = epoch_signal * window
        else:
            windowed = epoch_signal

        # FFT
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        # Power spectral density
        psd = (np.abs(fft_vals) ** 2) / (n * self.fs)
        if len(psd) > 1:
            psd[1:-1] *= 2.0  # Conserve power for single-sided spectrum

        return freqs, psd

    @staticmethod
    def band_power(
        freqs: np.ndarray,
        psd: np.ndarray,
        band: tuple[float, float],
    ) -> float:
        """Integrate Power Spectral Density within canonical frequency bounds."""
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        if not np.any(idx):
            return 0.0
        # Trapezoidal numerical integration
        return float(np.trapezoid(psd[idx], freqs[idx])) if hasattr(np, "trapezoid") else float(np.trapz(psd[idx], freqs[idx]))

    def process_epoch(
        self,
        raw_epoch: np.ndarray,
        epoch_idx: int = 0,
        timestamp_sec: float = 0.0,
    ) -> EpochSpectralFeatures:
        """Filter, check artifacts, and extract Theta/Alpha/Beta powers from an epoch."""
        peak = float(np.max(np.abs(raw_epoch))) if len(raw_epoch) > 0 else 0.0
        mean = float(np.mean(raw_epoch)) if len(raw_epoch) > 0 else 0.0

        # Artifact check: drop/flag epochs exceeding +/- 100 uV (e.g. jaw clenching, ocular blinks)
        is_artifact = peak > ARTIFACT_THRESHOLD_UV

        filtered = self.filter_signal(raw_epoch)
        freqs, psd = self.compute_psd(filtered, apply_hamming=True)

        theta = self.band_power(freqs, psd, THETA_BAND)
        alpha = self.band_power(freqs, psd, ALPHA_BAND)
        beta = self.band_power(freqs, psd, BETA_BAND)

        return EpochSpectralFeatures(
            epoch_index=epoch_idx,
            timestamp_sec=timestamp_sec,
            is_artifact=is_artifact,
            theta_power=theta,
            alpha_power=alpha,
            beta_power=beta,
            peak_uv=peak,
            mean_uv=mean,
        )

    def process_continuous_stream(
        self,
        raw_stream: np.ndarray,
        window_sec: float = EPOCH_WINDOW_SEC,
        step_sec: float = EPOCH_STEP_SEC,
    ) -> pd.DataFrame:
        """Process continuous 512 Hz telemetry using overlapping sliding windows."""
        window_samples = int(window_sec * self.fs)
        step_samples = int(step_sec * self.fs)

        records = []
        n_samples = len(raw_stream)
        epoch_idx = 0

        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            segment = raw_stream[start:end]
            ts = start / self.fs
            feat = self.process_epoch(segment, epoch_idx=epoch_idx, timestamp_sec=ts)
            records.append(
                {
                    "epoch_index": feat.epoch_index,
                    "timestamp_sec": feat.timestamp_sec,
                    "is_artifact": feat.is_artifact,
                    "peak_uv": feat.peak_uv,
                    "theta_power": feat.theta_power,
                    "alpha_power": feat.alpha_power,
                    "beta_power": feat.beta_power,
                }
            )
            epoch_idx += 1

        return pd.DataFrame(records)


def simulate_raw_eeg(
    duration_sec: float = 10.0,
    sampling_rate: int = SAMPLING_RATE_HZ,
    state: str = "baseline",
    inject_noise: bool = True,
    inject_artifacts: bool = False,
    random_state: int | None = None,
) -> np.ndarray:
    """Simulate realistic microvolt single-channel EEG telemetry (0.5 to 100 uV).

    States:
    - 'baseline': high alpha (10 Hz), moderate theta (6 Hz), low beta (20 Hz)
    - 'moderate_workload': moderate alpha, balanced theta and beta
    - 'fatigue': pronounced theta spike (5-7 Hz), suppressed alpha, elevated beta
    """
    rng = np.random.default_rng(random_state)
    n_points = int(duration_sec * sampling_rate)
    t = np.linspace(0, duration_sec, n_points, endpoint=False)

    if state == "baseline":
        theta_amp, alpha_amp, beta_amp = 6.0, 14.0, 4.0
    elif state in ("1-back", "moderate_workload"):
        theta_amp, alpha_amp, beta_amp = 10.0, 9.0, 7.0
    elif state in ("3-back", "mental_math", "fatigue"):
        theta_amp, alpha_amp, beta_amp = 20.0, 5.0, 11.0
    elif state in ("recovery_box_breathing", "recovery_passive"):
        theta_amp, alpha_amp, beta_amp = 7.0, 12.0, 4.5
    else:
        theta_amp, alpha_amp, beta_amp = 8.0, 10.0, 5.0

    # Synthetic neural oscillations with slight frequency jitter
    theta_sig = theta_amp * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
    alpha_sig = alpha_amp * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
    beta_sig = beta_amp * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi))

    # Base cortical signal in microvolts
    signal_uv = theta_sig + alpha_sig + beta_sig + rng.normal(0, 2.5, n_points)

    if inject_noise:
        # 60 Hz environmental power-line interference (microvolt scale)
        power_line = 8.0 * np.sin(2 * np.pi * 60.0 * t)
        # Slow baseline wander (0.2 Hz)
        wander = 5.0 * np.sin(2 * np.pi * 0.2 * t)
        signal_uv += power_line + wander

    if inject_artifacts:
        # Inject occasional eye blinks or jaw clenching (> +/- 100 uV)
        n_blinks = int(duration_sec // 4)
        for _ in range(n_blinks):
            loc = rng.integers(0, n_points - int(sampling_rate * 0.3))
            blink_len = int(sampling_rate * 0.25)
            signal_uv[loc : loc + blink_len] += 130.0 * np.sin(np.linspace(0, np.pi, blink_len))

    return signal_uv
