"""LFP signal processing shared by the LFP questions (Q2, Q3, Q5, Q6, Q8)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch


def bandpass(lfp: np.ndarray, sampling_rate_hz: float, band_hz: tuple[float, float],
             order: int) -> np.ndarray:
    """Zero-phase Butterworth band-pass of every channel (float32 output).

    Args:
        lfp: ``[channels × samples]`` LFP.
        sampling_rate_hz: Sampling frequency.
        band_hz: Lower and upper pass-band edges (Hz), below Nyquist.
        order: Butterworth order (applied forward and backward).

    Returns:
        Filtered array of the same shape.

    Scientific rationale:
        Forward-backward filtering adds no phase shift, so filtered events
        and phases stay aligned in time with the spikes and the raw LFP.
    """
    traces = np.asarray(lfp, dtype=np.float32)
    if traces.ndim != 2 or traces.shape[1] < 2:
        raise ValueError("lfp must be a 2-D array with at least two samples")
    low_hz, high_hz = (float(value) for value in band_hz)
    if sampling_rate_hz <= 0 or order < 1 or not 0 < low_hz < high_hz < sampling_rate_hz / 2:
        raise ValueError("Invalid sampling frequency, filter order, or band")
    sos = butter(order, (low_hz, high_hz), btype="bandpass", fs=sampling_rate_hz, output="sos")
    return np.asarray(sosfiltfilt(sos, traces, axis=1), dtype=np.float32)


def welch_psd(lfp: np.ndarray, sampling_rate_hz: float, window_s: float,
              overlap_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of every channel.

    Args:
        lfp: ``[channels × samples]`` LFP.
        sampling_rate_hz: Sampling frequency.
        window_s: Segment length (s); 4 s gives 0.25 Hz resolution.
        overlap_fraction: Overlap between consecutive segments.

    Returns:
        Frequencies (Hz) and one-sided PSD ``[channels × frequencies]``
        (signal units² / Hz; Hann window, constant detrend, mean average).

    Scientific rationale:
        Averaging periodograms of many segments reduces the variance of the
        spectrum while keeping a separate estimate for every channel.
    """
    nperseg = int(round(sampling_rate_hz * window_s))
    noverlap = int(round(nperseg * overlap_fraction))
    if nperseg < 2 or not 0 <= noverlap < nperseg or lfp.shape[-1] < nperseg:
        raise ValueError("Invalid Welch window/overlap for this recording")
    return welch(lfp, fs=sampling_rate_hz, nperseg=nperseg, noverlap=noverlap, axis=-1)


def band_mask(frequencies_hz: np.ndarray, band_hz: tuple[float, float]) -> np.ndarray:
    """Inclusive mask of frequency bins inside ``band_hz``."""
    low_hz, high_hz = band_hz
    return (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)


def bandpower(frequencies_hz: np.ndarray, psd: np.ndarray, band_hz: tuple[float, float]
              ) -> np.ndarray:
    """Integrate each channel's PSD exactly over ``band_hz`` (trapezoid rule).

    Args:
        frequencies_hz: Increasing frequency bins.
        psd: ``[channels × frequencies]`` linear PSD.
        band_hz: Band edges; the PSD is linearly interpolated at both edges.

    Returns:
        Power per channel (signal units²).

    Scientific rationale:
        The area under a linear PSD is the signal power in that band;
        interpolating the edges integrates every channel over exactly the
        requested band, independent of the bin grid.
    """
    low_hz, high_hz = band_hz
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    psd = np.atleast_2d(np.asarray(psd, dtype=float))
    if not frequencies_hz[0] <= low_hz < high_hz <= frequencies_hz[-1]:
        raise ValueError("Requested band must lie within the PSD frequency range")
    inside = (frequencies_hz > low_hz) & (frequencies_hz < high_hz)
    grid = np.concatenate(([low_hz], frequencies_hz[inside], [high_hz]))
    edges = [np.array([np.interp(f, frequencies_hz, row) for row in psd]) for f in band_hz]
    values = np.column_stack((edges[0], psd[:, inside], edges[1]))
    return np.asarray(trapezoid(values, grid, axis=1), dtype=float)
