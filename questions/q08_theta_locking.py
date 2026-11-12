"""Question 8: in which brain area are spikes most strongly phase-locked to its own 4–10 Hz LFP?

Final variable: per mouse and area, the median over units of PPC0 between the
units' spikes and the 4–10 Hz Hilbert phase of the area's most theta-prominent
LFP channel, using only spikes inside detected theta windows (unitless).

Steps per area: (1) Welch PSD of every channel → the channel whose theta peak
rises most above its 1/f background; (2) 2-s windows of that channel → log
theta/delta ratio → two-Gaussian mixture threshold → theta windows; (3) the
channel's 4–10 Hz phase at every theta-window spike → PPC0 per unit; (4) median
over units.

Run: ``python -m questions.q08_theta_locking --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import hilbert, spectrogram

from con2phys import cli, io, signal, stats

KEY = "q08"
QUESTION = ("Q8. In which brain area are spikes most strongly phase-locked to its own LFP "
            "in the 4–10 Hz range?")
TAPER = "hann"  # taper of the 2-s theta-detection windows (non-overlapping, constant detrend)


# ---------------------------------------------------------------------------
# Step 1: channel selection
# ---------------------------------------------------------------------------


def theta_prominence_db(frequencies_hz: np.ndarray, psd: np.ndarray,
                        theta_band_hz: tuple[float, float],
                        baseline_bands_hz: tuple[tuple[float, float], ...]
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Height of each channel's theta peak above its own 1/f baseline.

    Args:
        frequencies_hz: Increasing PSD frequency bins.
        psd: Channel PSDs ``[channels × frequencies]``.
        theta_band_hz: Band searched for the peak.
        baseline_bands_hz: Bands on both sides of theta used to fit the baseline.

    Returns:
        Prominence (dB) per channel and the fitted baseline PSD (NaN at 0 Hz).

    Scientific rationale:
        LFP power falls roughly as a power law of frequency, so a channel can
        have high theta power without any oscillation. A straight line in
        log-log coordinates through the neighbouring bands estimates that
        aperiodic background; the largest excess over it inside the theta band
        measures how oscillatory the channel is, independent of its amplitude.
    """
    frequencies = np.asarray(frequencies_hz, dtype=float)
    psd = np.atleast_2d(np.asarray(psd, dtype=float))
    fit = np.any([signal.band_mask(frequencies, tuple(band)) for band in baseline_bands_hz],
                 axis=0)
    slope, intercept = np.polyfit(np.log10(frequencies[fit]), np.log10(psd[:, fit]).T, 1)
    with np.errstate(divide="ignore"):
        log_frequencies = np.where(frequencies > 0, np.log10(frequencies), np.nan)
    baseline = 10.0 ** (slope[:, None] * log_frequencies[None, :] + intercept[:, None])
    theta = signal.band_mask(frequencies, theta_band_hz)
    prominence = 10.0 * np.log10(psd[:, theta] / baseline[:, theta]).max(axis=1)
    return prominence, baseline


# ---------------------------------------------------------------------------
# Step 2: theta windows
# ---------------------------------------------------------------------------


def window_spectra(trace: np.ndarray, sampling_rate_hz: float, band_hz: tuple[float, float],
                   window_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Complex Fourier coefficients of consecutive, non-overlapping windows of one channel.

    Returns:
        Frequencies inside ``band_hz`` and coefficients ``[frequencies × windows]``.
    """
    nperseg = int(round(window_s * sampling_rate_hz))
    frequencies, _, coefficients = spectrogram(np.asarray(trace)[None, :], fs=sampling_rate_hz,
                                               window=TAPER, nperseg=nperseg, noverlap=0,
                                               detrend="constant", mode="complex", axis=-1)
    keep = signal.band_mask(frequencies, band_hz)
    return frequencies[keep], coefficients[0, keep, :]


def window_log_ratio(spectra: np.ndarray, frequencies_hz: np.ndarray,
                     theta_band_hz: tuple[float, float],
                     delta_band_hz: tuple[float, float]) -> np.ndarray:
    """log10 of theta over delta power in every window (non-finite if a window has no power).

    Scientific rationale:
        Hippocampal theta dominates during locomotion and REM sleep, slow delta
        during quiet rest and slow-wave sleep. Without behavioural or sleep
        annotations, the theta/delta power ratio is the standard LFP marker
        of the theta state.
    """
    power = np.abs(spectra.T) ** 2
    theta = signal.bandpower(frequencies_hz, power, theta_band_hz)
    delta = signal.bandpower(frequencies_hz, power, delta_band_hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(theta / delta)


@dataclass(frozen=True)
class ThetaThreshold:
    """Two-Gaussian split of the log theta/delta ratio; arrays ordered (non-theta, theta).

    ``fallback`` is True when the fit was degenerate and every window is used.
    """

    threshold: float
    means: tuple[float, float]
    sds: tuple[float, float]
    weights: tuple[float, float]
    ashman_d: float
    converged: bool
    fallback: bool


def fit_theta_threshold(log_ratio: np.ndarray, n_init: int, seed: int, min_weight: float,
                        grid_points: int) -> ThetaThreshold:
    """Data-driven theta/non-theta threshold from a two-component Gaussian mixture.

    Args:
        log_ratio: log10 theta/delta ratios of the eligible windows.
        n_init: Number of mixture initialisations.
        seed: Random seed.
        min_weight: Smallest acceptable component weight.
        grid_points: Resolution of the threshold search between the two means.

    Returns:
        Threshold and fit diagnostics; a degenerate fit (not converged, a tiny
        component, or no crossing) gives ``threshold = -inf`` and ``fallback``.

    Scientific rationale:
        Theta and non-theta periods form two overlapping populations of the log
        ratio; their crossing adapts to each recording. The first grid point
        between the means where the theta posterior reaches 0.5 gives a
        monotone rule. Ashman's D (> 2 = clearly bimodal) measures how well the
        two populations separate.
    """
    from sklearn.mixture import GaussianMixture

    values = np.asarray(log_ratio, dtype=float)
    values = values[np.isfinite(values)].reshape(-1, 1)
    if len(values) < 4:
        raise ValueError("At least four finite windows are needed to fit the mixture")
    model = GaussianMixture(2, n_init=n_init, random_state=seed).fit(values)
    order = np.argsort(model.means_.ravel())
    means = model.means_.ravel()[order]
    sds = np.sqrt(model.covariances_.ravel()[order])
    weights = model.weights_[order]
    grid = np.linspace(means[0], means[1], grid_points).reshape(-1, 1)
    crossing = np.flatnonzero(model.predict_proba(grid)[:, order[1]] >= 0.5)
    fallback = (not model.converged_) or weights.min() < min_weight or len(crossing) == 0
    return ThetaThreshold(
        threshold=-np.inf if fallback else float(grid[crossing[0], 0]),
        means=(float(means[0]), float(means[1])), sds=(float(sds[0]), float(sds[1])),
        weights=(float(weights[0]), float(weights[1])),
        ashman_d=float(np.sqrt(2.0) * abs(means[1] - means[0]) / np.sqrt(np.sum(sds**2))),
        converged=bool(model.converged_), fallback=bool(fallback))


def theta_window_mask(log_ratio: np.ndarray, threshold: float) -> np.ndarray:
    """Windows at or above the threshold, never the first or last window.

    Scientific rationale:
        Filtering and the Hilbert transform are distorted at the recording
        ends, so the two end windows are always excluded; windows without a
        finite ratio carry no LFP power and are excluded too.
    """
    ratio = np.asarray(log_ratio, dtype=float)
    mask = np.isfinite(ratio)
    mask[[0, -1]] = False
    with np.errstate(invalid="ignore"):
        return mask & (ratio >= threshold)


# ---------------------------------------------------------------------------
# Step 3: spike phases and PPC0
# ---------------------------------------------------------------------------


def instantaneous_phase(trace: np.ndarray, sampling_rate_hz: float,
                        band_hz: tuple[float, float], order: int) -> np.ndarray:
    """Hilbert phase of one channel after zero-phase band-pass filtering.

    Returns:
        Phase (rad) per sample: 0 at the filtered LFP peak, ±π at the trough.

    Scientific rationale:
        The analytic signal of a narrow-band signal assigns every sample a
        position in the current cycle; zero-phase filtering keeps it aligned
        with the spikes. The FFT is padded to a fast length; the padding only
        affects the excluded end windows.
    """
    filtered = signal.bandpass(np.asarray(trace)[None, :], sampling_rate_hz, band_hz, order)[0]
    analytic = hilbert(filtered, N=next_fast_len(filtered.size))[: filtered.size]
    return np.angle(analytic).astype(np.float32)


def theta_spikes(spike_times_by_unit: list[np.ndarray], sampling_rate_hz: float,
                 window_samples: int, theta_mask: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LFP sample, unit index and window index of every spike inside a theta window.

    Scientific rationale:
        Spikes and LFP share one clock starting at 0 s. A spike is assigned to
        the LFP sample whose 2-ms interval contains it (floor, as in Elephant)
        and kept only during theta; spikes outside the recording are dropped.
    """
    limit = len(theta_mask) * window_samples
    samples, units = [np.empty(0, dtype=np.int64)], [np.empty(0, dtype=np.int64)]
    for index, times in enumerate(spike_times_by_unit):
        unit_samples = np.floor(np.asarray(times, dtype=float) * sampling_rate_hz).astype(np.int64)
        unit_samples = unit_samples[(unit_samples >= 0) & (unit_samples < limit)]
        unit_samples = unit_samples[theta_mask[unit_samples // window_samples]]
        samples.append(unit_samples)
        units.append(np.full(unit_samples.size, index, dtype=np.int64))
    samples_all = np.concatenate(samples)
    return samples_all, np.concatenate(units), samples_all // window_samples


def circular_stats(phases: np.ndarray, unit_index: np.ndarray, n_units: int
                   ) -> dict[str, np.ndarray]:
    """Per-unit phase-locking statistics from spike phases.

    Args:
        phases: Phase of every spike (rad).
        unit_index: Unit of every spike, ``0 .. n_units - 1``.
        n_units: Number of units.

    Returns:
        Arrays of length ``n_units``: ``n_spikes``, ``ppc0``, ``mrl``,
        ``preferred_phase`` and ``rayleigh_p``; NaN for units with < 2 spikes.

    Scientific rationale:
        PPC0 is the mean cosine of the phase difference over all spike pairs,
        (|S|² − N)/(N(N − 1)) with S the sum of unit phase vectors. Unlike the
        mean resultant length (MRL = |S|/N) it is not biased upward by a small
        spike count (Vinck et al. 2010); it approximates MRL² for many spikes.
        The Rayleigh p (Zar's approximation) is kept for the descriptive
        preferred-phase figure only.
    """
    cos, sin = np.cos(phases, dtype=float), np.sin(phases, dtype=float)
    n = np.bincount(unit_index, minlength=n_units).astype(float)
    sum_cos = np.bincount(unit_index, cos, n_units)
    sum_sin = np.bincount(unit_index, sin, n_units)
    resultant2 = sum_cos**2 + sum_sin**2
    with np.errstate(divide="ignore", invalid="ignore"):
        result = {
            "n_spikes": n, "ppc0": (resultant2 - n) / (n * (n - 1)),
            "mrl": np.sqrt(resultant2) / n, "preferred_phase": np.arctan2(sum_sin, sum_cos),
            "rayleigh_p": np.clip(np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - resultant2))
                                         - (1 + 2 * n)), 0.0, 1.0)}
    for name in ("ppc0", "mrl", "preferred_phase", "rayleigh_p"):
        result[name] = np.where(n >= 2, result[name], np.nan)
    return result


def median_finite(values: np.ndarray) -> float:
    """Median over units with a defined value (NaN if none).

    Scientific rationale:
        The median limits the influence of units with few theta spikes, whose
        PPC0 is very noisy; units with fewer than two spikes have no PPC0.
    """
    finite = np.isfinite(values)
    return float(np.median(values[finite])) if finite.any() else float("nan")


# ---------------------------------------------------------------------------
# One area, one animal, all animals
# ---------------------------------------------------------------------------


def select_channel(lfp: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Whole-recording Welch PSD of every channel and the most theta-prominent channel."""
    settings = config[KEY]
    frequencies, psd = signal.welch_psd(lfp, float(config["lfp_sampling_rate_hz"]),
                                        float(config["welch"]["window_s"]),
                                        float(config["welch"]["overlap_fraction"]))
    prominence, baseline = theta_prominence_db(
        frequencies, psd, tuple(settings["theta_band_hz"]),
        tuple(tuple(b) for b in settings["baseline_bands_hz"]))
    keep = signal.band_mask(frequencies, tuple(settings["spectrum_band_hz"]))
    return {"selected_channel": int(np.argmax(prominence)), "prominence_db": prominence,
            "welch_freqs": frequencies[keep], "psd": psd[:, keep], "baseline": baseline[:, keep]}


def detect_theta(trace: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Theta/delta ratio per 2-s window, mixture threshold, theta mask and quality flag.

    Scientific rationale:
        A mouse-area is flagged when the mixture is degenerate, the two
        populations are not clearly separated (Ashman's D < 2) or the theta
        fraction is extreme; flagged areas are kept but reported.
    """
    settings = config[KEY]
    frequencies, spectra = window_spectra(trace, float(config["lfp_sampling_rate_hz"]),
                                          tuple(settings["spectrum_band_hz"]),
                                          float(settings["window_s"]))
    log_ratio = window_log_ratio(spectra, frequencies, tuple(settings["theta_band_hz"]),
                                 tuple(settings["delta_band_hz"]))
    fit = fit_theta_threshold(log_ratio[1:-1], int(settings["gmm_n_init"]),
                              int(config["random_seed"]), float(settings["gmm_min_weight"]),
                              int(settings["gmm_grid_points"]))
    mask = theta_window_mask(log_ratio, fit.threshold)
    fraction = float(mask.sum() / (len(mask) - 2))
    low, high = settings["theta_fraction_flag"]
    flagged = fit.fallback or fit.ashman_d < float(settings["ashman_d_min"]) or not (
        low <= fraction <= high)
    return {"log_ratio": log_ratio, "theta_mask": mask, "theta_fraction": fraction,
            "n_theta_windows": int(mask.sum()), "flagged": bool(flagged),
            **{f"gmm_{key}": value for key, value in asdict(fit).items()}}


def analyze_area(lfp: np.ndarray, trains: list[np.ndarray], config: dict[str, Any]
                 ) -> dict[str, Any]:
    """Steps 1–3 for the units of one area against that area's LFP.

    Returns:
        Channel selection, theta detection and per-unit statistics; the phase
        trace and spike samples are included for the figures.
    """
    settings = config[KEY]
    fs = float(config["lfp_sampling_rate_hz"])
    channel = select_channel(lfp, config)
    trace = lfp[channel["selected_channel"]]
    theta = detect_theta(trace, config)
    window_samples = int(round(float(settings["window_s"]) * fs))
    samples, units, _ = theta_spikes(trains, fs, window_samples, theta["theta_mask"])
    phase = instantaneous_phase(trace, fs, tuple(settings["theta_band_hz"]),
                                int(settings["filter_order"]))
    unit_stats = circular_stats(phase[samples], units, len(trains))
    theta_seconds = theta["theta_mask"].sum() * float(settings["window_s"])
    unit_stats["theta_rate_hz"] = unit_stats["n_spikes"] / theta_seconds
    return {**channel, **theta, "units": unit_stats, "phase": phase,
            "spike_samples": samples, "spike_units": units}


def area_record(result: dict[str, Any], unit_ids: np.ndarray) -> dict[str, Any]:
    """JSON-serialisable record of one area: mouse value, theta QC and per-unit values."""
    units = result["units"]
    gmm = {k: v for k, v in result.items() if k.startswith("gmm_")}
    return {"ppc0": median_finite(units["ppc0"]), "n_units": int(len(unit_ids)),
            "n_undefined": int(np.sum(units["n_spikes"] < 2)),
            "selected_channel": result["selected_channel"],
            "n_channels": int(len(result["prominence_db"])),
            "prominence_db": np.asarray(result["prominence_db"]).tolist(),
            "theta_fraction": result["theta_fraction"],
            "n_theta_windows": result["n_theta_windows"], "flagged": result["flagged"], **gmm,
            "unit_ids": np.asarray(unit_ids).tolist(),
            **{f"unit_{k}": np.asarray(v).tolist() for k, v in units.items()}}


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """PPC0 per unit and its median per area for one animal (plus QC and figures)."""
    from questions import q08_figures as figures

    folder = io.mouse_dir(config, mouse)
    fs = float(config["lfp_sampling_rate_hz"])
    animal = io.load_spikes(folder)
    io.report_out_of_range_spikes(mouse, animal.spikes_s, io.recording_duration_s(folder, fs))
    all_trains = io.unit_spike_trains(animal)
    record: dict[str, Any] = {"areas": {}}
    results: dict[int, dict[str, Any]] = {}
    for area in (int(a) for a in config["areas"]):
        lfp = io.load_lfp(io.lfp_path(folder, area), fs, dtype=np.float32)
        in_area = animal.unit_areas == area
        trains = [t for t, keep in zip(all_trains, in_area) if keep]
        result = analyze_area(lfp, trains, config)
        record["areas"][str(area)] = area_record(result, animal.unit_ids[in_area])
        cli.log(f"mouse {mouse} area {area}: ch {result['selected_channel']}, "
                f"theta {100 * result['theta_fraction']:.0f}%, "
                f"PPC0 {record['areas'][str(area)]['ppc0']:.5f}")
        if mouse == int(config["example_mouse"]):
            result["snippet"] = figures.snippet(lfp[result["selected_channel"]], result, trains,
                                                animal.unit_ids[in_area], config)
            result["unit_ids"] = animal.unit_ids[in_area]
        results[area] = result
        del lfp
    figures.plot_theta_detection(config, mouse, results)
    if mouse == int(config["example_mouse"]):
        figures.plot_example(config, mouse, results)
    return record


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary, decision, QC table and figures."""
    from questions import q08_figures as figures

    labels = [str(a) for a in config["areas"]]
    values = {m: [r["areas"][a]["ppc0"] for a in labels] for m, r in records.items()}
    matrix = np.array(list(values.values()))
    summary = stats.summarize_conditions(matrix, labels, config, ranking="highest")
    figures.plot_aggregate(config, records, matrix, labels, summary)
    qc = {str(m): {a: {k: r["areas"][a][k] for k in ("selected_channel", "theta_fraction",
                                                      "gmm_ashman_d", "flagged", "n_units")}
                   for a in labels} for m, r in records.items()}
    flagged = [[m, int(a)] for m, r in records.items() for a in labels
               if r["areas"][a]["flagged"]]
    return cli.final_report(
        QUESTION, "median over units of PPC0 between spikes and the 4–10 Hz phase of the "
        "area's theta-prominent channel, theta windows only", "unitless (pairwise phase "
        "consistency)", values, labels, summary, flagged_mouse_areas=flagged,
        n_units={a: int(sum(r["areas"][a]["n_units"] for r in records.values()))
                 for a in labels}, theta_qc=qc)


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
