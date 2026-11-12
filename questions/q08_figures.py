"""Figures for Question 8 (theta phase locking): one per analysis step, plus summaries."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from con2phys import cli, plotting, signal

KEY = "q08"


def _folder(config: dict[str, Any]):
    return cli.output_dir(config, KEY) / "figures"


# ---------------------------------------------------------------------------
# Steps 1–2: channel selection and theta windows (every mouse, for QC)
# ---------------------------------------------------------------------------


def _channel_panel(axis: Any, result: dict[str, Any], area: int) -> None:
    """Theta prominence of every channel, with the selected channel marked."""
    prominence, selected = result["prominence_db"], int(result["selected_channel"])
    axis.plot(prominence, np.arange(len(prominence)), "-o", color="0.5", markersize=3)
    axis.plot(prominence[selected], selected, "o", color=plotting.area_color(area),
              markersize=9, label=f"selected: ch {selected}")
    axis.set(xlabel="Theta peak above 1/f (dB)", ylabel="Channel (0 = deepest)",
             title=f"Area {area}: step 1, channel selection")
    axis.legend(fontsize=8, loc="lower right")


def _psd_panel(axis: Any, result: dict[str, Any], area: int, theta_band: tuple) -> None:
    """All channel PSDs, the selected one with its fitted 1/f baseline."""
    freqs, psd, selected = result["welch_freqs"], result["psd"], int(result["selected_channel"])
    axis.loglog(freqs, psd.T, color="0.8", linewidth=0.5)
    axis.loglog(freqs, psd[selected], color=plotting.area_color(area), linewidth=1.8,
                label="selected")
    axis.loglog(freqs, result["baseline"][selected], "k--", linewidth=1, label="1/f fit")
    axis.axvspan(*theta_band, color="gold", alpha=0.2)
    axis.set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)", title="Whole-recording PSD per channel")
    axis.legend(fontsize=8)


def _ratio_histogram_panel(axis: Any, result: dict[str, Any]) -> None:
    """Window log ratios with the fitted two-Gaussian mixture and the threshold."""
    ratio = result["log_ratio"][1:-1]
    ratio = ratio[np.isfinite(ratio)]
    axis.hist(ratio, bins=80, density=True, color="0.75")
    grid = np.linspace(ratio.min(), ratio.max(), 400)
    for mean, sd, weight, color in zip(result["gmm_means"], result["gmm_sds"],
                                       result["gmm_weights"], ("0.3", "crimson")):
        axis.plot(grid, weight * norm.pdf(grid, mean, sd), color=color, linewidth=1.5)
    if not result["gmm_fallback"]:
        axis.axvline(result["gmm_threshold"], color="black", linestyle="--")
    flag = " FLAGGED" if result["flagged"] else ""
    axis.set(xlabel="log10(theta / delta) per 2-s window", ylabel="Density",
             title=f"Step 2: theta = {100 * result['theta_fraction']:.0f}% of windows, "
                   f"D = {result['gmm_ashman_d']:.1f}{flag}")


def _ratio_time_panel(axis: Any, result: dict[str, Any], window_s: float) -> None:
    """Window log ratio over the recording, theta windows highlighted."""
    ratio, mask = result["log_ratio"], result["theta_mask"].astype(bool)
    minutes = (np.arange(len(ratio)) + 0.5) * window_s / 60.0
    axis.scatter(minutes[~mask], ratio[~mask], s=2, color="0.6", label="non-theta")
    axis.scatter(minutes[mask], ratio[mask], s=2, color="crimson", label="theta")
    if not result["gmm_fallback"]:
        axis.axhline(result["gmm_threshold"], color="black", linestyle="--")
    axis.set(xlabel="Time (min)", ylabel="log10(theta / delta)", title="Theta windows over time")
    axis.legend(fontsize=8, markerscale=4, loc="upper right")


def plot_theta_detection(config: dict[str, Any], mouse: int,
                         results: dict[int, dict[str, Any]]) -> None:
    """Channel selection and theta-window detection of one mouse, one row per area."""
    settings = config[KEY]
    figure, axes = plt.subplots(len(results), 4, figsize=(21, 4.2 * len(results)),
                                squeeze=False, gridspec_kw={"width_ratios": [1, 1.3, 1.3, 2]})
    for row, (area, result) in zip(axes, results.items()):
        _channel_panel(row[0], result, area)
        _psd_panel(row[1], result, area, tuple(settings["theta_band_hz"]))
        _ratio_histogram_panel(row[2], result)
        _ratio_time_panel(row[3], result, float(settings["window_s"]))
    figure.suptitle(f"Mouse {mouse} – theta channel and theta-window detection")
    plotting.save(figure, _folder(config) / "per_animal" / f"mouse_{mouse:02d}_theta_detection.png",
                  int(config["dpi"]))


# ---------------------------------------------------------------------------
# Step 3: phase extraction and example units (example mouse only)
# ---------------------------------------------------------------------------


def snippet(trace: np.ndarray, result: dict[str, Any], trains: list[np.ndarray],
            unit_ids: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """A short theta-rich stretch of the selected channel for the phase-extraction figure."""
    settings = config[KEY]
    fs = float(config["lfp_sampling_rate_hz"])
    window_s, length_s = float(settings["window_s"]), float(settings["snippet_s"])
    ratio = np.where(result["theta_mask"], result["log_ratio"], -np.inf)
    start_s = max(0.0, (int(np.argmax(ratio)) + 0.5) * window_s - length_s / 2)
    start, stop = int(start_s * fs), int((start_s + length_s) * fs)
    padded_start = max(0, start - int(2 * fs))  # 2 s padding against filter edge effects
    filtered = signal.bandpass(trace[padded_start:stop + int(2 * fs)][None, :], fs,
                               tuple(settings["theta_band_hz"]), int(settings["filter_order"]))[0]
    spikes = [(int(u), t[(t >= start / fs) & (t < stop / fs)]) for u, t in zip(unit_ids, trains)]
    spikes = sorted(spikes, key=lambda item: -item[1].size)[: int(settings["snippet_units"])]
    windows = np.flatnonzero(result["theta_mask"])
    return {"channel": result["selected_channel"], "time_s": np.arange(start, stop) / fs,
            "raw": trace[start:stop].astype(float),
            "filtered": filtered[start - padded_start:stop - padded_start],
            "phase": result["phase"][start:stop],
            "theta_spans": [(max(w * window_s, start / fs), min((w + 1) * window_s, stop / fs))
                            for w in windows if start_s - window_s < w * window_s < stop / fs],
            "spikes": [(u, t, result["phase"][np.floor(t * fs).astype(int)]) for u, t in spikes]}


def _snippet_rows(axes: list[Any], snip: dict[str, Any], area: int) -> None:
    """Raw and filtered LFP, Hilbert phase, and spikes coloured by phase."""
    time = snip["time_s"]
    for start, stop in snip["theta_spans"]:
        for axis in axes:
            axis.axvspan(start, stop, color="gold", alpha=0.15, linewidth=0)
    scale = snip["raw"].std()
    axes[0].plot(time, (snip["raw"] - snip["raw"].mean()) / scale, color="0.6", linewidth=0.7,
                 label="raw LFP (z)")
    axes[0].plot(time, snip["filtered"] / scale, color=plotting.area_color(area), linewidth=1.5,
                 label="4–10 Hz filtered")
    axes[0].set(ylabel="LFP (z)", title=f"Area {area}, channel {snip['channel']}: "
                                        "gold = theta windows")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].plot(time, snip["phase"], color="0.3", linewidth=0.8)
    axes[1].set(ylabel="Phase (rad)", yticks=[-np.pi, 0, np.pi], yticklabels=["−π", "0 peak", "π"])
    points = None
    for row, (_, times, phases) in enumerate(snip["spikes"]):
        points = axes[2].scatter(times, np.full(times.size, row), c=phases, cmap="twilight",
                                 vmin=-np.pi, vmax=np.pi, marker="|", s=120)
    axes[2].set(ylabel="Unit", xlabel="Time (s)", yticks=range(len(snip["spikes"])),
                yticklabels=[str(u) for u, _, _ in snip["spikes"]], xlim=(time[0], time[-1]))
    if points is not None:
        plt.colorbar(points, cax=axes[2].inset_axes([1.005, 0.0, 0.012, 1.0]), label="phase (rad)")


def _example_units(result: dict[str, Any], min_spikes: int) -> list[dict[str, Any]]:
    """Units with the highest, median and lowest PPC0 among well-sampled units."""
    units = result["units"]
    eligible = np.flatnonzero((units["n_spikes"] >= min_spikes) & np.isfinite(units["ppc0"]))
    if eligible.size == 0:
        return []
    ranked = eligible[np.argsort(units["ppc0"][eligible])]
    picks = {"highest": ranked[-1], "median": ranked[len(ranked) // 2], "lowest": ranked[0]}
    phases = result["phase"][result["spike_samples"]]
    return [{"label": label, "unit_id": int(result["unit_ids"][i]),
             "phases": phases[result["spike_units"] == i], "ppc0": float(units["ppc0"][i]),
             "n_spikes": int(units["n_spikes"][i])} for label, i in picks.items()]


def plot_example(config: dict[str, Any], mouse: int, results: dict[int, dict[str, Any]]) -> None:
    """Step 3 for the example mouse: phase extraction and example-unit phase histograms."""
    figure = plt.figure(figsize=(13, 4.6 * len(results)))
    grid = figure.add_gridspec(3 * len(results), 1, height_ratios=[2, 1, 1.4] * len(results))
    for index, (area, result) in enumerate(results.items()):
        axes = [figure.add_subplot(grid[3 * index + offset]) for offset in range(3)]
        for axis in axes[1:]:
            axis.sharex(axes[0])
        _snippet_rows(axes, result["snippet"], area)
    figure.suptitle(f"Mouse {mouse} – LFP → 4–10 Hz → Hilbert phase → phase of every spike")
    plotting.save(figure, _folder(config) / "example_phase_extraction.png", int(config["dpi"]))

    bins = np.linspace(-np.pi, np.pi, int(config[KEY]["phase_histogram_bins"]) + 1)
    figure = plt.figure(figsize=(13, 4.2 * len(results)))
    for row, (area, result) in enumerate(results.items()):
        for column, unit in enumerate(_example_units(result, int(config[KEY]["example_min_spikes"]))):
            axis = figure.add_subplot(len(results), 3, 3 * row + column + 1, projection="polar")
            counts, _ = np.histogram(unit["phases"], bins=bins)
            axis.bar(bins[:-1], counts / counts.sum(), width=np.diff(bins), align="edge",
                     color=plotting.area_color(area), alpha=0.7, edgecolor="white")
            axis.set_title(f"Area {area}, unit {unit['unit_id']} ({unit['label']} PPC0)\n"
                           f"PPC0 = {unit['ppc0']:.4f}, {unit['n_spikes']} theta spikes",
                           fontsize=8)
            axis.set_yticklabels([])
    figure.suptitle(f"Mouse {mouse} – spike-phase histograms of example units "
                    "(0 = theta peak, ±π = trough)")
    plotting.save(figure, _folder(config) / "example_units.png", int(config["dpi"]))


# ---------------------------------------------------------------------------
# Across mice
# ---------------------------------------------------------------------------


def _qc_panel(axis: Any, records: dict[int, dict[str, Any]], labels: list[str], key: str,
              ylabel: str, line: float | None) -> None:
    """One QC quantity per mouse and area (dots, grey lines join one mouse)."""
    values = np.array([[r["areas"][a][key] for a in labels] for r in records.values()])
    for row in values:
        axis.plot(range(len(labels)), row, "-", color="0.8", linewidth=0.6)
    for i, area in enumerate(labels):
        axis.scatter(np.full(len(values), i), values[:, i], color=plotting.area_color(area), s=18)
    if line is not None:
        axis.axhline(line, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(labels)), [f"Area {a}" for a in labels])
    axis.set_ylabel(ylabel)


def plot_aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]],
                   matrix: np.ndarray, labels: list[str], summary: dict[str, Any]) -> None:
    """Summary of the final variable, theta-detection QC and preferred phases."""
    folder = _folder(config)
    plotting.plot_summary(matrix, labels, summary, "Median unit PPC0 (theta windows)", "Q8",
                          folder / "summary.png")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    _qc_panel(axes[0], records, labels, "gmm_ashman_d", "Ashman's D (bimodality)",
              float(config[KEY]["ashman_d_min"]))
    _qc_panel(axes[1], records, labels, "theta_fraction", "Fraction of 2-s windows in theta",
              None)
    figure.suptitle("Q8 theta detection quality per mouse and area (D < 2 → flagged)")
    plotting.save(figure, folder / "theta_qc.png", int(config["dpi"]))

    edges = np.linspace(-np.pi, np.pi, int(config[KEY]["phase_histogram_bins"]) + 1)
    figure = plt.figure(figsize=(14, 4.8))
    alpha = float(config["significance_level"])
    for index, area in enumerate(labels):
        entries = [r["areas"][area] for r in records.values()]
        phases = [p for e in entries for p, q in zip(e["unit_preferred_phase"], e["unit_rayleigh_p"])
                  if q is not None and np.isfinite(q) and q < alpha]
        total = sum(e["n_units"] for e in entries)
        axis = figure.add_subplot(1, 3, index + 1, projection="polar")
        counts, _ = np.histogram(phases, bins=edges)
        axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
                 color=plotting.area_color(area), alpha=0.7, edgecolor="white")
        axis.set_title(f"Area {area}: {len(phases)}/{total} units Rayleigh p < {alpha}",
                       fontsize=10)
    figure.suptitle("Preferred theta phase of locked units (0 = theta peak; descriptive only)")
    plotting.save(figure, folder / "preferred_phase.png", int(config["dpi"]))
