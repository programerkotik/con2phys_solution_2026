"""Figures for Question 15: Shannon entropy of single-unit spike counts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from con2phys import plotting


def _binary_entropy_curve(axis: Any, max_rate_hz: float, bin_s: float) -> None:
    """Entropy of a unit that fires at most one spike per bin, as a function of rate."""
    rate = np.linspace(1e-3, min(max_rate_hz, 0.3 / bin_s), 400)  # ≤ 1 spike/bin fails at higher p
    p = np.clip(rate * bin_s, 1e-12, 1 - 1e-12)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    axis.plot(rate, entropy, color="black", linewidth=1, linestyle="--",
              label="binary entropy of rate × bin")


def _plot_rate_scatter(axis: Any, areas_record: dict[str, Any], bin_s: float) -> None:
    """Trial-mean entropy against trial-mean rate, one point per unit."""
    max_rate = 1.0
    for area, entry in areas_record.items():
        axis.scatter(entry["unit_rate_hz"], entry["unit_entropy_bits"],
                     color=plotting.area_color(area), s=10, alpha=0.6, label=f"Area {area}")
        max_rate = max(max_rate, *entry["unit_rate_hz"])
    _binary_entropy_curve(axis, max_rate, bin_s)
    axis.set(xlabel="Unit firing rate in segments (spikes/s)",
             ylabel="Unit entropy (bits per bin)", title="Entropy follows firing rate")
    axis.legend(fontsize=7)


def _plot_raster(axis: Any, trains: list[np.ndarray], unit_areas: np.ndarray,
                 start: float, end: float, areas: list[int]) -> None:
    """Spikes of every unit in one example segment, grouped and coloured by area."""
    row = 0
    for area in areas:
        for unit in np.flatnonzero(unit_areas == area):
            spikes = trains[unit][(trains[unit] >= start) & (trains[unit] < end)] - start
            axis.vlines(spikes, row, row + 0.8, color=plotting.area_color(area), linewidth=0.8)
            row += 1
    axis.set(xlabel="Time from stimulus (s)", ylabel="Unit (grouped by area)",
             xlim=(0, end - start), ylim=(0, row),
             title=f"Example segment (median length, {end - start:.2f} s)")


def plot_animal(mouse: int, trains: list[np.ndarray], unit_areas: np.ndarray,
                record: dict[str, Any], areas: list[int], bin_s: float, path: Path,
                dpi: int) -> None:
    """One mouse: example raster, per-trial entropy per area, and entropy against rate.

    Args:
        mouse: Animal identifier.
        trains: Spike times of all units.
        unit_areas: Area label of each unit.
        record: Per-animal record from ``analyze_mouse``.
        areas: Area labels.
        bin_s: Bin width (s).
        path: Output PNG.
        dpi: Figure resolution.
    """
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    durations = np.asarray(record["durations_s"])
    example = int(np.argsort(durations)[len(durations) // 2])
    start = record["starts_s"][example]
    _plot_raster(axes[0], trains, unit_areas, start, start + durations[example], areas)
    for area, entry in record["areas"].items():
        axes[1].plot(entry["trial_entropy_bits"], color=plotting.area_color(area), linewidth=0.6,
                     alpha=0.8, label=f"Area {area}: mean {entry['entropy_bits']:.3f}")
    axes[1].set(xlabel="Trial", ylabel="Mean unit entropy (bits per bin)",
                title=f"{record['n_trials_used']} trials, stimulus → trial end")
    axes[1].legend(fontsize=7)
    _plot_rate_scatter(axes[2], record["areas"], bin_s)
    figure.suptitle(f"Mouse {mouse}: spike-count entropy in {bin_s * 1000:.0f} ms bins")
    plotting.save(figure, path, dpi)


def plot_rate_dependence(records: dict[int, dict[str, Any]], labels: list[str], bin_s: float,
                         path: Path, dpi: int) -> None:
    """All units of all mice: trial-mean entropy against trial-mean rate.

    Args:
        records: Per-animal results, keyed by mouse.
        labels: Area labels.
        bin_s: Bin width (s).
        path: Output PNG.
        dpi: Figure resolution.
    """
    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    pooled = {a: {key: [v for r in records.values() for v in r["areas"][a][key]]
                  for key in ("unit_rate_hz", "unit_entropy_bits")} for a in labels}
    _plot_rate_scatter(axis, pooled, bin_s)
    axis.set_title("All units of all mice: entropy follows firing rate")
    plotting.save(figure, path, dpi)
