"""Figures for Question 11: Skaggs information about variable A against its null."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from con2phys import plotting

REAL_COLOR, NULL_COLOR = "C0", "0.5"


def _ecdf(axis: Any, values: np.ndarray, **style: Any) -> None:
    """Empirical cumulative distribution of the finite values."""
    values = np.sort(values[np.isfinite(values)])
    axis.step(values, np.arange(1, values.size + 1) / max(values.size, 1), where="post", **style)


def _log_bins(*arrays: np.ndarray, n: int = 40) -> np.ndarray:
    """Logarithmic bin edges spanning the positive finite values of all arrays."""
    values = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in arrays])
    return np.geomspace(values.min(), values.max(), n + 1)


def _example_panel(axis: Any, data: dict[str, np.ndarray], area: int) -> None:
    """Null histogram of the unit with the largest excess, with its real score marked."""
    with np.errstate(invalid="ignore"):
        excess = data["information"] - np.nanmean(data["null"], axis=1)
    if not np.any(np.isfinite(excess)):
        axis.axis("off")
        return
    unit = int(np.nanargmax(excess))
    axis.hist(data["null"][unit], bins=40, color=NULL_COLOR, alpha=0.7, label="null (1000 shifts)")
    axis.axvline(data["information"][unit], color=REAL_COLOR, linewidth=2, label="real")
    axis.set(xlabel="Information (bits/spike)", ylabel="Shuffles",
             title=f"Area {area}, unit {data['unit_ids'][unit]}: "
                   f"{data['rate_a0'][unit]:.2f} Hz (A=0) vs {data['rate_a1'][unit]:.2f} Hz (A=1)\n"
                   f"{int(data['n_spikes'][unit])} spikes, p = {data['p_value'][unit]:.3g}")
    axis.title.set_fontsize(8)
    axis.legend(fontsize=7)


def plot_animal(mouse: int, by_area: dict[int, dict[str, np.ndarray]], path: Path,
                dpi: int) -> None:
    """Per-animal steps: real vs null distributions and one example unit per area.

    Args:
        mouse: Animal identifier.
        by_area: Per-area arrays from ``analyze_mouse`` (keys without the area prefix).
        path: Output PNG.
        dpi: Figure resolution.
    """
    figure, axes = plt.subplots(2, len(by_area), figsize=(5 * len(by_area), 8), squeeze=False)
    for column, (area, data) in enumerate(by_area.items()):
        axis = axes[0, column]
        _ecdf(axis, data["information"], color=REAL_COLOR, label="real")
        _ecdf(axis, data["null_one"], color=NULL_COLOR, label="null (one shift per unit)")
        axis.set_xscale("log")
        n = int(np.isfinite(data["information"]).sum())
        axis.set(xlabel="Information about A (bits/spike)", ylabel="Fraction of units",
                 title=f"Area {area} ({n} units)")
        axis.legend(fontsize=7, loc="upper left")
        _example_panel(axes[1, column], data, area)
    figure.suptitle(f"Mouse {mouse}: Skaggs information about variable A inside trials")
    plotting.save(figure, path, dpi)


def plot_distributions(unit_rows: list[dict[str, Any]], areas: list[int], path: Path,
                       dpi: int) -> None:
    """Pooled real vs null distributions per area, and both against in-trial spike count.

    Args:
        unit_rows: One dict per unit with ``area``, ``information``, ``null_one``,
            ``null_mean`` and ``n_spikes``.
        areas: Area labels, one column each.
        path: Output PNG.
        dpi: Figure resolution.
    """
    column = {key: np.array([r[key] for r in unit_rows], dtype=float)
              for key in ("area", "information", "null_one", "null_mean", "n_spikes")}
    bins = _log_bins(column["information"], column["null_one"])
    figure, axes = plt.subplots(2, len(areas), figsize=(5 * len(areas), 8), sharex="row",
                                sharey="row", squeeze=False)
    for index, area in enumerate(areas):
        mask = column["area"] == area
        axis = axes[0, index]
        axis.hist(column["information"][mask], bins=bins, color=REAL_COLOR, alpha=0.6, label="real")
        axis.hist(column["null_one"][mask], bins=bins, color=NULL_COLOR, alpha=0.6,
                  label="null (one shift per unit)")
        axis.set_xscale("log")
        n = int(np.isfinite(column["information"][mask]).sum())
        axis.set(xlabel="Information about A (bits/spike)", ylabel="Units",
                 title=f"Area {area}: {n} units, all mice")
        axis.legend(fontsize=7)
        axis = axes[1, index]
        axis.scatter(column["n_spikes"][mask], column["information"][mask], s=6,
                     color=REAL_COLOR, alpha=0.5, label="real")
        axis.scatter(column["n_spikes"][mask], column["null_mean"][mask], s=6,
                     color=NULL_COLOR, alpha=0.5, label="null mean")
        axis.set(xscale="log", yscale="log", xlabel="In-trial spikes",
                 ylabel="Information (bits/spike)", title="Real score and null mean vs spike count")
        axis.legend(fontsize=7)
    figure.suptitle("Question 11: real information vs circular-shift null, units pooled over mice")
    plotting.save(figure, path, dpi)
