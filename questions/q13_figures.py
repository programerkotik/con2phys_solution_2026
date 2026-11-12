"""Figures for Question 13: dimensionality of trial-averaged population trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from con2phys import plotting


def _explained_variance_curve(trajectory: np.ndarray) -> np.ndarray:
    """Cumulative explained variance (same as ``q13_dimensionality.explained_variance_curve``)."""
    centred = trajectory - trajectory.mean(axis=0)
    variances = np.linalg.svd(centred, compute_uv=False) ** 2
    return np.cumsum(variances) / variances.sum()


def _plot_heatmap(axis: Any, trajectory: np.ndarray) -> Any:
    """Centred normalised trajectory, units sorted by the time of their peak."""
    centred = trajectory - trajectory.mean(axis=0)
    order = np.argsort(np.argmax(centred, axis=0))
    limit = np.abs(centred).max()
    image = axis.imshow(centred[:, order].T, aspect="auto", cmap="RdBu_r", vmin=-limit,
                        vmax=limit, extent=(0, 1, trajectory.shape[1] - 0.5, -0.5),
                        interpolation="nearest")
    axis.set(xlabel="Time (0 = stimulus, 1 = outcome)", ylabel="Unit (sorted by peak)")
    return image


def _plot_scree(axis: Any, trajectory: np.ndarray, area_record: dict[str, Any],
                matched_n: int, threshold: float, n_plotted: int, color: str,
                rng: np.random.Generator) -> None:
    """Cumulative explained variance of random matched-n subsets, with the threshold."""
    total = trajectory.shape[1]
    for _ in range(n_plotted):
        curve = _explained_variance_curve(trajectory[:, rng.choice(total, matched_n,
                                                                   replace=False)])
        axis.plot(np.arange(1, len(curve) + 1), curve, color=color, alpha=0.2, linewidth=0.8)
    axis.axhline(threshold, color="black", linestyle="--", linewidth=1,
                 label=f"{threshold:.0%} variance")
    axis.set(xlabel=f"Number of PCs (subsets of n = {matched_n} units)",
             ylabel="Cumulative explained variance", ylim=(0, 1.02),
             title=f"mean k = {area_record['mean_k']:.2f}, k/n = "
                   f"{area_record['dimensionality']:.2f}")
    axis.legend(fontsize=7, loc="lower right")


def _plot_pc_plane(axis: Any, trajectory: np.ndarray) -> None:
    """Trajectory of all the area's units projected on its first two PCs."""
    centred = trajectory - trajectory.mean(axis=0)
    left, singular, _ = np.linalg.svd(centred, full_matrices=False)
    scores = left[:, :2] * singular[:2]
    time = np.linspace(0, 1, len(scores))
    axis.plot(scores[:, 0], scores[:, 1], color="0.6", linewidth=1, zorder=1)
    axis.scatter(scores[:, 0], scores[:, 1], c=time, cmap="Blues", vmin=-0.4, vmax=1,
                 s=18, zorder=2)
    axis.scatter(*scores[0], marker="o", s=60, facecolor="none", edgecolor="black",
                 label="stimulus", zorder=3)
    axis.scatter(*scores[-1], marker="s", s=60, facecolor="none", edgecolor="black",
                 label="outcome", zorder=3)
    axis.set(xlabel="PC 1", ylabel="PC 2",
             title=f"All {trajectory.shape[1]} units, PC 1–2 (darker = later)")
    axis.legend(fontsize=7)


def plot_animal(mouse: int, trajectories: dict[int, np.ndarray], record: dict[str, Any],
                settings: dict[str, Any], rng: np.random.Generator, path: Path,
                dpi: int) -> None:
    """Every processing step of one mouse: trajectory, subset spectra and PC plane.

    Args:
        mouse: Animal identifier.
        trajectories: Normalised ``[n_bins × n_units]`` trajectory per area.
        record: Per-animal record from ``analyze_mouse``.
        settings: The ``q13`` block of ``config.yaml``.
        rng: Random generator for the plotted subsets (used after the analysis).
        path: Output PNG.
        dpi: Figure resolution.
    """
    figure, axes = plt.subplots(len(trajectories), 3, figsize=(14, 3.6 * len(trajectories)),
                                squeeze=False)
    for row, (area, trajectory) in enumerate(trajectories.items()):
        image = _plot_heatmap(axes[row, 0], trajectory)
        figure.colorbar(image, ax=axes[row, 0], label="Normalised rate − unit mean")
        area_record = record["areas"][str(area)]
        axes[row, 0].set_title(f"Area {area}: {area_record['n_units_used']} of "
                               f"{area_record['n_units']} units")
        _plot_scree(axes[row, 1], trajectory, area_record, record["matched_n"],
                    float(settings["variance_threshold"]), int(settings["n_plotted_subsets"]),
                    plotting.area_color(area), rng)
        _plot_pc_plane(axes[row, 2], trajectory)
    figure.suptitle(f"Mouse {mouse}: trial-averaged stimulus→outcome trajectories "
                    f"({record['n_trials_used']} trials)")
    plotting.save(figure, path, dpi)


def plot_size_curves(records: dict[int, dict[str, Any]], areas: list[int], path: Path,
                     dpi: int) -> None:
    """Dimensionality against subset size: per mouse (thin) and mean over mice (bold).

    Args:
        records: Per-animal results, keyed by mouse.
        areas: Area labels.
        path: Output PNG.
        dpi: Figure resolution.
    """
    figure, axis = plt.subplots(figsize=(6, 4.5))
    for area in areas:
        color = plotting.area_color(area)
        curves = [r["areas"][str(area)]["size_curve"] for r in records.values()]
        for curve in curves:
            axis.plot([int(n) for n in curve], list(curve.values()), color=color,
                      alpha=0.25, linewidth=0.7)
        sizes = sorted({int(n) for curve in curves for n in curve})
        means = [np.mean([c[str(n)] for c in curves if str(n) in c]) for n in sizes]
        axis.plot(sizes, means, "-o", color=color, linewidth=2, markersize=5,
                  label=f"Area {area}")
    axis.set(xlabel="Units per subset (n)", ylabel="Dimensionality (k / n)", ylim=(0, None),
             title="Subset-size check (bold = mean over mice with ≥ n units)")
    axis.legend()
    plotting.save(figure, path, dpi)
