"""Plotting helpers shared by every question."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # batch jobs have no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AREA_COLORS = {1: "C0", 2: "C1", 3: "C2"}


def area_color(area: int | str) -> str:
    """Fixed colour of an area in every figure (area 1 blue, 2 orange, 3 green)."""
    return AREA_COLORS.get(int(area), "0.5")


def save(figure: Any, path: Path, dpi: int = 150) -> None:
    """Write a figure (creating its folder) and release its memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def summary_panel(axis: Any, values: np.ndarray, labels: list[str],
                  conditions: dict[str, dict[str, Any]], ylabel: str,
                  tick_labels: list[str] | None = None,
                  colors: list[str] | None = None) -> None:
    """Mean ± bootstrap 95% CI per condition, with one grey line per mouse.

    Args:
        axis: Matplotlib axis.
        values: ``[n_mice × n_conditions]`` mouse-level values.
        labels: Condition labels, in column order.
        conditions: ``summary["conditions"]`` from ``stats.summarize_conditions``;
            the plotted mean and CI are read from it, so the figure shows
            exactly the reported numbers.
        ylabel: Y-axis label (variable and units).
        tick_labels: Optional display names of the conditions.
        colors: Optional bar colours (default: area colours / C0–C2).

    Scientific rationale:
        Lines join the values of one mouse across conditions, which shows the
        paired structure that the Friedman and Wilcoxon tests use.
    """
    positions = np.arange(len(labels))
    means = np.array([conditions[label]["mean"] for label in labels])
    cis = np.array([conditions[label]["ci95"] for label in labels])
    colors = colors or [f"C{i}" for i in positions]
    axis.bar(positions, means, color=colors, alpha=0.45, width=0.6)
    for row in np.asarray(values, dtype=float):
        axis.plot(positions, row, "-o", color="0.45", markersize=3, linewidth=0.6, alpha=0.7)
    axis.errorbar(positions, means, yerr=[means - cis[:, 0], cis[:, 1] - means], fmt="o",
                  color="black", capsize=5, zorder=4)
    axis.set_xticks(positions, tick_labels or [f"Area {label}" for label in labels])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)


def plot_summary(values: np.ndarray, labels: list[str], summary: dict[str, Any], ylabel: str,
                 title: str, path: Path, tick_labels: list[str] | None = None,
                 log_scale: bool = False, reference: float | None = None) -> None:
    """Stand-alone summary figure of a question's final variable.

    Args:
        values: ``[n_mice × n_conditions]`` mouse-level values.
        labels: Condition labels.
        summary: Output of ``stats.summarize_conditions``.
        ylabel: Variable name and units.
        title: Figure title (question and answer).
        path: Output PNG.
        tick_labels: Optional display names of the conditions.
        log_scale: Log y-axis (for variables spanning orders of magnitude).
        reference: Optional null level drawn as a dashed line.
    """
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    summary_panel(axis, values, labels, summary["conditions"], ylabel, tick_labels)
    if log_scale:
        axis.set_yscale("log")
    if reference is not None:
        axis.axhline(reference, color="black", linestyle="--", linewidth=1)
    tests = summary["statistical_tests"]
    axis.set_title(f"{title}\n{summary['answer']} (Friedman p = {tests['p_value']:.2g}, "
                   f"n = {len(values)} mice)", fontsize=10)
    save(figure, path)
