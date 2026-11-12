"""Figures for Question 12: decoding variable C in each trial segment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from con2phys import plotting

SEGMENT_LABELS = {
    "pre_stimulus": "Trial start ⇒\nStim start",
    "stimulus": "Stim start ⇒\nOutcome",
    "post_outcome": "Outcome ⇒\nTrial end",
}


def plot_per_mouse(records: dict[int, dict[str, Any]], names: list[str], chance: float,
                   path: Path, dpi: int) -> None:
    """Fold-level balanced accuracy per segment, one panel per mouse.

    Args:
        records: Per-animal results, keyed by mouse.
        names: Segment keys.
        chance: Chance level drawn as a dashed line.
        path: Output PNG.
        dpi: Figure resolution.
    """
    columns = 6
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.8 * columns, 2.6 * rows),
                                sharey=True, squeeze=False)
    for axis, (mouse, record) in zip(axes.ravel(), records.items()):
        for i, name in enumerate(names):
            folds = record["segments"][name]["fold_scores"]
            axis.scatter(np.full(len(folds), i), folds, color=f"C{i}", s=12, alpha=0.7)
            axis.plot([i - 0.25, i + 0.25], [np.mean(folds)] * 2, color="black")
        axis.axhline(chance, color="black", linestyle="--", linewidth=0.8)
        counts = "/".join(str(n) for n in record["class_counts"].values())
        axis.set_title(f"Mouse {mouse}: {counts} trials, {record['n_units']} units", fontsize=8)
        axis.set_xticks(range(len(names)), ["pre", "stim", "post"][:len(names)])
    for axis in axes.ravel()[len(records):]:
        axis.axis("off")
    figure.suptitle("Balanced accuracy on each of the 5 test folds (bar = mean; "
                    "title: trials C=0/C=2)")
    plotting.save(figure, path, dpi)


def plot_durations(records: dict[int, dict[str, Any]], names: list[str], path: Path,
                   dpi: int) -> None:
    """Distribution of segment durations per mouse, on a log scale.

    Args:
        records: Per-animal results, keyed by mouse.
        names: Segment keys.
        path: Output PNG.
        dpi: Figure resolution.
    """
    figure, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), sharey=True)
    for axis, name in zip(np.atleast_1d(axes), names):
        data = [record["segments"][name]["durations_s"] for record in records.values()]
        axis.boxplot(data, tick_labels=[str(m) for m in records], showfliers=True,
                     flierprops={"markersize": 2})
        axis.set_yscale("log")
        axis.set(title=SEGMENT_LABELS.get(name, name).replace("\n", " "), xlabel="Mouse")
    np.atleast_1d(axes)[0].set_ylabel("Segment duration (s)")
    figure.suptitle("Segment durations of the decoded trials (longer segments → less noisy rates)")
    plotting.save(figure, path, dpi)
