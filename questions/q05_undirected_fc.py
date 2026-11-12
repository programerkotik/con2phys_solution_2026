"""Question 5: which brain area pair has the strongest undirected functional connectivity?

Final variable: per mouse and area pair, the mean over all cross-area channel
pairs (the full channels_A × channels_B block) of the absolute zero-lag
Pearson correlation of the raw LFP over the whole recording (|r|, unitless).

Run: ``python -m questions.q05_undirected_fc --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from con2phys import cli, io, plotting, stats

KEY = "q05"
QUESTION = "Q5. Which brain area pair has the strongest undirected functional connectivity?"


def pearson_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation between every row of ``x`` and every row of ``y``.

    Args:
        x: ``[n_channels_a × n_samples]``.
        y: ``[n_channels_b × n_samples]``.

    Returns:
        ``[n_channels_a × n_channels_b]`` correlation matrix.

    Scientific rationale:
        Zero-lag linear correlation is the simplest measure of whether two
        sites co-fluctuate. It is computed for every cross-area channel pair,
        so no channel averaging blurs the result.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2D with the same number of samples")
    x_z = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    y_z = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, keepdims=True)
    return x_z @ y_z.T / x.shape[1]


def pair_labels(areas: list[int]) -> list[str]:
    """Area-pair labels ``"1-2"``, ``"1-3"``, ``"2-3"``."""
    return [f"{a}-{b}" for a, b in combinations(areas, 2)]


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """Mean |r| and mean signed r of every cross-area channel block of one animal.

    Scientific rationale:
        The absolute value counts anti-correlated channels as coupled
        (polarity reverses across depth); the signed mean is kept to show
        the sign pattern that a shared reference produces.
    """
    folder = io.mouse_dir(config, mouse)
    fs = float(config["lfp_sampling_rate_hz"])
    lfps = {int(a): io.load_lfp(io.lfp_path(folder, a), fs) for a in config["areas"]}
    if len({lfp.shape[1] for lfp in lfps.values()}) != 1:
        raise ValueError(f"Area recordings have different lengths for mouse {mouse}")
    record: dict[str, Any] = {"n_channels": {str(a): int(l.shape[0]) for a, l in lfps.items()},
                              "pairs": {}}
    for a, b in combinations(sorted(lfps), 2):
        block = pearson_matrix(lfps[a], lfps[b])
        record["pairs"][f"{a}-{b}"] = {"mean_abs_r": float(np.abs(block).mean()),
                                       "mean_r": float(block.mean()),
                                       "n_channel_pairs": int(block.size)}
    if mouse == int(config["example_mouse"]):
        plot_example(config, mouse, lfps, fs)
    return record


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary, decision and figures."""
    labels = pair_labels([int(a) for a in config["areas"]])
    names = {label: "Brain area {} ⇔ Brain area {}".format(*label.split("-")) for label in labels}
    values = {m: [r["pairs"][p]["mean_abs_r"] for p in labels] for m, r in records.items()}
    matrix = np.array(list(values.values()))
    summary = stats.summarize_conditions(matrix, labels, config, answer_labels=names)
    folder = cli.output_dir(config, KEY) / "figures"
    ticks = [label.replace("-", "⇔") for label in labels]
    plotting.plot_summary(matrix, labels, summary, "Mean |r| over cross-area channel pairs",
                          "Q5", folder / "summary.png", tick_labels=ticks)
    signed = np.array([[r["pairs"][p]["mean_r"] for p in labels] for r in records.values()])
    plot_signed(signed, ticks, folder / "signed_r_per_mouse.png")
    return cli.final_report(QUESTION, "mean over all cross-area channel pairs of |Pearson r| "
                            "of the raw LFP (whole recording)", "|r| (unitless)", values,
                            labels, summary,
                            mean_signed_r={m: [r["pairs"][p]["mean_r"] for p in labels]
                                           for m, r in records.items()})


def plot_example(config: dict[str, Any], mouse: int, lfps: dict[int, np.ndarray],
                 fs: float) -> None:
    """Step figures for one mouse: raw traces of all areas and the full correlation matrix."""
    folder = cli.output_dir(config, KEY) / "figures"
    start = lfps[min(lfps)].shape[1] // 2
    window = slice(start, start + int(10 * fs))
    figure, axis = plt.subplots(figsize=(13, 7))
    offset = 0.0
    for area, lfp in lfps.items():
        channels = np.linspace(0, lfp.shape[0] - 1, 4).astype(int)
        spacing = 5 * np.median(lfp[:, window].std(axis=1))
        for channel in channels:
            trace = lfp[channel, window] - lfp[channel, window].mean()
            axis.plot(np.arange(trace.size) / fs, trace + offset,
                      color=plotting.area_color(area), linewidth=0.6)
            offset -= spacing
    axis.set(xlabel="Time (s, 10 s from the middle of the recording)", yticks=[],
             title=f"Mouse {mouse}: raw LFP, 4 channels per area (blue 1, orange 2, green 3)")
    plotting.save(figure, folder / "example_traces.png")

    stacked = np.vstack([lfps[a] for a in sorted(lfps)])
    full = pearson_matrix(stacked, stacked)
    plot_full_matrix(full, [lfps[a].shape[0] for a in sorted(lfps)], sorted(lfps),
                     f"Mouse {mouse}", folder / "example_matrix.png")


def plot_full_matrix(full: np.ndarray, sizes: list[int], areas: list[int], title: str,
                     path: Path) -> None:
    """Signed channel × channel r across all areas, and the |r| of each cross-area block."""
    edges = np.cumsum([0, *sizes])
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    image = axes[0].imshow(full, cmap="RdBu_r", vmin=-1, vmax=1)
    for edge in edges[1:-1]:
        axes[0].axhline(edge - 0.5, color="black", linewidth=1)
        axes[0].axvline(edge - 0.5, color="black", linewidth=1)
    centres = (edges[:-1] + edges[1:]) / 2
    axes[0].set_xticks(centres, [f"Area {a}" for a in areas])
    axes[0].set_yticks(centres, [f"Area {a}" for a in areas])
    axes[0].set_title("Channel × channel r (deep → superficial within area)")
    figure.colorbar(image, ax=axes[0], shrink=0.8)
    for i, j in combinations(range(len(areas)), 2):
        block = np.abs(full[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]).ravel()
        axes[1].hist(block, bins=np.linspace(0, 1, 51), histtype="step", linewidth=1.5,
                     label=f"{areas[i]}⇔{areas[j]}: mean {block.mean():.3f}")
    axes[1].set(xlabel="|r| of a cross-area channel pair", ylabel="Channel pairs",
                title="Cross-area blocks")
    axes[1].legend(fontsize=8)
    figure.suptitle(title)
    plotting.save(figure, path)


def plot_signed(signed: np.ndarray, ticks: list[str], path: Path) -> None:
    """Mean signed r per area pair, one line per mouse."""
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    for row in signed:
        axis.plot(range(len(ticks)), row, "-o", color="0.45", markersize=3, linewidth=0.6)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(ticks)), ticks)
    axis.set(ylabel="Mean signed r over channel pairs",
             title="Sign of cross-area coupling, one line per mouse")
    plotting.save(figure, path)


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
