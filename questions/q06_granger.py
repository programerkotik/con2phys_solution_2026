"""Question 6: which brain area pair has the strongest directed functional connectivity?

Final variable: per mouse and direction X⇒Y, bivariate time-domain Granger
causality (nats) between one signal per area, the sign-aligned average of the
area's z-scored channels, from exact least-squares autoregressive fits of
fixed order 20 (40 ms) on the whole raw recording.

Run: ``python -m questions.q06_granger --animal 1`` then ``--aggregate``.
"""

from __future__ import annotations

from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from con2phys import cli, io, plotting, stats

KEY = "q06"
QUESTION = "Q6. Which brain area pair has the strongest directed functional connectivity?"


def zscore_rows(signals: np.ndarray) -> np.ndarray:
    """Standardise every row to zero mean and unit variance."""
    return (signals - signals.mean(axis=1, keepdims=True)) / signals.std(axis=1, keepdims=True)


def area_signal(lfp: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Sign-aligned average of all z-scored channels of one area.

    Args:
        lfp: ``[channels × samples]`` LFP of one area.

    Returns:
        The area signal ``(n_samples,)``, the sign applied to each channel
        (+1/−1), the reference channel index, and the within-area channel
        correlation matrix.

    Scientific rationale:
        Channels within an area are strongly correlated, so one average
        captures the shared activity. Where the LFP reverses polarity across
        depth, channels on opposite sides are anti-correlated and would cancel
        in a plain average; each channel is therefore flipped if it is
        negatively correlated with the reference channel (highest mean |r| to
        all others). Z-scoring first stops one high-amplitude site dominating.
    """
    z = zscore_rows(np.asarray(lfp, dtype=float))
    standardized = zscore_rows(z)  # as Pearson r of z; keeps the pre-refactor numbers exact
    correlations = standardized @ standardized.T / z.shape[1]
    reference = int(np.argmax(np.abs(correlations).mean(axis=1)))
    signs = np.where(correlations[reference] < 0, -1.0, 1.0)
    return (signs[:, None] * z).mean(axis=0), signs, reference, correlations


def lagged_design(signals: np.ndarray, order: int, start: int) -> np.ndarray:
    """Regressors predicting samples ``start:`` from lags 1..order of every signal, plus a constant."""
    n_samples = signals.shape[1]
    columns = [s[start - lag:n_samples - lag] for s in signals for lag in range(1, order + 1)]
    return np.column_stack(columns + [np.ones(n_samples - start)])


def residuals(targets: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Ordinary least-squares residuals of each target column."""
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    return targets - design @ coefficients


def granger_ols(pair: np.ndarray, order: int) -> dict[str, float]:
    """Bivariate time-domain Granger causality from exact least-squares fits.

    Args:
        pair: Two signals ``[2 × n_samples]``, X first and Y second.
        order: Autoregressive order shared by the full and restricted models.

    Returns:
        ``x_to_y`` and ``y_to_x`` (nats), ``instantaneous`` causality
        −ln(1 − ρ²) (nats) and ``residual_correlation`` ρ of the full model.

    Scientific rationale:
        X⇒Y = ln(variance of Y's residual from Y's own past / variance from
        both pasts): how much X's past improves the prediction of Y. This is
        Elephant's ``pairwise_granger`` quantity, estimated by least squares
        because Elephant's Yule–Walker estimate failed on these long, highly
        predictable signals. ρ measures what the two areas share at zero lag.
    """
    targets = pair[:, order:].T
    full = residuals(targets, lagged_design(pair, order, order))
    own_x = residuals(targets[:, 0], lagged_design(pair[:1], order, order))
    own_y = residuals(targets[:, 1], lagged_design(pair[1:], order, order))
    rho = float(np.corrcoef(full.T)[0, 1])
    return {"x_to_y": float(np.log(own_y.var() / full[:, 1].var())),
            "y_to_x": float(np.log(own_x.var() / full[:, 0].var())),
            "instantaneous": float(-np.log(1 - rho ** 2)), "residual_correlation": rho}


def granger_between_areas(signals: dict[int, np.ndarray], order: int) -> dict[str, Any]:
    """Granger causality in both directions for every pair of area signals.

    Returns:
        ``directed`` keyed ``"a>b"`` (a⇒b, nats); ``instantaneous`` and
        ``residual_correlation`` keyed ``"a-b"`` (a < b).

    Scientific rationale:
        The pair is z-scored before fitting, which does not change Granger
        causality but keeps the least-squares problem well scaled.
    """
    result: dict[str, Any] = {"directed": {}, "instantaneous": {}, "residual_correlation": {}}
    for a, b in combinations(sorted(signals), 2):
        fit = granger_ols(zscore_rows(np.vstack((signals[a], signals[b]))), order)
        result["directed"][f"{a}>{b}"] = fit["x_to_y"]
        result["directed"][f"{b}>{a}"] = fit["y_to_x"]
        result["instantaneous"][f"{a}-{b}"] = fit["instantaneous"]
        result["residual_correlation"][f"{a}-{b}"] = fit["residual_correlation"]
    return result


def area_signals(config: dict[str, Any], mouse: int) -> tuple[dict[int, np.ndarray],
                                                               dict[int, tuple]]:
    """Sign-aligned average signal of every area of one animal, plus its diagnostics."""
    folder = io.mouse_dir(config, mouse)
    fs = float(config["lfp_sampling_rate_hz"])
    signals, details = {}, {}
    for area in config["areas"]:
        lfp = io.load_lfp(io.lfp_path(folder, area), fs)
        signal, signs, reference, correlations = area_signal(lfp)
        signals[int(area)] = signal
        details[int(area)] = (signs, reference, correlations)
    return signals, details


def run_animal(config: dict[str, Any], mouse: int) -> dict[str, Any]:
    """Granger causality between the three area averages of one animal."""
    signals, details = area_signals(config, mouse)
    order = int(config[KEY]["granger_order"])
    record = granger_between_areas(signals, order)
    record["order"] = order
    record["flipped_channels"] = {str(a): int((d[0] < 0).sum()) for a, d in details.items()}
    record["within_area_mean_abs_r"] = {
        str(a): float(np.abs(d[2][~np.eye(len(d[0]), dtype=bool)]).mean())
        for a, d in details.items()}
    if mouse == int(config["example_mouse"]):
        plot_example(config, mouse, signals, details, order)
    return record


def order_sensitivity(config: dict[str, Any], mice: list[int], order: int) -> dict[str, Any]:
    """One-off check (not part of the pipeline): Granger at another model order.

    Scientific rationale:
        BIC chose the 20-lag ceiling everywhere, so the order is set by the cap.
        Refitting a few mice with a longer history shows whether the ranking
        of the questionnaire directions depends on it.
    """
    out: dict[str, Any] = {"order": order, "mice": {}}
    for mouse in mice:
        signals, _ = area_signals(config, mouse)
        out["mice"][str(mouse)] = granger_between_areas(signals, order)["directed"]
        cli.log(f"order {order} mouse {mouse}: {out['mice'][str(mouse)]}")
    return out


def aggregate(config: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Across-mouse summary of the three questionnaire directions, and figures."""
    labels = [f"{a}>{b}" for a, b in config[KEY]["questionnaire_directions"]]
    names = {label: "Brain area {} ⇒ Brain area {}".format(*label.split(">"))
             for label in labels}
    values = {m: [r["directed"][d] for d in labels] for m, r in records.items()}
    matrix = np.array(list(values.values()))
    summary = stats.summarize_conditions(matrix, labels, config, answer_labels=names)
    folder = cli.output_dir(config, KEY) / "figures"
    ticks = [label.replace(">", "⇒") for label in labels]
    plotting.plot_summary(matrix, labels, summary, "Granger causality (nats)", "Q6",
                          folder / "summary.png", tick_labels=ticks)
    plot_all_directions(records, [int(a) for a in config["areas"]],
                        folder / "all_directions_and_instantaneous.png")
    return cli.final_report(
        QUESTION, "bivariate least-squares Granger causality X⇒Y between sign-aligned "
        "z-scored channel averages, fixed order 20 (40 ms), whole raw recording", "nats",
        values, labels, summary,
        all_directions={m: r["directed"] for m, r in records.items()},
        instantaneous={m: r["instantaneous"] for m, r in records.items()},
        residual_correlation={m: r["residual_correlation"] for m, r in records.items()})


def plot_example(config: dict[str, Any], mouse: int, signals: dict[int, np.ndarray],
                 details: dict[int, tuple], order: int) -> None:
    """Step figures: sign alignment, area signals and residual cross-correlation."""
    fs = float(config["lfp_sampling_rate_hz"])
    folder = cli.output_dir(config, KEY) / "figures"
    figure, axes = plt.subplots(1, len(details), figsize=(15, 4), sharey=True)
    for axis, (area, (signs, reference, correlations)) in zip(axes, details.items()):
        channels = np.arange(len(signs))
        axis.plot(channels, correlations[reference], "o-", color="0.6", markersize=3,
                  label="before flip")
        axis.plot(channels, signs * correlations[reference], "o-", markersize=3,
                  color=plotting.area_color(area), label="after flip")
        axis.axvline(reference, color="black", linestyle=":", linewidth=1)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(xlabel="Channel (0 = deepest)", title=f"Area {area}: r to reference channel "
                 f"(dotted), {int((signs < 0).sum())} flipped")
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(fontsize=8)
    plotting.save(figure, folder / "example_sign_alignment.png")

    figure, axes = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    start = len(signals[1]) // 2
    window = slice(start, start + int(10 * fs))
    for k, (area, signal) in enumerate(signals.items()):
        axes[0].plot(np.arange(int(10 * fs)) / fs, signal[window] - 4 * k, linewidth=0.6,
                     color=plotting.area_color(area), label=f"Area {area}")
    axes[0].set(xlabel="Time (s)", yticks=[], title="Sign-aligned area averages (z units)")
    axes[0].legend(fontsize=8)
    pair = zscore_rows(np.vstack((signals[1], signals[3])))
    full = residuals(pair[:, order:].T, lagged_design(pair, order, order))
    lags = np.arange(-25, 26)
    xcorr = [np.corrcoef(full[max(0, -l):len(full) - max(0, l), 0],
                         full[max(0, l):len(full) - max(0, -l), 1])[0, 1] for l in lags]
    axes[1].stem(lags / fs * 1000, xcorr)
    axes[1].set(xlabel="Lag (ms; area 3 residual relative to area 1)", ylabel="r",
                title="Residual cross-correlation, areas 1 & 3")
    figure.suptitle(f"Mouse {mouse}")
    plotting.save(figure, folder / "example_signals_residuals.png")


def plot_all_directions(records: dict[int, dict[str, Any]], areas: list[int], path: Path) -> None:
    """All six Granger directions and the instantaneous causality per area pair, per mouse."""
    directions = [f"{a}>{b}" for a, b in permutations(areas, 2)]
    pairs = [f"{a}-{b}" for a, b in combinations(areas, 2)]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    for axis, keys, field in ((axes[0], directions, "directed"),
                              (axes[1], pairs, "instantaneous")):
        values = np.array([[r[field][k] for k in keys] for r in records.values()])
        for row in values:
            axis.plot(range(len(keys)), row, "o", color="0.45", markersize=3, alpha=0.7)
        axis.errorbar(range(len(keys)), values.mean(axis=0), yerr=values.std(axis=0), fmt="s",
                      color="black", capsize=4)
        axis.set_xticks(range(len(keys)), [k.replace(">", "⇒").replace("-", "⇔") for k in keys])
    axes[0].set(ylabel="Granger causality (nats)", title="All six directions (mean ± SD)")
    axes[1].set(ylabel="Instantaneous causality (nats)", title="Zero-lag sharing of residuals")
    plotting.save(figure, path)


if __name__ == "__main__":
    cli.main(KEY, run_animal, aggregate, __doc__)
