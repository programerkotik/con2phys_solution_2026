"""Across-mouse statistics shared by every question.

Mice are the independent observations. Every question reduces its data to one
value per mouse and condition (area, area pair or trial segment), and this
module turns the resulting ``[n_mice × n_conditions]`` matrix into

* the mean of each condition with a percentile-bootstrap 95% CI over mice,
* a Friedman test across conditions,
* Holm-corrected paired Wilcoxon signed-rank tests between conditions,
* the questionnaire answer (see ``summarize_conditions`` for the rule).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


def bootstrap_mean_ci(values: np.ndarray, replicates: int, confidence_level: float,
                      seed: int) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval of a mean over mice.

    Args:
        values: One estimate per mouse.
        replicates: Number of bootstrap resamples.
        confidence_level: Central confidence level, e.g. 0.95.
        seed: Random seed.

    Returns:
        Lower and upper bound.

    Scientific rationale:
        Resampling mice estimates uncertainty across animals, the unit the
        questions generalise over.
    """
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.asarray(values, dtype=float), size=(replicates, len(values)),
                         replace=True)
    means = samples.mean(axis=1)
    alpha = (1 - confidence_level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down correction; adjusted p-values in the original order.

    Scientific rationale:
        Controls the family-wise error rate over the three contrasts with more
        power than Bonferroni.
    """
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def pairwise_wilcoxon(matrix: np.ndarray, labels: list[str], alpha: float
                      ) -> list[dict[str, Any]]:
    """Paired two-sided Wilcoxon signed-rank test for every pair of conditions, Holm-corrected.

    Args:
        matrix: ``[n_mice × n_conditions]``.
        labels: Condition labels.
        alpha: Family-wise significance level.

    Returns:
        One entry per pair: mean difference (first − second), W, raw and Holm p.
    """
    pairwise: list[dict[str, Any]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            differences = matrix[:, i] - matrix[:, j]
            if np.all(differences == 0):
                w, p = 0.0, 1.0
            else:
                result = wilcoxon(matrix[:, i], matrix[:, j], alternative="two-sided")
                w, p = float(result.statistic), float(result.pvalue)
            pairwise.append({"conditions": [labels[i], labels[j]],
                             "mean_difference": float(differences.mean()),
                             "statistic": w, "p_value": p})
    for comparison, adjusted in zip(pairwise, holm_adjust([c["p_value"] for c in pairwise])):
        comparison["p_value_holm"] = float(adjusted)
        comparison["significant"] = bool(adjusted < alpha)
    return pairwise


def repeated_measures_tests(values: np.ndarray, labels: list[str], alpha: float,
                            ranking: str = "highest") -> dict[str, Any]:
    """Friedman test, Holm-corrected paired Wilcoxon tests and the winning condition.

    Args:
        values: ``[n_mice × n_conditions]``; each row is one mouse.
        labels: Condition labels of the columns.
        alpha: Family-wise significance level.
        ranking: Whether the question asks for the ``"highest"`` or ``"lowest"`` mean.

    Returns:
        JSON-serialisable omnibus and pairwise results, the condition with the
        highest (or lowest) mean (``selected``) and whether it is
        ``supported``: Friedman significant and the selected condition differs
        in the expected direction from every other one after Holm correction.

    Scientific rationale:
        Non-parametric repeated-measures tests respect the matched design
        (every mouse contributes every condition) without assuming normality.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels) or matrix.shape[0] < 2:
        raise ValueError("values must be [n_mice >= 2, n_conditions]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite")
    if ranking not in {"highest", "lowest"}:
        raise ValueError("ranking must be 'highest' or 'lowest'")

    # Exact equality, not np.allclose: values such as LFP power in V² are far
    # below allclose's default absolute tolerance of 1e-8.
    if np.all(matrix == matrix[:, [0]]):
        statistic, p_value = 0.0, 1.0
    else:
        omnibus = friedmanchisquare(*matrix.T)
        statistic, p_value = float(omnibus.statistic), float(omnibus.pvalue)

    pairwise = pairwise_wilcoxon(matrix, labels, alpha)
    means = matrix.mean(axis=0)
    selected = labels[int(np.argmax(means) if ranking == "highest" else np.argmin(means))]
    sign = 1 if ranking == "highest" else -1
    beats_all = all(
        c["significant"] and sign * c["mean_difference"] * (1 if c["conditions"][0] == selected
                                                            else -1) > 0
        for c in pairwise if selected in c["conditions"]
    )
    return {"test": "friedman", "statistic": statistic, "p_value": p_value,
            "significant": bool(p_value < alpha), "alpha": float(alpha),
            "pairwise_test": "wilcoxon_signed_rank", "p_value_correction": "holm",
            "pairwise": pairwise, "ranking": ranking, "selected": selected,
            "supported": bool(p_value < alpha and beats_all)}


def test_against_reference(values: np.ndarray, labels: list[str], reference: float,
                           alpha: float) -> dict[str, dict[str, float]]:
    """One-sided Wilcoxon tests of each condition against a reference level, Holm-corrected.

    Args:
        values: ``[n_mice × n_conditions]``.
        labels: Condition labels.
        reference: Null level, e.g. 0 excess information or 0.5 chance accuracy.
        alpha: Family-wise significance level.

    Returns:
        Per condition the raw and Holm-adjusted p-value of ``value > reference``.

    Scientific rationale:
        A condition can only be the answer if the effect exists at all; this
        checks that before comparing conditions.
    """
    raw = [float(wilcoxon(column - reference, alternative="greater").pvalue)
           for column in np.asarray(values, dtype=float).T]
    holm = holm_adjust(raw)
    return {label: {"p_value": p, "p_value_holm": float(h), "significant": bool(h < alpha)}
            for label, p, h in zip(labels, raw, holm)}


def condition_estimates(values: np.ndarray, labels: list[str], settings: dict[str, Any]
                        ) -> dict[str, dict[str, Any]]:
    """Mean over mice and percentile-bootstrap CI of every condition (condition i: seed + i)."""
    replicates = int(settings["bootstrap_replicates"])
    confidence = float(settings["confidence_level"])
    seed = int(settings["random_seed"])
    return {label: {"mean": float(values[:, i].mean()),
                    "ci95": list(bootstrap_mean_ci(values[:, i], replicates, confidence, seed + i)),
                    "n_mice": int(values.shape[0])}
            for i, label in enumerate(labels)}


def summarize_conditions(values: np.ndarray, labels: list[str], settings: dict[str, Any],
                         ranking: str = "highest", answer_labels: dict[str, str] | None = None,
                         reference: float | None = None) -> dict[str, Any]:
    """Group-level summary and questionnaire answer for one question.

    Args:
        values: ``[n_mice × n_conditions]`` mouse-level values of the final variable.
        labels: Condition labels of the columns (e.g. ``"1"``, ``"1-2"``, ``"stimulus"``).
        settings: The top-level ``config.yaml`` (bootstrap replicates,
            confidence level, random seed, significance level).
        ranking: ``"highest"`` or ``"lowest"``, as the question asks.
        answer_labels: Questionnaire wording of each condition
            (default ``"Brain area <label>"``).
        reference: Optional null level; when given, the answer additionally
            requires at least one condition significantly above it.

    Returns:
        ``conditions`` (mean, 95% CI, n per condition), ``statistical_tests``,
        optional ``against_reference``, ``selected``, ``supported`` and ``answer``.

    Scientific rationale (decision rule, identical for every question):
        The condition with the highest (lowest) mean is the answer only if the
        Friedman test is significant **and** it differs from each of the other
        two conditions in the expected direction after Holm correction (and,
        where a null level exists, some condition exceeds it). Otherwise the
        answer is "Not enough data / no differences" — this includes the case
        of a significant Friedman test without a single clear winner.
    """
    values = np.asarray(values, dtype=float)
    alpha = float(settings["significance_level"])
    conditions = condition_estimates(values, labels, settings)
    tests = repeated_measures_tests(values, labels, alpha, ranking)
    summary: dict[str, Any] = {"conditions": conditions, "statistical_tests": tests}
    supported = tests["supported"]
    if reference is not None:
        summary["against_reference"] = test_against_reference(values, labels, reference, alpha)
        summary["reference"] = float(reference)
        supported = supported and any(r["significant"]
                                      for r in summary["against_reference"].values())
    names = answer_labels or {label: f"Brain area {label}" for label in labels}
    summary["selected"] = tests["selected"]
    summary["supported"] = bool(supported)
    summary["answer"] = names[tests["selected"]] if supported else \
        "Not enough data / no differences"
    return summary
