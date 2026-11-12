"""Command line and result storage shared by every question script.

Every question script in ``questions/`` exposes two functions:

* ``run_animal(config, mouse) -> dict`` analyses one animal and returns a
  JSON-serialisable record (written to ``<output_dir>/per_animal/mouse_XX.json``);
* ``aggregate(config, records) -> dict`` combines all animals into the final
  report (written to ``<output_dir>/final_report.json``).

``main`` turns such a script into a CLI with ``--animal N`` (one animal; animals
can run in parallel), ``--aggregate`` (after all animals) and ``--list-animals``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io import included_mice, read_config


def output_dir(config: dict[str, Any], key: str) -> Path:
    """Output folder of one question (``config[key]["output_dir"]``)."""
    return Path(config[key]["output_dir"])


def animal_path(config: dict[str, Any], key: str, mouse: int, suffix: str = "json") -> Path:
    """Per-animal result file of one question."""
    return output_dir(config, key) / "per_animal" / f"mouse_{mouse:02d}.{suffix}"


def _json_default(value: Any) -> Any:
    """Make NumPy scalars and arrays JSON-serialisable."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON-serialisable: {type(value)}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON, creating the folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=_json_default))


def collect(config: dict[str, Any], key: str) -> dict[int, dict[str, Any]]:
    """Every included animal's saved record, refusing to aggregate if one is missing."""
    records = {}
    for mouse in included_mice(config):
        path = animal_path(config, key, mouse)
        if not path.exists():
            raise FileNotFoundError(f"Missing per-animal result {path}; run --animal {mouse}")
        records[mouse] = json.loads(path.read_text())
    return records


def log(message: str) -> None:
    """Time-stamped progress line for the run log."""
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def final_report(question: str, variable: str, units: str, mouse_values: dict[int, list[float]],
                 labels: list[str], summary: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Uniform final report shared by all questions (the report builder reads this).

    Args:
        question: Questionnaire wording.
        variable: Definition of the final mouse-level variable.
        units: Units of that variable.
        mouse_values: Mouse → value per condition (column order = ``labels``).
        labels: Condition labels.
        summary: Output of ``stats.summarize_conditions``.
        **extra: Question-specific diagnostics (kept under ``extra``).
    """
    return {"question": question, "variable": variable, "units": units,
            "n_mice": len(mouse_values), "mice": sorted(mouse_values), "labels": labels,
            "mouse_values": {str(m): v for m, v in sorted(mouse_values.items())},
            **summary, "extra": extra}


def markdown_report(report: dict[str, Any]) -> str:
    """Short Markdown rendering of a final report (answer, table, tests)."""
    tests = report["statistical_tests"]
    lines = [f"# {report['question']}", "", f"**Answer:** {report['answer']} "
             f"(selected: {report['selected']}, supported = {report['supported']})", "",
             f"**Final variable:** {report['variable']} — units: {report['units']}; "
             f"n = {report['n_mice']} mice.", "",
             "| Condition | Mean | 95% CI |", "|---|---|---|"]
    for label, entry in report["conditions"].items():
        lines.append(f"| {label} | {entry['mean']:.4g} | [{entry['ci95'][0]:.4g}, "
                     f"{entry['ci95'][1]:.4g}] |")
    lines += ["", f"Friedman χ² = {tests['statistic']:.3g}, p = {tests['p_value']:.3g}", "",
              "| Pair | Mean difference | W | p | p (Holm) |", "|---|---|---|---|---|"]
    for c in tests["pairwise"]:
        lines.append(f"| {' vs '.join(c['conditions'])} | {c['mean_difference']:.4g} | "
                     f"{c['statistic']:.1f} | {c['p_value']:.3g} | {c['p_value_holm']:.3g} |")
    return "\n".join(lines) + "\n"


def save_final_report(config: dict[str, Any], key: str, report: dict[str, Any]) -> None:
    """Write ``final_report.json`` and ``final_report.md`` of one question."""
    folder = output_dir(config, key)
    write_json(folder / "final_report.json", report)
    (folder / "final_report.md").write_text(markdown_report(report))


def main(key: str, run_animal: Callable[[dict[str, Any], int], dict[str, Any]],
         aggregate: Callable[[dict[str, Any], dict[int, dict[str, Any]]], dict[str, Any]],
         description: str | None = None) -> None:
    """CLI entry point of a question script.

    Args:
        key: The question's section in ``config.yaml`` (e.g. ``"q01"``).
        run_animal: Per-animal analysis.
        aggregate: Across-animal summary returning the final report.
        description: Help text.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--animal", type=int, help="analyse one animal")
    stage.add_argument("--aggregate", action="store_true", help="combine all animals")
    stage.add_argument("--list-animals", action="store_true", help="print included animals")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.list_animals:
        print("\n".join(str(m) for m in included_mice(config)))
    elif args.animal is not None:
        record = run_animal(config, args.animal)
        write_json(animal_path(config, key, args.animal), {"mouse": args.animal, **record})
        log(f"{key} mouse {args.animal} done")
    else:
        report = aggregate(config, collect(config, key))
        save_final_report(config, key, report)
        log(f"{key}: {report['answer']}")
