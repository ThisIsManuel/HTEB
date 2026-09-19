"""Console summaries with equal weighting of transformations within each axis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .constants import AXES
from .dataset_preparation import filtering_changed, filtering_summary


def print_score_tables(groups: Sequence[Mapping[str, Any]]) -> None:
    """Print per-dataset and across-dataset scores with equally weighted axis totals."""
    if not groups:
        return
    models = list(dict.fromkeys(str(row["model"]) for row in groups))
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in groups))
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    originals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for group in groups:
        dataset, model = str(group["dataset"]), str(group["model"])
        originals[dataset, model].append(float(group["original"]))
        for transformation, score in group["transformations"].items():
            grouped[dataset, model, transformation].append(float(score))
    scores = {key: mean(values) for key, values in grouped.items()}
    details = Table(title="Scores per dataset", title_style="bold green", box=box.ROUNDED)
    totals = Table(
        title="Total HTEB scores (average across datasets)",
        title_style="bold green",
        box=box.ROUNDED,
    )
    for table in (details, totals):
        if table is details:
            table.add_column("Dataset")
        table.add_column("Axis / transformation")
        for model in models:
            table.add_column(Text(model), justify="right")

    def formatted(values: Sequence[float | None]) -> list[str]:
        return ["-" if value is None else f"{value:.2f}" for value in values]

    dataset_originals = {key: mean(values) for key, values in originals.items()}

    def add_score_rows(
        table: Table,
        values_by_model: Mapping[tuple[str, str], float],
        dataset: str | None = None,
    ) -> None:
        prefix = [] if dataset is None else [Text(dataset)]
        child_prefix = [] if dataset is None else [""]
        model_axes: dict[str, list[float]] = {model: [] for model in models}
        for axis, transformations in AXES.items():
            selected = [
                transformation
                for transformation in transformations
                if any((model, transformation) in values_by_model for model in models)
            ]
            if not selected:
                continue
            axis_scores: list[float | None] = []
            for model in models:
                values = [
                    values_by_model[model, transformation]
                    for transformation in selected
                    if (model, transformation) in values_by_model
                ]
                axis_score = mean(values) if values else None
                axis_scores.append(axis_score)
                if axis_score is not None:
                    model_axes[model].append(axis_score)
            table.add_row(*prefix, axis, *formatted(axis_scores), style="bold")
            for transformation in selected:
                table.add_row(
                    *child_prefix,
                    f"  {transformation}",
                    *formatted([values_by_model.get((model, transformation)) for model in models]),
                )
            table.add_section()
        axis_totals = [mean(model_axes[model]) if model_axes[model] else None for model in models]
        table.add_row(*prefix, "HTEB Total", *formatted(axis_totals), style="bold")
        table.add_section()

    for dataset in datasets:
        details.add_row(
            Text(dataset),
            "Original",
            *formatted([dataset_originals.get((dataset, model)) for model in models]),
            style="bold",
        )
        details.add_section()
        add_score_rows(
            details,
            {
                (model, transformation): score
                for (name, model, transformation), score in scores.items()
                if name == dataset
            },
            dataset,
        )
    original_means: dict[str, list[float]] = defaultdict(list)
    for (_, model), score in dataset_originals.items():
        original_means[model].append(score)
    totals.add_row(
        "Original", *formatted([mean(original_means[model]) for model in models]), style="bold"
    )
    totals.add_section()
    across_datasets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_, model, transformation), score in scores.items():
        across_datasets[model, transformation].append(score)
    add_score_rows(totals, {key: mean(values) for key, values in across_datasets.items()})
    console = Console()
    console.print()
    labels = list(
        dict.fromkeys(
            f"{group['dataset']}: {filtering_summary(report)}"
            for group in groups
            for report in group.get("data_filtering", {}).values()
            if filtering_changed(report)
        )
    )
    if labels:
        details.caption = "Filtered data: " + "; ".join(labels)
        totals.title = "HTEB scores including filtered datasets"
    console.print(details)
    console.print()
    console.print(totals)
