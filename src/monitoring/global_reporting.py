"""Human- and machine-readable global monitoring outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _group_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Group | Drift | Regime | Error |",
        "|---|---|---|---|",
    ]
    for group in report["universe"]["groups"]:
        lines.append(
            f"| {group} | {report['feature_drift']['groups'][group]['status']} "
            f"| {report['volatility_regime']['groups'][group]['status']} "
            f"| {report['forecast_error']['groups'][group]['status']} |"
        )
    return lines


def global_report_markdown(report: dict[str, Any]) -> str:
    """Render one concise summary; ticker detail remains in JSON/CSV."""
    drift = report["feature_drift"]
    regime = report["volatility_regime"]
    error = report["forecast_error"]
    lines = [
        "# Global volatility monitoring",
        "",
        f"**Overall status:** `{report['status'].upper()}`  ",
        f"**As of:** {report['as_of']}  ",
        f"**Model:** `{report['model_version']}`  ",
        f"**Coverage:** {report['universe']['target_count']} targets + "
        f"{len(report['universe']['context'])} context series",
        "",
        "## Component status",
        "",
        "| Data quality | Feature drift | Regime | Delayed error |",
        "|---|---|---|---|",
        f"| {report['data_quality']['status']} | {drift['status']} "
        f"| {regime['status']} | {error['status']} |",
        "",
        "## Asset groups",
        "",
        *_group_table(report),
        "",
        "## Fleet summary",
        "",
        f"- Drift-affected targets: {drift['aggregate']['warning'] + drift['aggregate']['critical']}/"
        f"{drift['aggregate']['total']}",
        f"- Elevated-regime targets: {regime['aggregate']['warning'] + regime['aggregate']['critical']}/"
        f"{regime['aggregate']['total']}",
        f"- Error-affected targets: {error['aggregate']['warning'] + error['aggregate']['critical']}/"
        f"{error['aggregate']['total']}",
        f"- Micro RMSE/MAE/QLIKE: {error['micro']['rmse']:.4f} / "
        f"{error['micro']['mae']:.4f} / {error['micro']['qlike']:.4f}",
        f"- Recent/baseline micro RMSE ratio: "
        f"{error['micro']['recent_to_baseline_rmse_ratio']:.3f}",
        "",
        "Per-ticker metrics and forecasts are available in `report.json` and "
        "`predictions.csv`. One workflow annotation represents the global run.",
        "",
    ]
    return "\n".join(lines)


def write_global_outputs(
    output_dir: Path,
    report: dict[str, Any],
    predictions: pd.DataFrame,
) -> None:
    """Persist complete global monitoring evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "report.md").write_text(global_report_markdown(report))
    predictions.to_csv(output_dir / "predictions.csv", index=False)
