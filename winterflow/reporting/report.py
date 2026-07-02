from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

SYNTHETIC_REPORT_DISCLAIMER = (
    "All default data is synthetic. WINTER-Flow is a decision-support demonstration only, "
    "not a clinical tool, and must not be used for patient-care decisions."
)


def build_executive_report_markdown(
    scenario_name: str,
    policy_assumptions: dict[str, Any] | pd.DataFrame | str | None = None,
    regional_kpis: dict[str, Any] | None = None,
    highest_risk_hospitals: pd.DataFrame | None = None,
    forecast_summary: dict[str, Any] | None = None,
    optimizer_recommendations: pd.DataFrame | None = None,
    before_after_comparison: pd.DataFrame | None = None,
    generated_at: datetime | None = None,
    title: str = "WINTER-Flow Executive Report",
) -> str:
    """Build a downloadable Markdown executive report."""

    generated_at = generated_at or datetime.now(timezone.utc)
    regional_kpis = regional_kpis or {}
    forecast_summary = forecast_summary or {}

    sections = [
        f"# {title}",
        "",
        f"**Date generated:** {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Scenario Summary",
        f"- Scenario: **{scenario_name}**",
        f"- Synthetic-data status: **synthetic by default**",
        "",
        "## Policy Assumptions",
        _format_policy_assumptions(policy_assumptions),
        "",
        "## Regional KPI Summary",
        _format_key_values(regional_kpis, empty_text="No simulation KPIs were supplied."),
        "",
        "## Highest-Risk Hospitals",
        _format_table(highest_risk_hospitals, max_rows=8, empty_text="No highest-risk hospital table was supplied."),
        "",
        "## Forecast Summary",
        _format_key_values(forecast_summary, empty_text="No forecast summary was supplied."),
        "",
        "## Optimizer Recommendations",
        _format_table(optimizer_recommendations, max_rows=8, empty_text="No optimizer recommendations were supplied."),
        "",
        "## Before/After Comparison",
        _format_table(before_after_comparison, max_rows=10, empty_text="No before/after comparison was supplied."),
        "",
        "## Limitations",
        "- Uses synthetic hospital, patient, demand, workforce, and operational-pressure data by default.",
        "- Model outputs are scenario-planning signals, not clinical predictions.",
        "- Forecast intervals and optimizer recommendations are generated from simplified synthetic assumptions.",
        "- Real deployment would require local validation, governance, calibration, security review, and clinical safety review.",
        "",
        "## Synthetic-Data Disclaimer",
        SYNTHETIC_REPORT_DISCLAIMER,
        "",
    ]
    return "\n".join(sections)


def metrics_to_csv(metrics_df: pd.DataFrame) -> str:
    """Export daily or aggregate metrics as CSV text."""

    return _safe_dataframe(metrics_df).to_csv(index=False)


def optimizer_plan_to_csv(plan_df: pd.DataFrame) -> str:
    """Export optimizer recommendations as CSV text."""

    return _safe_dataframe(plan_df).to_csv(index=False)


def create_report_bundle(
    scenario_name: str,
    policy_assumptions: dict[str, Any] | pd.DataFrame | str | None = None,
    regional_kpis: dict[str, Any] | None = None,
    highest_risk_hospitals: pd.DataFrame | None = None,
    forecast_summary: dict[str, Any] | None = None,
    optimizer_recommendations: pd.DataFrame | None = None,
    before_after_comparison: pd.DataFrame | None = None,
    daily_metrics: pd.DataFrame | None = None,
    patient_sample: pd.DataFrame | None = None,
    generated_at: datetime | None = None,
) -> dict[str, str]:
    """Create all downloadable report artifacts as in-memory text assets."""

    markdown = build_executive_report_markdown(
        scenario_name=scenario_name,
        policy_assumptions=policy_assumptions,
        regional_kpis=regional_kpis,
        highest_risk_hospitals=highest_risk_hospitals,
        forecast_summary=forecast_summary,
        optimizer_recommendations=optimizer_recommendations,
        before_after_comparison=before_after_comparison,
        generated_at=generated_at,
    )
    return {
        "markdown": markdown,
        "daily_metrics_csv": metrics_to_csv(_safe_dataframe(daily_metrics)),
        "optimizer_plan_csv": optimizer_plan_to_csv(_safe_dataframe(optimizer_recommendations)),
        "patient_sample_csv": _safe_dataframe(patient_sample).to_csv(index=False),
    }


def _format_policy_assumptions(policy_assumptions: dict[str, Any] | pd.DataFrame | str | None) -> str:
    if policy_assumptions is None:
        return "No policy assumptions were supplied."
    if isinstance(policy_assumptions, str):
        return policy_assumptions
    if isinstance(policy_assumptions, pd.DataFrame):
        return _format_table(policy_assumptions, max_rows=20, empty_text="No active policy assumptions.")
    if isinstance(policy_assumptions, dict):
        return _format_key_values(policy_assumptions, empty_text="No active policy assumptions.")
    return str(policy_assumptions)


def _format_key_values(values: dict[str, Any], empty_text: str) -> str:
    if not values:
        return empty_text
    lines = []
    for key, value in values.items():
        label = str(key).replace("_", " ").title()
        lines.append(f"- {label}: **{value}**")
    return "\n".join(lines)


def _format_table(df: pd.DataFrame | None, max_rows: int, empty_text: str) -> str:
    if df is None or df.empty:
        return empty_text
    display_df = df.head(max_rows).fillna("")
    columns = [str(column) for column in display_df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in display_df.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |")
    return "\n".join([header, separator, *rows])


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _safe_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame({"message": ["No data supplied"]})
    if df.empty:
        return pd.DataFrame({"message": ["No rows available"]})
    return df.copy()
