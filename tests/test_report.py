import pandas as pd

from winterflow.reporting.report import (
    SYNTHETIC_REPORT_DISCLAIMER,
    build_executive_report_markdown,
    create_report_bundle,
    metrics_to_csv,
    optimizer_plan_to_csv,
)


def test_report_is_generated_as_string() -> None:
    report = build_executive_report_markdown(
        scenario_name="severe_combined_surge",
        regional_kpis={"regional_risk_score": 82},
    )

    assert isinstance(report, str)
    assert len(report) > 100


def test_report_includes_scenario_name() -> None:
    report = build_executive_report_markdown(scenario_name="combined_winter_surge")

    assert "combined_winter_surge" in report


def test_report_includes_synthetic_data_disclaimer() -> None:
    report = build_executive_report_markdown(scenario_name="baseline")

    assert SYNTHETIC_REPORT_DISCLAIMER in report
    assert "not a clinical tool" in report


def test_csv_exports_are_non_empty() -> None:
    metrics = pd.DataFrame({"day": [0, 1], "risk_score": [40, 55]})
    plan = pd.DataFrame({"actions": ["Open 20 beds"], "objective_score": [61]})

    assert len(metrics_to_csv(metrics)) > 0
    assert len(optimizer_plan_to_csv(plan)) > 0


def test_report_bundle_contains_downloadable_assets() -> None:
    bundle = create_report_bundle(
        scenario_name="influenza_wave",
        daily_metrics=pd.DataFrame({"day": [0], "risk_score": [50]}),
        optimizer_recommendations=pd.DataFrame({"actions": ["Do nothing"], "objective_score": [70]}),
        patient_sample=pd.DataFrame({"patient_id": ["P001"]}),
    )

    assert set(bundle) == {"markdown", "daily_metrics_csv", "optimizer_plan_csv", "patient_sample_csv"}
    assert "influenza_wave" in bundle["markdown"]
    assert len(bundle["daily_metrics_csv"]) > 0

