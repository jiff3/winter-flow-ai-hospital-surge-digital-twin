import pandas as pd

from winterflow.data.hospitals import get_default_hospitals
from winterflow.data.scenarios import ScenarioConfig
from winterflow.optimization.policy import (
    PolicyControls,
    apply_policy_to_hospitals,
    apply_policy_to_scenario_config,
    build_before_after_comparison,
    build_policy_resource_overrides,
    summarize_simulation_outputs,
)


def test_policy_modifiers_apply_to_scenario_config() -> None:
    config = ScenarioConfig(scenario="combined_winter_surge", n_days=30)
    controls = PolicyControls(
        virus_intensity_multiplier=1.2,
        rsv_peak_shift_days=-14,
        staff_availability_reduction_pct=15,
    )

    updated = apply_policy_to_scenario_config(config, controls)

    assert updated.virus_intensity_multiplier == 1.2
    assert updated.rsv_peak_shift_days == -14
    assert updated.staff_absence_multiplier == 1.15


def test_policy_resource_overrides_add_surge_beds_and_reduce_staff_when_requested() -> None:
    hospitals = get_default_hospitals(seed=1).head(2)
    controls = PolicyControls(open_surge_beds=20, staff_availability_reduction_pct=20)

    overrides = build_policy_resource_overrides(hospitals, controls)

    total_policy_beds = sum(value["ward_beds"] for value in overrides.values())
    assert total_policy_beds == int(hospitals["general_beds"].sum()) + 20
    for hospital in hospitals.to_dict("records"):
        hospital_id = hospital["hospital_id"]
        assert overrides[hospital_id]["inpatient_nurses"] <= round(int(hospital["nurses"]) / 8)


def test_discharge_acceleration_reduces_delay_assumptions() -> None:
    hospitals = get_default_hospitals(seed=2)
    controls = PolicyControls(discharge_acceleration_pct=20)

    updated = apply_policy_to_hospitals(hospitals, controls)

    assert updated["average_discharge_delay_days"].mean() < hospitals["average_discharge_delay_days"].mean()


def test_before_after_comparison_calculates_direction_and_percent_change() -> None:
    baseline = {
        "mean_ed_wait": 5.0,
        "p90_ed_wait": 9.0,
        "peak_trolley_count": 100.0,
        "peak_ward_occupancy": 98.0,
        "peak_icu_occupancy": 92.0,
        "mean_staff_stress": 1.2,
        "regional_risk_score": 80.0,
        "regional_risk_label": "Red",
    }
    policy = {
        "mean_ed_wait": 4.0,
        "p90_ed_wait": 7.0,
        "peak_trolley_count": 75.0,
        "peak_ward_occupancy": 90.0,
        "peak_icu_occupancy": 91.0,
        "mean_staff_stress": 1.1,
        "regional_risk_score": 62.0,
        "regional_risk_label": "Amber",
    }

    comparison = build_before_after_comparison(baseline, policy)
    trolley_row = comparison.loc[comparison["metric"] == "Peak trolley count"].iloc[0]

    assert trolley_row["percent_change"] == -25.0
    assert trolley_row["direction"] == "Better"


def test_summarize_simulation_outputs_returns_regional_metrics() -> None:
    patients = pd.DataFrame({"ed_wait_hours": [1.0, 3.0, 5.0]})
    daily = pd.DataFrame(
        {
            "day": [0, 0, 1],
            "max_trolley_count": [5, 6, 3],
            "max_ward_occupancy_pct": [88.0, 92.0, 85.0],
            "max_icu_occupancy_pct": [78.0, 82.0, 80.0],
            "mean_staff_stress": [1.0, 1.2, 0.9],
            "risk_score": [40.0, 60.0, 45.0],
        }
    )

    summary = summarize_simulation_outputs(patients, daily)

    assert summary["peak_trolley_count"] == 11
    assert summary["regional_risk_score"] == 50

