from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from winterflow.data.scenarios import ScenarioConfig, apply_policy_modifiers
from winterflow.simulation.metrics import risk_label


@dataclass(frozen=True)
class PolicyControls:
    """User-facing policy levers for the sandbox comparison."""

    virus_intensity_multiplier: float = 1.0
    rsv_peak_shift_days: int = 0
    flu_peak_shift_days: int = 0
    covid_peak_shift_days: int = 0
    staff_availability_reduction_pct: float = 0.0
    open_surge_beds: int = 0
    open_surge_icu_beds: int = 0
    temporary_nurses: int = 0
    temporary_doctors: int = 0
    reduce_elective_admissions_pct: float = 0.0
    discharge_acceleration_pct: float = 0.0
    transfer_capacity: int = 0


POLICY_PRESETS: dict[str, PolicyControls] = {
    "RSV peaks two weeks earlier": PolicyControls(rsv_peak_shift_days=-14),
    "Staff availability drops by 15%": PolicyControls(staff_availability_reduction_pct=15),
    "Open 20 surge beds": PolicyControls(open_surge_beds=20),
    "Flu and COVID peak together": PolicyControls(flu_peak_shift_days=-13, covid_peak_shift_days=13),
    "Delayed discharges increase by 25%": PolicyControls(discharge_acceleration_pct=-25),
    "Improve discharge speed by 20%": PolicyControls(discharge_acceleration_pct=20),
}


def neutral_policy_controls() -> PolicyControls:
    return PolicyControls()


def apply_policy_to_scenario_config(config: ScenarioConfig, controls: PolicyControls) -> ScenarioConfig:
    """Apply demand-side policy controls to a scenario config."""

    return apply_policy_modifiers(
        config,
        {
            "virus_intensity_multiplier": controls.virus_intensity_multiplier,
            "rsv_peak_shift_days": controls.rsv_peak_shift_days,
            "flu_peak_shift_days": controls.flu_peak_shift_days,
            "covid_peak_shift_days": controls.covid_peak_shift_days,
            "staff_absence_multiplier": 1 + controls.staff_availability_reduction_pct / 100,
        },
    )


def apply_policy_to_hospitals(hospitals_df: pd.DataFrame, controls: PolicyControls) -> pd.DataFrame:
    """Apply discharge and elective-flow assumptions to hospital profiles."""

    policy_hospitals = hospitals_df.copy()
    delay_factor = max(0.15, 1 - controls.discharge_acceleration_pct / 100)
    policy_hospitals["discharge_delay_rate"] = (
        policy_hospitals["discharge_delay_rate"].astype(float) * delay_factor
    ).clip(0, 0.75)
    policy_hospitals["average_discharge_delay_days"] = (
        policy_hospitals["average_discharge_delay_days"].astype(float) * delay_factor
    ).clip(lower=0.05)
    policy_hospitals["elective_beds_per_day"] = (
        policy_hospitals["elective_beds_per_day"].astype(float)
        * max(0, 1 - controls.reduce_elective_admissions_pct / 100)
    ).round().astype(int)
    return policy_hospitals


def build_policy_resource_overrides(
    hospitals_df: pd.DataFrame,
    controls: PolicyControls,
    baseline_daily_metrics_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, int]]:
    """Build resource overrides for the SimPy engine from policy levers."""

    hospital_ids = [str(hospital_id) for hospital_id in hospitals_df["hospital_id"]]
    priority_ids = _priority_hospital_ids(hospitals_df, baseline_daily_metrics_df)
    surge_bed_allocation = _allocate_integer_total(controls.open_surge_beds, priority_ids[:2] or hospital_ids)
    surge_icu_allocation = _allocate_integer_total(controls.open_surge_icu_beds, priority_ids[:2] or hospital_ids)
    transfer_allocation = _allocate_integer_total(controls.transfer_capacity, priority_ids[:3] or hospital_ids)
    nurse_allocation = _allocate_integer_total(controls.temporary_nurses, priority_ids or hospital_ids)
    doctor_allocation = _allocate_integer_total(controls.temporary_doctors, priority_ids or hospital_ids)
    staff_factor = max(0.45, 1 - controls.staff_availability_reduction_pct / 100)

    overrides: dict[str, dict[str, int]] = {}
    for hospital in hospitals_df.to_dict("records"):
        hospital_id = str(hospital["hospital_id"])
        base_ward_beds = int(hospital["general_beds"])
        base_icu_beds = int(hospital["icu_beds"])
        base_ed_cubicles = int(hospital["ed_cubicles"])
        base_nurses = int(hospital["nurses"])
        elective_bed_release = int(
            round(float(hospital["elective_beds_per_day"]) * controls.reduce_elective_admissions_pct / 100 * 3)
        )
        base_inpatient_nurses = max(1, round(base_nurses / 8))
        base_ed_doctors = max(1, round(base_ed_cubicles / 4))
        base_triage_nurses = max(1, round(base_ed_cubicles / 8))

        overrides[hospital_id] = {
            "ward_beds": max(
                1,
                base_ward_beds
                + elective_bed_release
                + surge_bed_allocation.get(hospital_id, 0)
                + transfer_allocation.get(hospital_id, 0),
            ),
            "icu_beds": max(1, base_icu_beds + surge_icu_allocation.get(hospital_id, 0)),
            "inpatient_nurses": max(1, round(base_inpatient_nurses * staff_factor) + nurse_allocation.get(hospital_id, 0)),
            "ed_doctors": max(1, round(base_ed_doctors * staff_factor) + doctor_allocation.get(hospital_id, 0)),
            "triage_nurses": max(1, round(base_triage_nurses * staff_factor) + round(nurse_allocation.get(hospital_id, 0) / 5)),
            "discharge_team_capacity": max(
                1,
                round(base_ward_beds / 75 * max(0.5, 1 + controls.discharge_acceleration_pct / 100)),
            ),
        }
    return overrides


def summarize_simulation_outputs(patient_results_df: pd.DataFrame, daily_metrics_df: pd.DataFrame) -> dict[str, float | str]:
    """Summarize simulation outputs into policy comparison metrics."""

    if patient_results_df.empty or daily_metrics_df.empty:
        return {
            "mean_ed_wait": 0.0,
            "p90_ed_wait": 0.0,
            "peak_trolley_count": 0.0,
            "peak_ward_occupancy": 0.0,
            "peak_icu_occupancy": 0.0,
            "mean_staff_stress": 0.0,
            "regional_risk_score": 0.0,
            "regional_risk_label": "Green",
        }

    regional_trolley = daily_metrics_df.groupby("day")["max_trolley_count"].sum()
    regional_risk_score = float(daily_metrics_df.groupby("day")["risk_score"].mean().max())
    return {
        "mean_ed_wait": float(patient_results_df["ed_wait_hours"].mean()),
        "p90_ed_wait": float(np.percentile(patient_results_df["ed_wait_hours"], 90)),
        "peak_trolley_count": float(regional_trolley.max()),
        "peak_ward_occupancy": float(daily_metrics_df["max_ward_occupancy_pct"].max()),
        "peak_icu_occupancy": float(daily_metrics_df["max_icu_occupancy_pct"].max()),
        "mean_staff_stress": float(daily_metrics_df["mean_staff_stress"].mean()),
        "regional_risk_score": regional_risk_score,
        "regional_risk_label": risk_label(regional_risk_score),
    }


def build_before_after_comparison(
    baseline_summary: dict[str, float | str],
    policy_summary: dict[str, float | str],
) -> pd.DataFrame:
    """Create a before/after metric table from simulation summaries."""

    metric_labels = {
        "mean_ed_wait": "Mean ED wait",
        "p90_ed_wait": "P90 ED wait",
        "peak_trolley_count": "Peak trolley count",
        "peak_ward_occupancy": "Peak ward occupancy",
        "peak_icu_occupancy": "Peak ICU occupancy",
        "mean_staff_stress": "Mean staff stress",
        "regional_risk_score": "Regional risk score",
    }
    rows: list[dict[str, object]] = []
    for key, label in metric_labels.items():
        baseline_value = float(baseline_summary[key])
        policy_value = float(policy_summary[key])
        absolute_change = policy_value - baseline_value
        percent_change = 0.0 if baseline_value == 0 else absolute_change / baseline_value * 100
        rows.append(
            {
                "metric": label,
                "baseline": round(baseline_value, 2),
                "policy": round(policy_value, 2),
                "absolute_change": round(absolute_change, 2),
                "percent_change": round(percent_change, 1),
                "direction": _impact_direction(key, absolute_change),
            }
        )
    return pd.DataFrame(rows)


def interpret_policy_impact(
    controls: PolicyControls,
    comparison_df: pd.DataFrame,
    baseline_summary: dict[str, float | str],
    policy_summary: dict[str, float | str],
) -> str:
    """Generate a short plain-English policy interpretation."""

    trolley_row = comparison_df.loc[comparison_df["metric"] == "Peak trolley count"].iloc[0]
    risk_before = str(baseline_summary["regional_risk_label"])
    risk_after = str(policy_summary["regional_risk_label"])
    trolley_change = float(trolley_row["percent_change"])
    phrase = describe_policy_controls(controls)
    direction = "reduced" if trolley_change < 0 else "increased"
    return (
        f"{phrase} {direction} peak trolley count by {abs(trolley_change):.1f}% "
        f"and moved regional risk from {risk_before} to {risk_after}."
    )


def describe_policy_controls(controls: PolicyControls) -> str:
    """Describe the dominant active policy lever in plain English."""

    if controls.open_surge_beds:
        return f"Opening {controls.open_surge_beds} surge beds across the highest-risk hospitals"
    if controls.open_surge_icu_beds:
        return f"Opening {controls.open_surge_icu_beds} surge ICU beds"
    if controls.discharge_acceleration_pct > 0:
        return f"Improving discharge speed by {controls.discharge_acceleration_pct:.0f}%"
    if controls.discharge_acceleration_pct < 0:
        return f"Increasing delayed discharges by {abs(controls.discharge_acceleration_pct):.0f}%"
    if controls.staff_availability_reduction_pct:
        return f"Reducing staff availability by {controls.staff_availability_reduction_pct:.0f}%"
    if controls.rsv_peak_shift_days:
        return f"Moving the RSV peak by {controls.rsv_peak_shift_days} days"
    if controls.flu_peak_shift_days or controls.covid_peak_shift_days:
        return "Changing the relative timing of flu and COVID peaks"
    if controls.temporary_nurses or controls.temporary_doctors:
        return "Adding temporary clinical staff"
    if controls.transfer_capacity:
        return f"Adding transfer capacity for {controls.transfer_capacity} patients"
    if controls.virus_intensity_multiplier != 1.0:
        return f"Changing virus intensity to {controls.virus_intensity_multiplier:.2f}x"
    return "The selected policy package"


def _priority_hospital_ids(
    hospitals_df: pd.DataFrame,
    baseline_daily_metrics_df: pd.DataFrame | None,
) -> list[str]:
    if baseline_daily_metrics_df is not None and not baseline_daily_metrics_df.empty:
        return (
            baseline_daily_metrics_df.groupby("hospital_id")["risk_score"]
            .max()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
    return (
        hospitals_df.sort_values("baseline_ed_arrivals_per_day", ascending=False)["hospital_id"]
        .astype(str)
        .tolist()
    )


def _allocate_integer_total(total: int, hospital_ids: list[str]) -> dict[str, int]:
    total = max(0, int(total))
    if total == 0 or not hospital_ids:
        return {}
    allocation = {hospital_id: total // len(hospital_ids) for hospital_id in hospital_ids}
    remainder = total % len(hospital_ids)
    for hospital_id in hospital_ids[:remainder]:
        allocation[hospital_id] += 1
    return allocation


def _impact_direction(metric_key: str, absolute_change: float) -> str:
    if abs(absolute_change) < 1e-9:
        return "Neutral"
    lower_is_better = {
        "mean_ed_wait",
        "p90_ed_wait",
        "peak_trolley_count",
        "peak_ward_occupancy",
        "peak_icu_occupancy",
        "mean_staff_stress",
        "regional_risk_score",
    }
    if metric_key in lower_is_better:
        return "Better" if absolute_change < 0 else "Worse"
    return "Better" if absolute_change > 0 else "Worse"

