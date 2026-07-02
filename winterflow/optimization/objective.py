from __future__ import annotations

import numpy as np
import pandas as pd

from winterflow.optimization.actions import InterventionAction


def estimate_baseline_pressure(daily_demand_df: pd.DataFrame, hospitals_df: pd.DataFrame) -> dict[str, float]:
    """Estimate baseline operational pressure from generated demand."""

    demand = daily_demand_df.merge(
        hospitals_df[["hospital_id", "ed_cubicles", "general_beds", "icu_beds", "nurses"]],
        on="hospital_id",
        how="left",
    )
    demand["ed_pressure"] = demand["synthetic_arrivals"] / np.maximum(demand["ed_cubicles"] * 4.8, 1)
    demand["ward_pressure"] = demand["expected_admissions"] / np.maximum(demand["general_beds"] / 13, 1)
    demand["icu_pressure"] = demand["expected_icu_admissions"] / np.maximum(demand["icu_beds"] / 7, 1)
    demand["trolley_proxy"] = np.clip(
        (demand["ward_pressure"] - 0.82) * 36
        + (demand["combined_demand_multiplier"] - 1) * 18
        + demand["staff_absence_rate"] * 40,
        0,
        None,
    )
    demand["wait_proxy"] = np.clip(demand["ed_pressure"] * 3.4 + demand["trolley_proxy"] * 0.10, 0, None)
    demand["staff_proxy"] = np.clip(
        0.55 * demand["ward_pressure"] + 0.35 * demand["icu_pressure"] + demand["staff_absence_rate"] * 4,
        0,
        None,
    )
    return {
        "ed_overcrowding_score": _scale_0_100(float(demand["ed_pressure"].quantile(0.95)), 0.75, 1.40),
        "icu_pressure_score": _scale_0_100(float(demand["icu_pressure"].quantile(0.95)), 0.65, 1.35),
        "trolley_score": _scale_0_100(float(demand["trolley_proxy"].quantile(0.95)), 0, 45),
        "wait_time_score": _scale_0_100(float(demand["wait_proxy"].quantile(0.95)), 2, 8),
        "staff_stress_score": _scale_0_100(float(demand["staff_proxy"].quantile(0.95)), 0.70, 1.60),
    }


def project_action_package_metrics(
    baseline_metrics: dict[str, float],
    package: list[InterventionAction],
) -> dict[str, float]:
    """Apply fast deterministic approximations for an intervention package."""

    projected = baseline_metrics.copy()
    for action in package:
        if action.action_type == "open_surge_general_beds":
            projected["trolley_score"] -= action.amount * 0.75
            projected["wait_time_score"] -= action.amount * 0.20
            projected["staff_stress_score"] += action.amount * 0.04
        elif action.action_type == "open_surge_icu_beds":
            projected["icu_pressure_score"] -= action.amount * 3.6
            projected["staff_stress_score"] += action.amount * 0.20
        elif action.action_type == "add_temporary_nurses":
            projected["staff_stress_score"] -= action.amount * 1.2
            projected["trolley_score"] -= action.amount * 0.15
        elif action.action_type == "add_temporary_doctors":
            projected["ed_overcrowding_score"] -= action.amount * 2.0
            projected["wait_time_score"] -= action.amount * 1.5
        elif action.action_type == "reduce_elective_admissions":
            projected["trolley_score"] -= action.amount * 0.32
            projected["staff_stress_score"] -= action.amount * 0.08
        elif action.action_type == "accelerate_discharge":
            projected["trolley_score"] -= action.amount * 0.55
            projected["wait_time_score"] -= action.amount * 0.18
            projected["staff_stress_score"] -= action.amount * 0.10
        elif action.action_type == "create_ed_overflow_spaces":
            projected["ed_overcrowding_score"] -= action.amount * 2.4
            projected["wait_time_score"] -= action.amount * 0.6
        elif action.action_type == "transfer_patients":
            projected["trolley_score"] -= action.amount * 0.70
            projected["icu_pressure_score"] -= action.amount * 0.18
    return {key: float(np.clip(value, 0, 100)) for key, value in projected.items()}


def calculate_objective_score(
    metrics: dict[str, float],
    cost_points: float,
    budget: float,
) -> float:
    """Calculate the optimizer objective score; lower is better."""

    normalized_intervention_cost = 0.0 if budget <= 0 else min(100, cost_points / budget * 100)
    total_score = (
        0.25 * metrics["ed_overcrowding_score"]
        + 0.25 * metrics["icu_pressure_score"]
        + 0.20 * metrics["trolley_score"]
        + 0.15 * metrics["wait_time_score"]
        + 0.10 * metrics["staff_stress_score"]
        + 0.05 * normalized_intervention_cost
    )
    return round(float(np.clip(total_score, 0, 100)), 2)


def package_cost(package: list[InterventionAction]) -> float:
    return float(sum(action.cost_points for action in package))


def _scale_0_100(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low) * 100, 0, 100))

