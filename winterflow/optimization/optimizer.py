from __future__ import annotations

import itertools
import random

import pandas as pd

from winterflow.optimization.actions import ACTION_TYPES, InterventionAction, generate_feasible_actions
from winterflow.optimization.objective import (
    calculate_objective_score,
    estimate_baseline_pressure,
    package_cost,
    project_action_package_metrics,
)


def generate_candidate_packages(
    actions: list[InterventionAction],
    budget: float,
    max_actions: int = 3,
    max_candidates: int = 120,
    seed: int = 42,
) -> list[list[InterventionAction]]:
    """Generate feasible intervention packages for candidate search."""

    do_nothing = [action for action in actions if action.action_type == "do_nothing"]
    non_zero_actions = [action for action in actions if action.action_type != "do_nothing"]
    packages: list[list[InterventionAction]] = [do_nothing[:1]]

    for action in non_zero_actions:
        if action.cost_points <= budget:
            packages.append([action])

    rng = random.Random(seed)
    action_pool = non_zero_actions.copy()
    rng.shuffle(action_pool)
    for size in range(2, max_actions + 1):
        for combo in itertools.combinations(action_pool[: min(len(action_pool), 45)], size):
            package = list(combo)
            if package_cost(package) <= budget and _no_duplicate_action_types_for_hospital(package):
                packages.append(package)
            if len(packages) >= max_candidates:
                return packages
    return packages[:max_candidates]


def optimize_resource_allocation(
    daily_demand_df: pd.DataFrame,
    hospitals_df: pd.DataFrame,
    budget: float = 100,
    allowed_action_types: list[str] | tuple[str, ...] | None = None,
    max_actions: int = 3,
    max_candidates: int = 120,
    top_n: int = 5,
) -> dict[str, object]:
    """Search feasible intervention packages and rank recommendations."""

    allowed = allowed_action_types or ACTION_TYPES
    actions = generate_feasible_actions(hospitals_df, budget=budget, allowed_action_types=allowed)
    packages = generate_candidate_packages(actions, budget=budget, max_actions=max_actions, max_candidates=max_candidates)
    baseline_metrics = estimate_baseline_pressure(daily_demand_df, hospitals_df)
    baseline_score = calculate_objective_score(baseline_metrics, cost_points=0, budget=budget)

    rows: list[dict[str, object]] = []
    for package in packages:
        cost = package_cost(package)
        projected_metrics = project_action_package_metrics(baseline_metrics, package)
        objective_score = calculate_objective_score(projected_metrics, cost_points=cost, budget=budget)
        rows.append(
            {
                "actions": _package_description(package),
                "hospital_id": _package_hospitals(package),
                "objective_score": objective_score,
                "baseline_score": baseline_score,
                "projected_score": objective_score,
                "risk_reduction": round(baseline_score - objective_score, 2),
                "estimated_cost_points": round(cost, 2),
                "rationale": _rationale(package, baseline_score, objective_score),
            }
        )

    recommendations = pd.DataFrame(rows).sort_values(
        ["objective_score", "estimated_cost_points"],
        ascending=[True, True],
    ).head(top_n)
    best = recommendations.iloc[0]
    return {
        "recommendations": recommendations.reset_index(drop=True),
        "baseline_score": float(baseline_score),
        "projected_score": float(best["projected_score"]),
        "risk_reduction": float(best["risk_reduction"]),
        "estimated_cost_points": float(best["estimated_cost_points"]),
        "rationale_text": str(best["rationale"]),
        "actions": actions,
    }


def _no_duplicate_action_types_for_hospital(package: list[InterventionAction]) -> bool:
    seen: set[tuple[str, str]] = set()
    for action in package:
        key = (action.hospital_id, action.action_type)
        if key in seen:
            return False
        seen.add(key)
    return True


def _package_description(package: list[InterventionAction]) -> str:
    if not package or package[0].action_type == "do_nothing":
        return "Do nothing"
    return "; ".join(action.description for action in package)


def _package_hospitals(package: list[InterventionAction]) -> str:
    hospital_ids = sorted({action.hospital_id for action in package if action.hospital_id != "network"})
    return ", ".join(hospital_ids) if hospital_ids else "network"


def _rationale(package: list[InterventionAction], baseline_score: float, objective_score: float) -> str:
    if not package or package[0].action_type == "do_nothing":
        return "Baseline comparator with no additional intervention."
    reduction = baseline_score - objective_score
    dominant = package[0]
    return (
        f"{dominant.description} anchors the package because it targets the largest modeled pressure component. "
        f"Projected objective score improves by {reduction:.1f} points."
    )

