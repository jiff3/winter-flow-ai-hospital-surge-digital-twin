from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


ACTION_TYPES = (
    "do_nothing",
    "open_surge_general_beds",
    "open_surge_icu_beds",
    "add_temporary_nurses",
    "add_temporary_doctors",
    "reduce_elective_admissions",
    "accelerate_discharge",
    "create_ed_overflow_spaces",
    "transfer_patients",
)


@dataclass(frozen=True)
class InterventionAction:
    action_id: str
    hospital_id: str
    action_type: str
    amount: int
    cost_points: float
    description: str
    effects: dict[str, Any]


def do_nothing_action() -> InterventionAction:
    return InterventionAction(
        action_id="DO_NOTHING",
        hospital_id="network",
        action_type="do_nothing",
        amount=0,
        cost_points=0.0,
        description="Do nothing",
        effects={},
    )


def generate_feasible_actions(
    hospitals_df: pd.DataFrame,
    budget: float = 100,
    allowed_action_types: list[str] | tuple[str, ...] | None = None,
) -> list[InterventionAction]:
    """Generate feasible single-hospital intervention actions."""

    allowed = set(allowed_action_types or ACTION_TYPES)
    actions = [do_nothing_action()]
    for hospital in hospitals_df.to_dict("records"):
        hospital_id = str(hospital["hospital_id"])
        hospital_name = str(hospital["name"])
        candidates = [
            _action(hospital_id, "open_surge_general_beds", 10, 10, f"{hospital_name}: open 10 surge beds", {"ward_beds": 10}),
            _action(hospital_id, "open_surge_general_beds", 20, 20, f"{hospital_name}: open 20 surge beds", {"ward_beds": 20}),
            _action(hospital_id, "open_surge_general_beds", 30, 30, f"{hospital_name}: open 30 surge beds", {"ward_beds": 30}),
            _action(hospital_id, "open_surge_icu_beds", 4, 24, f"{hospital_name}: open 4 surge ICU beds", {"icu_beds": 4}),
            _action(hospital_id, "open_surge_icu_beds", 6, 36, f"{hospital_name}: open 6 surge ICU beds", {"icu_beds": 6}),
            _action(hospital_id, "add_temporary_nurses", 8, 16, f"{hospital_name}: add 8 temporary nurses", {"temporary_nurses": 8}),
            _action(hospital_id, "add_temporary_nurses", 12, 24, f"{hospital_name}: add 12 temporary nurses", {"temporary_nurses": 12}),
            _action(hospital_id, "add_temporary_doctors", 3, 18, f"{hospital_name}: add 3 temporary doctors", {"temporary_doctors": 3}),
            _action(hospital_id, "reduce_elective_admissions", 25, 10, f"{hospital_name}: reduce electives by 25%", {"elective_reduction_pct": 25}),
            _action(hospital_id, "reduce_elective_admissions", 40, 18, f"{hospital_name}: reduce electives by 40%", {"elective_reduction_pct": 40}),
            _action(hospital_id, "accelerate_discharge", 15, 14, f"{hospital_name}: accelerate discharge by 15%", {"discharge_acceleration_pct": 15}),
            _action(hospital_id, "accelerate_discharge", 20, 20, f"{hospital_name}: accelerate discharge by 20%", {"discharge_acceleration_pct": 20}),
            _action(hospital_id, "create_ed_overflow_spaces", 8, 12, f"{hospital_name}: create 8 ED overflow spaces", {"ed_overflow_spaces": 8}),
            _action(hospital_id, "transfer_patients", 8, 10, f"{hospital_name}: add 8 transfer slots", {"transfer_capacity": 8}),
        ]
        actions.extend(
            action
            for action in candidates
            if action.action_type in allowed and action.cost_points <= budget
        )
    return actions


def _action(
    hospital_id: str,
    action_type: str,
    amount: int,
    cost_points: float,
    description: str,
    effects: dict[str, Any],
) -> InterventionAction:
    return InterventionAction(
        action_id=f"{hospital_id}_{action_type}_{amount}",
        hospital_id=hospital_id,
        action_type=action_type,
        amount=amount,
        cost_points=float(cost_points),
        description=description,
        effects=effects,
    )

