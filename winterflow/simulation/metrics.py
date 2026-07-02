from __future__ import annotations

import numpy as np


RISK_LABELS = ("Green", "Amber", "Red")


def calculate_staff_stress(
    ed_crowding_pct: float,
    ward_occupancy_pct: float,
    icu_occupancy_pct: float,
    trolley_count: int,
    staff_absence_rate: float,
    ed_cubicles: int,
) -> float:
    """Estimate a synthetic staff stress index from load and absence pressure."""

    trolley_pressure = trolley_count / max(ed_cubicles, 1)
    load_index = (
        0.30 * (ed_crowding_pct / 100)
        + 0.30 * (ward_occupancy_pct / 100)
        + 0.28 * (icu_occupancy_pct / 100)
        + 0.12 * trolley_pressure
    )
    available_staff_factor = max(0.45, 1.0 - staff_absence_rate)
    return float(np.clip(load_index / available_staff_factor, 0, 3.0))


def calculate_risk_score(
    ed_crowding_pct: float,
    ward_occupancy_pct: float,
    icu_occupancy_pct: float,
    trolley_count: int,
    mean_staff_stress: float,
    mean_ed_wait_hours: float,
) -> float:
    """Calculate a 0-100 operational risk score."""

    ward_component = _scaled(ward_occupancy_pct, 78, 105)
    icu_component = _scaled(icu_occupancy_pct, 70, 100)
    ed_component = _scaled(ed_crowding_pct, 85, 150)
    trolley_component = _scaled(trolley_count, 0, 35)
    stress_component = _scaled(mean_staff_stress, 0.75, 1.45)
    wait_component = _scaled(mean_ed_wait_hours, 1.0, 8.0)
    score = (
        0.22 * ward_component
        + 0.22 * icu_component
        + 0.20 * ed_component
        + 0.16 * trolley_component
        + 0.12 * stress_component
        + 0.08 * wait_component
    )
    return float(np.clip(score, 0, 100))


def risk_label(score: float) -> str:
    """Map a risk score to a command-center label."""

    if score >= 70:
        return "Red"
    if score >= 40:
        return "Amber"
    return "Green"


def _scaled(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip(((value - low) / (high - low)) * 100, 0, 100))

