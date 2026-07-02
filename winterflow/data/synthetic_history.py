from __future__ import annotations

import numpy as np
import pandas as pd


AGE_GROUPS = ("0-4", "5-17", "18-49", "50-64", "65-79", "80+")
ACUITY_LEVELS = ("low", "moderate", "high", "critical")
VIRUS_TYPES = ("flu", "rsv", "covid", "other_respiratory", "non_virus")
QUICK_MODE_SCALE = 0.23

_DAY_OF_WEEK_EFFECT = {
    0: 1.08,
    1: 1.03,
    2: 1.00,
    3: 0.98,
    4: 1.02,
    5: 0.92,
    6: 0.88,
}

_AGE_PROBABILITIES = {
    "flu": np.array([0.08, 0.12, 0.28, 0.18, 0.20, 0.14]),
    "rsv": np.array([0.28, 0.11, 0.20, 0.12, 0.17, 0.12]),
    "covid": np.array([0.04, 0.08, 0.30, 0.22, 0.22, 0.14]),
    "other_respiratory": np.array([0.10, 0.14, 0.32, 0.18, 0.16, 0.10]),
    "non_virus": np.array([0.06, 0.10, 0.38, 0.20, 0.16, 0.10]),
}

_VIRUS_SEVERITY = {
    "flu": 1.12,
    "rsv": 1.18,
    "covid": 1.22,
    "other_respiratory": 1.04,
    "non_virus": 0.94,
}

_ACUITY_ADMISSION_MULTIPLIER = {
    "low": 0.22,
    "moderate": 0.70,
    "high": 1.65,
    "critical": 3.10,
}

_ACUITY_ICU_MULTIPLIER = {
    "low": 0.02,
    "moderate": 0.12,
    "high": 0.72,
    "critical": 2.40,
}

_AGE_ADMISSION_MULTIPLIER = {
    "0-4": 0.92,
    "5-17": 0.54,
    "18-49": 0.72,
    "50-64": 1.02,
    "65-79": 1.46,
    "80+": 1.82,
}


def generate_daily_hospital_demand(
    hospitals_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily ED demand by hospital and scenario day."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for hospital in hospitals_df.to_dict("records"):
        for scenario_day in scenario_df.to_dict("records"):
            date_value = pd.Timestamp(scenario_day["date"])
            day_of_week_effect = _DAY_OF_WEEK_EFFECT[int(date_value.dayofweek)]
            expected_arrivals = (
                float(hospital["baseline_ed_arrivals_per_day"])
                * float(scenario_day["combined_demand_multiplier"])
                * day_of_week_effect
            )
            admission_rate = _clip_probability(
                float(hospital["baseline_admission_rate"])
                * float(scenario_day["admission_pressure_multiplier"])
                * (1.0 + float(scenario_day["severity_shift"]))
            )
            icu_rate = _clip_probability(
                float(hospital["baseline_icu_admission_rate"])
                * float(scenario_day["icu_pressure_multiplier"])
                * (1.0 + 1.4 * float(scenario_day["severity_shift"])),
                upper=0.35,
            )
            synthetic_arrivals = int(rng.poisson(expected_arrivals))

            rows.append(
                {
                    "date": date_value,
                    "day": int(scenario_day["day"]),
                    "hospital_id": hospital["hospital_id"],
                    "hospital_name": hospital["name"],
                    "hospital_type": hospital["type"],
                    "day_of_week_effect": round(day_of_week_effect, 3),
                    "combined_demand_multiplier": float(scenario_day["combined_demand_multiplier"]),
                    "expected_arrivals": round(expected_arrivals, 2),
                    "synthetic_arrivals": synthetic_arrivals,
                    "expected_admissions": round(expected_arrivals * admission_rate, 2),
                    "expected_icu_admissions": round(expected_arrivals * icu_rate, 2),
                    "admission_rate": round(admission_rate, 4),
                    "icu_rate": round(icu_rate, 4),
                    "staff_absence_rate": float(scenario_day["staff_absence_rate"]),
                }
            )

    return pd.DataFrame(rows)


def generate_patient_arrivals(
    hospitals_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    seed: int = 42,
    quick_mode: bool = True,
) -> pd.DataFrame:
    """Generate deterministic synthetic patient-level ED arrivals."""

    rng = np.random.default_rng(seed)
    scale = QUICK_MODE_SCALE if quick_mode else 1.0
    rows: list[dict[str, object]] = []

    for hospital in hospitals_df.to_dict("records"):
        for scenario_day in scenario_df.to_dict("records"):
            date_value = pd.Timestamp(scenario_day["date"])
            arrival_lambda = (
                float(hospital["baseline_ed_arrivals_per_day"])
                * float(scenario_day["combined_demand_multiplier"])
                * _DAY_OF_WEEK_EFFECT[int(date_value.dayofweek)]
                * scale
            )
            arrivals = int(rng.poisson(arrival_lambda))

            for _ in range(arrivals):
                virus_type = _sample_virus_type(rng, scenario_day)
                age_group = _sample_age_group(rng, virus_type)
                acuity = _sample_acuity(rng, scenario_day, virus_type, age_group)
                needs_admission, needs_icu = _sample_disposition(
                    rng,
                    hospital,
                    scenario_day,
                    virus_type,
                    age_group,
                    acuity,
                )
                arrival_time = date_value + pd.to_timedelta(_sample_arrival_hour(rng), unit="h")
                ed_service_time_hours = _sample_ed_service_time(rng, acuity, virus_type)
                ward_los_days = _sample_ward_los_days(rng, needs_admission, needs_icu, virus_type, age_group)
                icu_los_days = _sample_icu_los_days(rng, needs_icu, virus_type, age_group)
                discharge_delay_days = _sample_discharge_delay_days(
                    rng,
                    hospital,
                    scenario_day,
                    needs_admission,
                )

                rows.append(
                    {
                        "patient_id": "",
                        "hospital_id": hospital["hospital_id"],
                        "arrival_time": arrival_time,
                        "arrival_day": int(scenario_day["day"]),
                        "virus_type": virus_type,
                        "acuity": acuity,
                        "age_group": age_group,
                        "needs_admission": bool(needs_admission),
                        "needs_icu": bool(needs_icu),
                        "ed_service_time_hours": round(ed_service_time_hours, 3),
                        "ward_los_days": round(ward_los_days, 3),
                        "icu_los_days": round(icu_los_days, 3),
                        "discharge_delay_days": round(discharge_delay_days, 3),
                    }
                )

    arrivals_df = pd.DataFrame(rows)
    if arrivals_df.empty:
        return pd.DataFrame(columns=_PATIENT_COLUMNS)

    arrivals_df = arrivals_df.sort_values(["arrival_time", "hospital_id"]).reset_index(drop=True)
    arrivals_df["patient_id"] = [f"P{index:08d}" for index in range(1, len(arrivals_df) + 1)]
    return arrivals_df[_PATIENT_COLUMNS]


_PATIENT_COLUMNS = [
    "patient_id",
    "hospital_id",
    "arrival_time",
    "arrival_day",
    "virus_type",
    "acuity",
    "age_group",
    "needs_admission",
    "needs_icu",
    "ed_service_time_hours",
    "ward_los_days",
    "icu_los_days",
    "discharge_delay_days",
]


def _clip_probability(value: float, lower: float = 0.0, upper: float = 0.95) -> float:
    return float(np.clip(value, lower, upper))


def _sample_virus_type(rng: np.random.Generator, scenario_day: dict[str, object]) -> str:
    flu_excess = max(float(scenario_day["flu_multiplier"]) - 1.0, 0.0)
    rsv_excess = max(float(scenario_day["rsv_multiplier"]) - 1.0, 0.0)
    covid_excess = max(float(scenario_day["covid_multiplier"]) - 1.0, 0.0)
    weights = np.array(
        [
            0.06 + 0.38 * flu_excess,
            0.05 + 0.34 * rsv_excess,
            0.05 + 0.35 * covid_excess,
            0.18,
            0.66,
        ]
    )
    probabilities = weights / weights.sum()
    return str(rng.choice(VIRUS_TYPES, p=probabilities))


def _sample_age_group(rng: np.random.Generator, virus_type: str) -> str:
    return str(rng.choice(AGE_GROUPS, p=_AGE_PROBABILITIES[virus_type]))


def _sample_acuity(
    rng: np.random.Generator,
    scenario_day: dict[str, object],
    virus_type: str,
    age_group: str,
) -> str:
    severity_shift = float(scenario_day["severity_shift"])
    high_risk_age = age_group in {"0-4", "65-79", "80+"}
    virus_severity = _VIRUS_SEVERITY[virus_type]
    pressure = severity_shift + max(float(scenario_day["combined_demand_multiplier"]) - 1.0, 0.0)
    high_adjustment = min(0.11, 0.035 * virus_severity + 0.08 * pressure + (0.03 if high_risk_age else 0.0))
    critical_adjustment = min(0.045, 0.012 * virus_severity + 0.035 * pressure + (0.01 if age_group == "80+" else 0.0))
    probabilities = np.array(
        [
            0.46 - high_adjustment * 0.55 - critical_adjustment,
            0.36 - high_adjustment * 0.25,
            0.15 + high_adjustment,
            0.03 + critical_adjustment,
        ]
    )
    probabilities = np.clip(probabilities, 0.02, None)
    probabilities = probabilities / probabilities.sum()
    return str(rng.choice(ACUITY_LEVELS, p=probabilities))


def _sample_disposition(
    rng: np.random.Generator,
    hospital: dict[str, object],
    scenario_day: dict[str, object],
    virus_type: str,
    age_group: str,
    acuity: str,
) -> tuple[bool, bool]:
    admission_probability = _clip_probability(
        float(hospital["baseline_admission_rate"])
        * float(scenario_day["admission_pressure_multiplier"])
        * _VIRUS_SEVERITY[virus_type]
        * _AGE_ADMISSION_MULTIPLIER[age_group]
        * _ACUITY_ADMISSION_MULTIPLIER[acuity],
        upper=0.92,
    )
    icu_probability = _clip_probability(
        float(hospital["baseline_icu_admission_rate"])
        * float(scenario_day["icu_pressure_multiplier"])
        * _VIRUS_SEVERITY[virus_type]
        * _AGE_ADMISSION_MULTIPLIER[age_group]
        * _ACUITY_ICU_MULTIPLIER[acuity],
        upper=0.45,
    )
    needs_icu = bool(rng.random() < icu_probability)
    needs_admission = bool(needs_icu or rng.random() < admission_probability)
    return needs_admission, needs_icu


def _sample_arrival_hour(rng: np.random.Generator) -> float:
    if rng.random() < 0.58:
        hour = rng.normal(13.5, 4.2)
    else:
        hour = rng.normal(20.0, 3.1)
    return float(np.clip(hour, 0, 23.999))


def _sample_ed_service_time(rng: np.random.Generator, acuity: str, virus_type: str) -> float:
    base_by_acuity = {
        "low": (1.45, 0.55),
        "moderate": (2.85, 0.85),
        "high": (4.80, 1.20),
        "critical": (6.40, 1.55),
    }
    mean, sigma = base_by_acuity[acuity]
    service_time = rng.gamma(shape=(mean / sigma) ** 2, scale=(sigma**2) / mean)
    return float(np.clip(service_time * (0.96 + 0.06 * _VIRUS_SEVERITY[virus_type]), 0.35, 18.0))


def _sample_ward_los_days(
    rng: np.random.Generator,
    needs_admission: bool,
    needs_icu: bool,
    virus_type: str,
    age_group: str,
) -> float:
    if not needs_admission:
        return 0.0
    age_factor = 1.0 + (0.25 if age_group in {"65-79", "80+"} else 0.0) + (0.15 if age_group == "0-4" else 0.0)
    mean_los = (3.4 if not needs_icu else 5.1) * _VIRUS_SEVERITY[virus_type] * age_factor
    return float(np.clip(rng.gamma(shape=2.4, scale=mean_los / 2.4), 0.35, 28.0))


def _sample_icu_los_days(
    rng: np.random.Generator,
    needs_icu: bool,
    virus_type: str,
    age_group: str,
) -> float:
    if not needs_icu:
        return 0.0
    age_factor = 1.0 + (0.22 if age_group in {"65-79", "80+"} else 0.0)
    mean_los = 4.2 * _VIRUS_SEVERITY[virus_type] * age_factor
    return float(np.clip(rng.gamma(shape=2.0, scale=mean_los / 2.0), 0.5, 24.0))


def _sample_discharge_delay_days(
    rng: np.random.Generator,
    hospital: dict[str, object],
    scenario_day: dict[str, object],
    needs_admission: bool,
) -> float:
    if not needs_admission:
        return 0.0
    delay_probability = _clip_probability(
        float(hospital["discharge_delay_rate"])
        * (1.0 + 0.55 * max(float(scenario_day["combined_demand_multiplier"]) - 1.0, 0.0)),
        upper=0.55,
    )
    if rng.random() >= delay_probability:
        return 0.0
    average_delay = float(hospital["average_discharge_delay_days"])
    return float(np.clip(rng.gamma(shape=1.8, scale=average_delay / 1.8), 0.1, 10.0))
