from __future__ import annotations

import numpy as np
import pandas as pd

from winterflow.data.hospitals import get_default_hospitals
from winterflow.data.scenarios import ScenarioConfig, generate_winter_scenario
from winterflow.data.synthetic_history import generate_daily_hospital_demand


FORECAST_TARGETS = ("ed_arrivals", "admissions", "icu_admissions", "risk_score", "trolley_count")


def generate_synthetic_historical_demand(
    hospitals_df: pd.DataFrame | None = None,
    years: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate 2-3 years of deterministic synthetic daily hospital demand."""

    if years < 2:
        raise ValueError("At least 2 years of history are required for forecasting.")
    hospitals_df = get_default_hospitals(seed=seed) if hospitals_df is None else hospitals_df.copy()
    rng = np.random.default_rng(seed)
    scenario_cycle = [
        "baseline",
        "influenza_wave",
        "rsv_wave",
        "covid_wave",
        "combined_winter_surge",
        "severe_combined_surge",
    ]
    history_frames: list[pd.DataFrame] = []
    start_date = pd.Timestamp("2023-10-01")

    for year_index in range(years):
        scenario_name = scenario_cycle[int(rng.integers(0, len(scenario_cycle)))]
        season_start = start_date + pd.DateOffset(years=year_index)
        scenario_df = generate_winter_scenario(
            ScenarioConfig(
                scenario=scenario_name,
                start_date=season_start.date(),
                n_days=365,
                seed=seed + year_index,
                rsv_peak_day=58 + int(rng.integers(-10, 11)),
                covid_peak_day=82 + int(rng.integers(-14, 15)),
                flu_peak_day=108 + int(rng.integers(-14, 15)),
                virus_intensity_multiplier=float(rng.uniform(0.86, 1.20)),
                staff_absence_multiplier=float(rng.uniform(0.92, 1.14)),
            )
        )
        demand_df = generate_daily_hospital_demand(hospitals_df, scenario_df, seed=seed + 100 + year_index)
        scenario_features = scenario_df[
            [
                "date",
                "day",
                "scenario",
                "flu_multiplier",
                "rsv_multiplier",
                "covid_multiplier",
                "severity_shift",
                "admission_pressure_multiplier",
                "icu_pressure_multiplier",
            ]
        ]
        demand_df = demand_df.merge(scenario_features, on=["date", "day"], how="left")
        demand_df = demand_df.merge(
            hospitals_df[
                [
                    "hospital_id",
                    "region",
                    "ed_cubicles",
                    "general_beds",
                    "icu_beds",
                    "nurses",
                    "doctors",
                    "discharge_delay_rate",
                ]
            ],
            on="hospital_id",
            how="left",
        )
        history_frames.append(_derive_forecast_targets(demand_df, rng, year_index))

    history_df = pd.concat(history_frames, ignore_index=True)
    history_df = history_df.sort_values(["hospital_id", "date"]).reset_index(drop=True)
    history_df["history_day"] = (history_df["date"] - history_df["date"].min()).dt.days
    return history_df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric forecasting features from hospital-day history."""

    features_df = df.copy()
    features_df["date"] = pd.to_datetime(features_df["date"])
    features_df = features_df.sort_values(["hospital_id", "date"]).reset_index(drop=True)
    features_df["day_of_week"] = features_df["date"].dt.dayofweek
    features_df["is_weekend"] = features_df["day_of_week"].isin([5, 6]).astype(int)
    features_df["month"] = features_df["date"].dt.month
    features_df["day_of_year"] = features_df["date"].dt.dayofyear
    features_df["week_of_year"] = features_df["date"].dt.isocalendar().week.astype(int)
    features_df["sin_day_of_year"] = np.sin(2 * np.pi * features_df["day_of_year"] / 365.25)
    features_df["cos_day_of_year"] = np.cos(2 * np.pi * features_df["day_of_year"] / 365.25)

    for target in FORECAST_TARGETS:
        if target not in features_df:
            features_df[target] = 0.0
        group = features_df.groupby("hospital_id")[target]
        features_df[f"{target}_lag_1"] = group.shift(1)
        features_df[f"{target}_lag_7"] = group.shift(7)
        features_df[f"{target}_rolling_7"] = group.transform(
            lambda values: values.shift(1).rolling(7, min_periods=1).mean()
        )
        features_df[f"{target}_rolling_14"] = group.transform(
            lambda values: values.shift(1).rolling(14, min_periods=1).mean()
        )

    categorical_columns = [column for column in ["hospital_id", "hospital_type", "region", "scenario"] if column in features_df]
    features_df = pd.get_dummies(features_df, columns=categorical_columns, prefix=categorical_columns, dtype=int)
    numeric_columns = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_columns] = features_df[numeric_columns].fillna(0)
    features_df.attrs["feature_columns"] = get_feature_columns(features_df)
    return features_df


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    excluded = set(FORECAST_TARGETS) | {"synthetic_arrivals", "expected_admissions", "expected_icu_admissions"}
    excluded |= {"date", "hospital_name"}
    return [
        column
        for column in features_df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(features_df[column])
    ]


def _derive_forecast_targets(demand_df: pd.DataFrame, rng: np.random.Generator, year_index: int) -> pd.DataFrame:
    derived = demand_df.copy()
    derived["ed_arrivals"] = derived["synthetic_arrivals"].astype(int)
    derived["admissions"] = rng.poisson(np.clip(derived["expected_admissions"].astype(float), 0, None)).astype(int)
    derived["icu_admissions"] = rng.poisson(np.clip(derived["expected_icu_admissions"].astype(float), 0, None)).astype(int)
    capacity_pressure = derived["admissions"] / np.maximum(derived["general_beds"] / 9, 1)
    icu_pressure = derived["icu_admissions"] / np.maximum(derived["icu_beds"] / 5, 1)
    demand_pressure = np.maximum(derived["combined_demand_multiplier"].astype(float) - 1.0, 0)
    absence_pressure = derived["staff_absence_rate"].astype(float)
    trolley_lambda = np.clip(
        capacity_pressure * 0.60
        + demand_pressure * 8
        + absence_pressure * 18
        + derived["discharge_delay_rate"].astype(float) * 5,
        0,
        None,
    )
    derived["trolley_count"] = rng.poisson(trolley_lambda).astype(int)
    derived["risk_score"] = np.clip(
        18
        + 24 * demand_pressure
        + 16 * capacity_pressure
        + 20 * icu_pressure
        + 110 * absence_pressure
        + 2.6 * derived["trolley_count"]
        + year_index * 1.2,
        0,
        100,
    ).round(2)
    return derived
