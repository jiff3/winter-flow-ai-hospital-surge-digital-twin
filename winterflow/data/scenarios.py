from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


SUPPORTED_SCENARIOS = (
    "baseline",
    "influenza_wave",
    "rsv_wave",
    "covid_wave",
    "combined_winter_surge",
    "severe_combined_surge",
)

_SCENARIO_AMPLITUDES = {
    "baseline": {"flu": 0.05, "rsv": 0.04, "covid": 0.04, "severity": 0.02},
    "influenza_wave": {"flu": 0.58, "rsv": 0.08, "covid": 0.08, "severity": 0.08},
    "rsv_wave": {"flu": 0.08, "rsv": 0.66, "covid": 0.08, "severity": 0.10},
    "covid_wave": {"flu": 0.08, "rsv": 0.08, "covid": 0.62, "severity": 0.11},
    "combined_winter_surge": {"flu": 0.55, "rsv": 0.52, "covid": 0.46, "severity": 0.14},
    "severe_combined_surge": {"flu": 0.98, "rsv": 0.88, "covid": 0.78, "severity": 0.24},
}

_SCENARIO_WIDTHS = {
    "baseline": {"flu": 21.0, "rsv": 18.0, "covid": 24.0},
    "influenza_wave": {"flu": 18.0, "rsv": 20.0, "covid": 24.0},
    "rsv_wave": {"flu": 20.0, "rsv": 15.0, "covid": 24.0},
    "covid_wave": {"flu": 20.0, "rsv": 18.0, "covid": 18.0},
    "combined_winter_surge": {"flu": 18.0, "rsv": 15.0, "covid": 20.0},
    "severe_combined_surge": {"flu": 20.0, "rsv": 16.0, "covid": 21.0},
}


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a deterministic synthetic winter virus scenario."""

    scenario: str = "combined_winter_surge"
    start_date: date | str = date(2026, 11, 1)
    n_days: int = 120
    seed: int = 42
    flu_peak_day: int = 68
    rsv_peak_day: int = 38
    covid_peak_day: int = 55
    rsv_peak_shift_days: int = 0
    flu_peak_shift_days: int = 0
    covid_peak_shift_days: int = 0
    virus_intensity_multiplier: float = 1.0
    staff_absence_multiplier: float = 1.0
    baseline_arrival_multiplier: float = 1.0
    admission_rate_multiplier: float = 1.0
    icu_rate_multiplier: float = 1.0
    noise_std: float = 0.015


def generate_winter_scenario(config: ScenarioConfig) -> pd.DataFrame:
    """Generate synthetic winter respiratory pressure curves."""

    scenario = _normalise_scenario_name(config.scenario)
    if config.n_days <= 0:
        raise ValueError("n_days must be positive.")

    rng = np.random.default_rng(config.seed)
    day = np.arange(config.n_days)
    dates = pd.date_range(pd.to_datetime(config.start_date), periods=config.n_days, freq="D")
    amplitudes = _SCENARIO_AMPLITUDES[scenario]
    widths = _SCENARIO_WIDTHS[scenario]

    flu_curve = _virus_multiplier(
        day,
        peak_day=config.flu_peak_day + config.flu_peak_shift_days,
        width=widths["flu"],
        amplitude=amplitudes["flu"],
        intensity=config.virus_intensity_multiplier,
        rng=rng,
        noise_std=config.noise_std,
    )
    rsv_curve = _virus_multiplier(
        day,
        peak_day=config.rsv_peak_day + config.rsv_peak_shift_days,
        width=widths["rsv"],
        amplitude=amplitudes["rsv"],
        intensity=config.virus_intensity_multiplier,
        rng=rng,
        noise_std=config.noise_std,
    )
    covid_curve = _virus_multiplier(
        day,
        peak_day=config.covid_peak_day + config.covid_peak_shift_days,
        width=widths["covid"],
        amplitude=amplitudes["covid"],
        intensity=config.virus_intensity_multiplier,
        rng=rng,
        noise_std=config.noise_std,
    )

    combined_demand_multiplier = np.clip(
        config.baseline_arrival_multiplier
        * ((0.42 * flu_curve) + (0.28 * rsv_curve) + (0.30 * covid_curve)),
        0,
        None,
    )
    excess_demand = np.clip(combined_demand_multiplier - 1.0, 0, None)
    virus_excess = (
        0.32 * np.clip(flu_curve - 1.0, 0, None)
        + 0.32 * np.clip(rsv_curve - 1.0, 0, None)
        + 0.36 * np.clip(covid_curve - 1.0, 0, None)
    )

    severity_shift = np.clip(amplitudes["severity"] * virus_excess, 0, None)
    staff_absence_rate = np.clip(
        (0.045 + 0.052 * excess_demand + rng.normal(0, config.noise_std / 4, size=config.n_days))
        * config.staff_absence_multiplier,
        0,
        0.28,
    )
    admission_pressure_multiplier = np.clip(
        (1.0 + 0.44 * excess_demand + severity_shift) * config.admission_rate_multiplier,
        0,
        None,
    )
    icu_pressure_multiplier = np.clip(
        (1.0 + 0.62 * excess_demand + 1.45 * severity_shift) * config.icu_rate_multiplier,
        0,
        None,
    )

    return pd.DataFrame(
        {
            "date": dates,
            "day": day,
            "scenario": scenario,
            "flu_multiplier": np.round(flu_curve, 4),
            "rsv_multiplier": np.round(rsv_curve, 4),
            "covid_multiplier": np.round(covid_curve, 4),
            "combined_demand_multiplier": np.round(combined_demand_multiplier, 4),
            "severity_shift": np.round(severity_shift, 4),
            "staff_absence_rate": np.round(staff_absence_rate, 4),
            "admission_pressure_multiplier": np.round(admission_pressure_multiplier, 4),
            "icu_pressure_multiplier": np.round(icu_pressure_multiplier, 4),
        }
    )


def list_available_scenarios() -> list[str]:
    """Return supported synthetic scenario names."""

    return list(SUPPORTED_SCENARIOS)


def apply_policy_modifiers(config: ScenarioConfig, modifiers: dict[str, Any]) -> ScenarioConfig:
    """Return a new scenario config with selected policy modifiers applied."""

    valid_modifier_names = {
        "rsv_peak_shift_days",
        "flu_peak_shift_days",
        "covid_peak_shift_days",
        "virus_intensity_multiplier",
        "staff_absence_multiplier",
        "baseline_arrival_multiplier",
        "admission_rate_multiplier",
        "icu_rate_multiplier",
    }
    unknown_modifiers = set(modifiers) - valid_modifier_names
    if unknown_modifiers:
        names = ", ".join(sorted(unknown_modifiers))
        raise ValueError(f"Unsupported policy modifier(s): {names}")

    return replace(config, **modifiers)


def plot_scenario_curves(scenario_df: pd.DataFrame) -> go.Figure:
    """Build a Plotly line chart for virus and combined demand multipliers."""

    fig = go.Figure()
    curve_columns = {
        "flu_multiplier": "Flu",
        "rsv_multiplier": "RSV",
        "covid_multiplier": "COVID",
        "combined_demand_multiplier": "Combined demand",
    }
    for column, label in curve_columns.items():
        fig.add_trace(
            go.Scatter(
                x=scenario_df["date"],
                y=scenario_df[column],
                mode="lines",
                name=label,
            )
        )

    fig.update_layout(
        title="Winter Virus Waves And Combined Demand Multiplier",
        xaxis_title="Date",
        yaxis_title="Multiplier",
        hovermode="x unified",
        legend_title_text="Curve",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def _normalise_scenario_name(scenario: str) -> str:
    normalized = scenario.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized not in SUPPORTED_SCENARIOS:
        supported = ", ".join(SUPPORTED_SCENARIOS)
        raise ValueError(f"Unsupported scenario '{scenario}'. Choose one of: {supported}.")
    return normalized


def _virus_multiplier(
    day: np.ndarray,
    peak_day: int,
    width: float,
    amplitude: float,
    intensity: float,
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    seasonal_wave = np.exp(-0.5 * ((day - peak_day) / width) ** 2)
    noise = rng.normal(0, noise_std, size=len(day))
    return np.clip(1.0 + (amplitude * intensity * seasonal_wave) + noise, 0, None)

