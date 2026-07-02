import pandas as pd

from winterflow.data.scenarios import (
    ScenarioConfig,
    apply_policy_modifiers,
    generate_winter_scenario,
    list_available_scenarios,
    plot_scenario_curves,
)


def test_scenario_has_expected_number_of_days() -> None:
    config = ScenarioConfig(scenario="influenza_wave", n_days=60, seed=10)

    df = generate_winter_scenario(config)

    assert len(df) == 60


def test_multipliers_are_non_negative() -> None:
    df = generate_winter_scenario(ScenarioConfig(scenario="combined_winter_surge", seed=10))
    multiplier_columns = [
        "flu_multiplier",
        "rsv_multiplier",
        "covid_multiplier",
        "combined_demand_multiplier",
        "admission_pressure_multiplier",
        "icu_pressure_multiplier",
    ]

    assert (df[multiplier_columns] >= 0).all().all()
    assert (df["staff_absence_rate"] >= 0).all()


def test_severe_combined_surge_has_higher_mean_demand_than_baseline() -> None:
    baseline = generate_winter_scenario(ScenarioConfig(scenario="baseline", seed=10))
    severe = generate_winter_scenario(ScenarioConfig(scenario="severe_combined_surge", seed=10))

    assert severe["combined_demand_multiplier"].mean() > baseline["combined_demand_multiplier"].mean()


def test_peak_shift_moves_peak_day_in_expected_direction() -> None:
    base_config = ScenarioConfig(scenario="influenza_wave", seed=10, noise_std=0)
    shifted_config = apply_policy_modifiers(base_config, {"flu_peak_shift_days": 10})

    base_df = generate_winter_scenario(base_config)
    shifted_df = generate_winter_scenario(shifted_config)

    assert shifted_df.loc[shifted_df["flu_multiplier"].idxmax(), "day"] > base_df.loc[
        base_df["flu_multiplier"].idxmax(), "day"
    ]


def test_same_seed_gives_same_output() -> None:
    config = ScenarioConfig(scenario="rsv_wave", seed=123)

    first = generate_winter_scenario(config)
    second = generate_winter_scenario(config)

    pd.testing.assert_frame_equal(first, second)


def test_available_scenarios_and_plot_are_available() -> None:
    assert "severe_combined_surge" in list_available_scenarios()

    df = generate_winter_scenario(ScenarioConfig(n_days=14))
    fig = plot_scenario_curves(df)

    assert len(fig.data) == 4

