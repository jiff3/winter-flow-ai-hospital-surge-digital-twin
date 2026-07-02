import pandas as pd

from winterflow.data.hospitals import get_default_hospitals
from winterflow.data.scenarios import ScenarioConfig, generate_winter_scenario
from winterflow.data.synthetic_history import generate_patient_arrivals


REQUIRED_PATIENT_COLUMNS = {
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
}


def test_patient_arrivals_have_required_columns() -> None:
    hospitals = get_default_hospitals(seed=1).head(2)
    scenario = generate_winter_scenario(ScenarioConfig(scenario="baseline", n_days=7, seed=1))

    patients = generate_patient_arrivals(hospitals, scenario, seed=1)

    assert REQUIRED_PATIENT_COLUMNS.issubset(patients.columns)


def test_arrival_times_fall_within_simulation_horizon() -> None:
    hospitals = get_default_hospitals(seed=1).head(2)
    scenario = generate_winter_scenario(ScenarioConfig(scenario="baseline", n_days=7, seed=1))

    patients = generate_patient_arrivals(hospitals, scenario, seed=1)
    start = pd.Timestamp(scenario["date"].min())
    end = pd.Timestamp(scenario["date"].max()) + pd.Timedelta(days=1)

    assert (patients["arrival_time"] >= start).all()
    assert (patients["arrival_time"] < end).all()


def test_needs_icu_implies_needs_admission() -> None:
    hospitals = get_default_hospitals(seed=1).head(3)
    scenario = generate_winter_scenario(ScenarioConfig(scenario="severe_combined_surge", n_days=10, seed=1))

    patients = generate_patient_arrivals(hospitals, scenario, seed=3)

    assert patients.loc[patients["needs_icu"], "needs_admission"].all()


def test_admission_rate_increases_under_severe_surge_compared_with_baseline() -> None:
    hospitals = get_default_hospitals(seed=4).head(4)
    baseline = generate_winter_scenario(ScenarioConfig(scenario="baseline", n_days=30, seed=4))
    severe = generate_winter_scenario(ScenarioConfig(scenario="severe_combined_surge", n_days=30, seed=4))

    baseline_patients = generate_patient_arrivals(hospitals, baseline, seed=4)
    severe_patients = generate_patient_arrivals(hospitals, severe, seed=4)

    assert severe_patients["needs_admission"].mean() > baseline_patients["needs_admission"].mean()


def test_same_seed_gives_same_patient_arrivals() -> None:
    hospitals = get_default_hospitals(seed=2).head(2)
    scenario = generate_winter_scenario(ScenarioConfig(scenario="rsv_wave", n_days=7, seed=2))

    first = generate_patient_arrivals(hospitals, scenario, seed=99)
    second = generate_patient_arrivals(hospitals, scenario, seed=99)

    pd.testing.assert_frame_equal(first, second)

