import pandas as pd

from winterflow.data.hospitals import get_default_hospitals
from winterflow.data.scenarios import ScenarioConfig, generate_winter_scenario
from winterflow.data.synthetic_history import generate_patient_arrivals
from winterflow.simulation.engine import run_simulation
from winterflow.simulation.entities import SimulationConfig


def _small_simulation_inputs():
    hospitals = get_default_hospitals(seed=11).head(2).copy()
    hospitals.loc[:, "general_beds"] = [12, 14]
    hospitals.loc[:, "icu_beds"] = [3, 3]
    hospitals.loc[:, "ed_cubicles"] = [8, 8]
    hospitals.loc[:, "nurses"] = [80, 90]
    scenario = generate_winter_scenario(ScenarioConfig(scenario="combined_winter_surge", n_days=5, seed=11))
    patients = generate_patient_arrivals(hospitals, scenario, seed=11, quick_mode=True).head(80)
    config = SimulationConfig(initial_ward_occupancy_pct=0.25, initial_icu_occupancy_pct=0.0)
    return hospitals, scenario, patients, config


def test_simulation_returns_all_four_outputs() -> None:
    hospitals, scenario, patients, config = _small_simulation_inputs()

    outputs = run_simulation(hospitals, patients, scenario, config=config)

    assert len(outputs) == 4
    assert all(isinstance(output, pd.DataFrame) for output in outputs)


def test_no_negative_occupancy() -> None:
    hospitals, scenario, patients, config = _small_simulation_inputs()

    _, hourly_metrics, _, _ = run_simulation(hospitals, patients, scenario, config=config)

    occupancy_columns = ["ed_occupied", "ward_occupied", "icu_occupied", "trolley_count"]
    assert (hourly_metrics[occupancy_columns] >= 0).all().all()


def test_risk_scores_and_labels_are_valid() -> None:
    hospitals, scenario, patients, config = _small_simulation_inputs()

    _, _, daily_metrics, _ = run_simulation(hospitals, patients, scenario, config=config)

    assert daily_metrics["risk_score"].between(0, 100).all()
    assert set(daily_metrics["risk_label"]).issubset({"Green", "Amber", "Red"})


def test_reducing_ward_beds_increases_trolley_count_in_controlled_comparison() -> None:
    hospitals = get_default_hospitals(seed=5).head(1).copy()
    hospitals.loc[:, "general_beds"] = [8]
    hospitals.loc[:, "icu_beds"] = [2]
    hospitals.loc[:, "ed_cubicles"] = [20]
    hospitals.loc[:, "nurses"] = [50]
    scenario = generate_winter_scenario(ScenarioConfig(scenario="baseline", n_days=2, seed=5))
    patients = pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(16)],
            "hospital_id": [hospitals.iloc[0]["hospital_id"]] * 16,
            "arrival_time": [pd.Timestamp("2026-11-01 08:00") + pd.Timedelta(minutes=i * 2) for i in range(16)],
            "arrival_day": [0] * 16,
            "virus_type": ["non_virus"] * 16,
            "acuity": ["moderate"] * 16,
            "age_group": ["65-79"] * 16,
            "needs_admission": [True] * 16,
            "needs_icu": [False] * 16,
            "ed_service_time_hours": [0.2] * 16,
            "ward_los_days": [1.5] * 16,
            "icu_los_days": [0.0] * 16,
            "discharge_delay_days": [0.0] * 16,
        }
    )
    config = SimulationConfig(initial_ward_occupancy_pct=0.0, initial_icu_occupancy_pct=0.0)

    _, _, daily_roomy, _ = run_simulation(
        hospitals,
        patients,
        scenario,
        config=config,
        resource_overrides={hospitals.iloc[0]["hospital_id"]: {"ward_beds": 8}},
    )
    _, _, daily_constrained, _ = run_simulation(
        hospitals,
        patients,
        scenario,
        config=config,
        resource_overrides={hospitals.iloc[0]["hospital_id"]: {"ward_beds": 2}},
    )

    assert daily_constrained["max_trolley_count"].max() > daily_roomy["max_trolley_count"].max()


def test_patients_not_admitted_do_not_occupy_ward_or_icu_beds() -> None:
    hospitals = get_default_hospitals(seed=6).head(1).copy()
    scenario = generate_winter_scenario(ScenarioConfig(scenario="baseline", n_days=1, seed=6))
    patients = pd.DataFrame(
        {
            "patient_id": ["P001", "P002"],
            "hospital_id": [hospitals.iloc[0]["hospital_id"], hospitals.iloc[0]["hospital_id"]],
            "arrival_time": [pd.Timestamp("2026-11-01 08:00"), pd.Timestamp("2026-11-01 08:10")],
            "arrival_day": [0, 0],
            "virus_type": ["non_virus", "flu"],
            "acuity": ["low", "moderate"],
            "age_group": ["18-49", "50-64"],
            "needs_admission": [False, False],
            "needs_icu": [False, False],
            "ed_service_time_hours": [0.2, 0.2],
            "ward_los_days": [0.0, 0.0],
            "icu_los_days": [0.0, 0.0],
            "discharge_delay_days": [0.0, 0.0],
        }
    )
    config = SimulationConfig(initial_ward_occupancy_pct=0.0, initial_icu_occupancy_pct=0.0)

    patient_results, _, _, _ = run_simulation(hospitals, patients, scenario, config=config)

    assert (patient_results["bed_type"] == "none").all()
    assert (patient_results["time_to_bed_hours"] == 0).all()
