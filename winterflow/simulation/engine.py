from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import simpy

from winterflow.simulation.entities import (
    HospitalResourcePlan,
    HospitalResources,
    SimulationConfig,
    build_resource_plan,
    build_simpy_resources,
)
from winterflow.simulation.metrics import calculate_risk_score, calculate_staff_stress, risk_label
from winterflow.simulation.validation import validate_simulation_inputs


class _SimulationState:
    def __init__(self) -> None:
        self.patient_results: list[dict[str, object]] = []
        self.hourly_metrics: list[dict[str, object]] = []
        self.event_log: list[dict[str, object]] = []
        self.trolley_counts: dict[str, int] = defaultdict(int)


def run_simulation(
    hospitals_df: pd.DataFrame,
    patient_arrivals_df: pd.DataFrame,
    scenario_df: pd.DataFrame | None = None,
    config: SimulationConfig | None = None,
    resource_overrides: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the synthetic hospital patient-flow simulation."""

    validate_simulation_inputs(hospitals_df, patient_arrivals_df)
    config = config or SimulationConfig()
    resource_overrides = resource_overrides or {}

    if patient_arrivals_df.empty:
        empty_patients = _empty_patient_results()
        empty_hourly = _empty_hourly_metrics()
        empty_daily = _empty_daily_metrics()
        empty_events = _empty_event_log()
        return empty_patients, empty_hourly, empty_daily, empty_events

    arrivals = patient_arrivals_df.copy()
    arrivals["arrival_time"] = pd.to_datetime(arrivals["arrival_time"])
    simulation_start = arrivals["arrival_time"].min().floor("D")
    arrivals["arrival_hour"] = (arrivals["arrival_time"] - simulation_start).dt.total_seconds() / 3600
    arrivals = arrivals.sort_values("arrival_hour").reset_index(drop=True)

    env = simpy.Environment()
    state = _SimulationState()
    resources = _build_resources(env, hospitals_df, resource_overrides)
    staff_absence_by_day = _staff_absence_lookup(scenario_df)

    for hospital_id, hospital_resources in resources.items():
        _preload_beds(env, hospital_id, hospital_resources, config)

    monitor_process = env.process(
        _monitor_resources(
            env,
            resources,
            state,
            staff_absence_by_day,
            config,
        )
    )
    patient_processes = [
        env.process(_patient_process(env, row, resources[str(row["hospital_id"])], state, simulation_start))
        for row in arrivals.to_dict("records")
    ]
    env.run(until=env.all_of(patient_processes))
    monitor_process.interrupt("simulation complete")

    patient_results_df = pd.DataFrame(state.patient_results)
    hourly_metrics_df = pd.DataFrame(state.hourly_metrics)
    event_log_df = pd.DataFrame(state.event_log)
    daily_metrics_df = _build_daily_metrics(patient_results_df, hourly_metrics_df, simulation_start)

    return patient_results_df, hourly_metrics_df, daily_metrics_df, event_log_df


run_patient_flow_simulation = run_simulation


def _build_resources(
    env: simpy.Environment,
    hospitals_df: pd.DataFrame,
    overrides: dict[str, dict[str, int]],
) -> dict[str, HospitalResources]:
    resources: dict[str, HospitalResources] = {}
    for hospital in hospitals_df.to_dict("records"):
        plan = build_resource_plan(hospital)
        hospital_id = plan.hospital_id
        if hospital_id in overrides:
            plan = _apply_resource_overrides(plan, overrides[hospital_id])
        resources[hospital_id] = build_simpy_resources(env, plan)
    return resources


def _apply_resource_overrides(
    plan: HospitalResourcePlan,
    overrides: dict[str, int],
) -> HospitalResourcePlan:
    allowed = set(HospitalResourcePlan.__dataclass_fields__) - {"hospital_id"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unsupported resource override(s): {sorted(unknown)}")
    normalized = {key: max(1, int(value)) for key, value in overrides.items()}
    return replace(plan, **normalized)


def _staff_absence_lookup(scenario_df: pd.DataFrame | None) -> dict[int, float]:
    if scenario_df is None or scenario_df.empty or "staff_absence_rate" not in scenario_df:
        return {}
    return {
        int(row["day"]): float(row["staff_absence_rate"])
        for row in scenario_df[["day", "staff_absence_rate"]].to_dict("records")
    }


def _preload_beds(
    env: simpy.Environment,
    hospital_id: str,
    resources: HospitalResources,
    config: SimulationConfig,
) -> None:
    ward_count = _initial_occupancy_count(resources.plan.ward_beds, config.initial_ward_occupancy_pct)
    icu_count = _initial_occupancy_count(resources.plan.icu_beds, config.initial_icu_occupancy_pct)
    for index in range(ward_count):
        env.process(
            _hold_initial_capacity(
                env,
                resources.ward_beds,
                config.initial_occupancy_hold_hours,
                hospital_id,
                f"initial_ward_{index}",
            )
        )
    for index in range(icu_count):
        env.process(
            _hold_initial_capacity(
                env,
                resources.icu_beds,
                config.initial_occupancy_hold_hours,
                hospital_id,
                f"initial_icu_{index}",
            )
        )


def _initial_occupancy_count(capacity: int, occupancy_pct: float) -> int:
    if capacity <= 1 or occupancy_pct <= 0:
        return 0
    return min(capacity - 1, max(0, int(round(capacity * occupancy_pct))))


def _hold_initial_capacity(
    env: simpy.Environment,
    resource: simpy.Resource,
    duration: float,
    hospital_id: str,
    blocker_id: str,
):
    with resource.request() as request:
        yield request
        yield env.timeout(duration)


def _monitor_resources(
    env: simpy.Environment,
    resources: dict[str, HospitalResources],
    state: _SimulationState,
    staff_absence_by_day: dict[int, float],
    config: SimulationConfig,
):
    try:
        while True:
            hour = float(env.now)
            day = int(hour // config.hours_per_day)
            for hospital_id, hospital_resources in resources.items():
                plan = hospital_resources.plan
                trolley_count = int(state.trolley_counts[hospital_id])
                ed_queue = len(hospital_resources.ed_cubicles.queue)
                ed_occupied = hospital_resources.ed_cubicles.count
                ward_occupied = hospital_resources.ward_beds.count
                icu_occupied = hospital_resources.icu_beds.count
                ward_occupancy_pct = 100 * ward_occupied / plan.ward_beds
                icu_occupancy_pct = 100 * icu_occupied / plan.icu_beds
                ed_crowding_pct = 100 * (ed_occupied + 0.35 * ed_queue + 0.75 * trolley_count) / plan.ed_cubicles
                staff_absence_rate = staff_absence_by_day.get(day, 0.045)
                staff_stress = calculate_staff_stress(
                    ed_crowding_pct=ed_crowding_pct,
                    ward_occupancy_pct=ward_occupancy_pct,
                    icu_occupancy_pct=icu_occupancy_pct,
                    trolley_count=trolley_count,
                    staff_absence_rate=staff_absence_rate,
                    ed_cubicles=plan.ed_cubicles,
                )
                state.hourly_metrics.append(
                    {
                        "hour": hour,
                        "day": day,
                        "hospital_id": hospital_id,
                        "ed_occupied": ed_occupied,
                        "ed_queue": ed_queue,
                        "ed_cubicles": plan.ed_cubicles,
                        "ed_crowding_pct": round(ed_crowding_pct, 3),
                        "trolley_count": trolley_count,
                        "ward_occupied": ward_occupied,
                        "ward_beds": plan.ward_beds,
                        "ward_occupancy_pct": round(ward_occupancy_pct, 3),
                        "icu_occupied": icu_occupied,
                        "icu_beds": plan.icu_beds,
                        "icu_occupancy_pct": round(icu_occupancy_pct, 3),
                        "staff_absence_rate": round(staff_absence_rate, 4),
                        "staff_stress": round(staff_stress, 4),
                    }
                )
            yield env.timeout(config.monitor_interval_hours)
    except simpy.Interrupt:
        return


def _patient_process(
    env: simpy.Environment,
    patient: dict[str, Any],
    resources: HospitalResources,
    state: _SimulationState,
    simulation_start: pd.Timestamp,
):
    arrival_hour = float(patient["arrival_hour"])
    if arrival_hour > env.now:
        yield env.timeout(arrival_hour - env.now)

    hospital_id = str(patient["hospital_id"])
    patient_id = str(patient["patient_id"])
    needs_admission = bool(patient["needs_admission"])
    needs_icu = bool(patient["needs_icu"])
    _log_event(state, env.now, "arrival", hospital_id, patient_id)

    triage_start = env.now
    with resources.triage_nurses.request() as triage_request:
        yield triage_request
        triage_start = env.now
        yield env.timeout(_triage_duration_hours(str(patient.get("acuity", "moderate"))))
    _log_event(state, env.now, "triage_complete", hospital_id, patient_id)

    ed_cubicle_request = resources.ed_cubicles.request()
    ed_doctor_request = resources.ed_doctors.request()
    yield ed_cubicle_request & ed_doctor_request
    ed_service_start = env.now
    ed_wait_hours = ed_service_start - arrival_hour
    _log_event(state, env.now, "ed_service_start", hospital_id, patient_id)
    yield env.timeout(max(0.05, float(patient["ed_service_time_hours"])))
    resources.ed_doctors.release(ed_doctor_request)

    time_to_bed_hours = 0.0
    bed_type = "none"
    inpatient_stay_hours = 0.0
    discharge_delay_hours = 0.0

    if not needs_admission:
        resources.ed_cubicles.release(ed_cubicle_request)
        exit_hour = env.now
        _log_event(state, exit_hour, "ed_discharge", hospital_id, patient_id)
    else:
        bed_request_hour = env.now
        state.trolley_counts[hospital_id] += 1
        _log_event(state, env.now, "bed_request", hospital_id, patient_id)
        if needs_icu:
            bed_type = "icu"
            icu_request = resources.icu_beds.request()
            yield icu_request
            time_to_bed_hours = env.now - bed_request_hour
            state.trolley_counts[hospital_id] -= 1
            resources.ed_cubicles.release(ed_cubicle_request)
            _log_event(state, env.now, "icu_bed_assigned", hospital_id, patient_id)
            yield from _inpatient_handover(env, resources)
            icu_stay_hours = max(1.0, float(patient["icu_los_days"]) * 24)
            inpatient_stay_hours += icu_stay_hours
            yield env.timeout(icu_stay_hours)
            ward_stay_hours = max(0.0, float(patient["ward_los_days"]) * 24)
            if ward_stay_hours > 0:
                ward_request = resources.ward_beds.request()
                yield ward_request
                resources.icu_beds.release(icu_request)
                bed_type = "icu_then_ward"
                _log_event(state, env.now, "ward_transfer", hospital_id, patient_id)
                yield env.timeout(ward_stay_hours)
                inpatient_stay_hours += ward_stay_hours
                discharge_delay_hours = yield from _run_discharge(env, resources, patient)
                resources.ward_beds.release(ward_request)
            else:
                discharge_delay_hours = yield from _run_discharge(env, resources, patient)
                resources.icu_beds.release(icu_request)
        else:
            bed_type = "ward"
            ward_request = resources.ward_beds.request()
            yield ward_request
            time_to_bed_hours = env.now - bed_request_hour
            state.trolley_counts[hospital_id] -= 1
            resources.ed_cubicles.release(ed_cubicle_request)
            _log_event(state, env.now, "ward_bed_assigned", hospital_id, patient_id)
            yield from _inpatient_handover(env, resources)
            ward_stay_hours = max(1.0, float(patient["ward_los_days"]) * 24)
            inpatient_stay_hours += ward_stay_hours
            yield env.timeout(ward_stay_hours)
            discharge_delay_hours = yield from _run_discharge(env, resources, patient)
            resources.ward_beds.release(ward_request)

        exit_hour = env.now
        _log_event(state, exit_hour, "inpatient_discharge", hospital_id, patient_id)

    state.patient_results.append(
        {
            "patient_id": patient_id,
            "hospital_id": hospital_id,
            "arrival_time": simulation_start + pd.to_timedelta(arrival_hour, unit="h"),
            "arrival_hour": round(arrival_hour, 4),
            "arrival_day": int(patient["arrival_day"]),
            "exit_time": simulation_start + pd.to_timedelta(exit_hour, unit="h"),
            "exit_hour": round(exit_hour, 4),
            "exit_day": int(exit_hour // 24),
            "needs_admission": needs_admission,
            "needs_icu": needs_icu,
            "bed_type": bed_type,
            "triage_wait_hours": round(triage_start - arrival_hour, 4),
            "ed_wait_hours": round(ed_wait_hours, 4),
            "time_to_bed_hours": round(time_to_bed_hours, 4),
            "inpatient_stay_hours": round(inpatient_stay_hours, 4),
            "discharge_delay_hours": round(discharge_delay_hours, 4),
            "total_los_hours": round(exit_hour - arrival_hour, 4),
        }
    )


def _triage_duration_hours(acuity: str) -> float:
    return {
        "low": 0.18,
        "moderate": 0.25,
        "high": 0.32,
        "critical": 0.12,
    }.get(acuity, 0.25)


def _inpatient_handover(env: simpy.Environment, resources: HospitalResources):
    with resources.inpatient_nurses.request() as request:
        yield request
        yield env.timeout(0.20)


def _run_discharge(env: simpy.Environment, resources: HospitalResources, patient: dict[str, Any]) -> float:
    delay_hours = max(0.0, float(patient["discharge_delay_days"]) * 24)
    if delay_hours > 0:
        yield env.timeout(delay_hours)
    with resources.discharge_team_capacity.request() as request:
        yield request
        yield env.timeout(0.25)
    return delay_hours


def _log_event(
    state: _SimulationState,
    hour: float,
    event: str,
    hospital_id: str,
    patient_id: str,
) -> None:
    state.event_log.append(
        {
            "hour": round(hour, 4),
            "day": int(hour // 24),
            "hospital_id": hospital_id,
            "patient_id": patient_id,
            "event": event,
        }
    )


def _build_daily_metrics(
    patient_results_df: pd.DataFrame,
    hourly_metrics_df: pd.DataFrame,
    simulation_start: pd.Timestamp,
) -> pd.DataFrame:
    if hourly_metrics_df.empty:
        return _empty_daily_metrics()

    hospitals = sorted(hourly_metrics_df["hospital_id"].unique())
    days = range(0, int(hourly_metrics_df["day"].max()) + 1)
    base = pd.MultiIndex.from_product([hospitals, days], names=["hospital_id", "day"]).to_frame(index=False)
    base["date"] = base["day"].apply(lambda day: simulation_start + pd.Timedelta(days=int(day)))

    arrivals = (
        patient_results_df.groupby(["hospital_id", "arrival_day"])
        .agg(
            ed_arrivals=("patient_id", "count"),
            admissions=("needs_admission", "sum"),
            icu_admissions=("needs_icu", "sum"),
            mean_ed_wait_hours=("ed_wait_hours", "mean"),
            p90_ed_wait_hours=("ed_wait_hours", lambda values: _percentile(values, 90)),
        )
        .reset_index()
        .rename(columns={"arrival_day": "day"})
    )
    admitted = patient_results_df[patient_results_df["needs_admission"]]
    if admitted.empty:
        time_to_bed = pd.DataFrame(columns=["hospital_id", "day", "mean_time_to_bed_hours", "p90_time_to_bed_hours"])
    else:
        time_to_bed = (
            admitted.groupby(["hospital_id", "arrival_day"])
            .agg(
                mean_time_to_bed_hours=("time_to_bed_hours", "mean"),
                p90_time_to_bed_hours=("time_to_bed_hours", lambda values: _percentile(values, 90)),
            )
            .reset_index()
            .rename(columns={"arrival_day": "day"})
        )
    discharges = (
        patient_results_df.groupby(["hospital_id", "exit_day"])
        .size()
        .reset_index(name="discharges")
        .rename(columns={"exit_day": "day"})
    )
    hourly = (
        hourly_metrics_df.groupby(["hospital_id", "day"])
        .agg(
            max_trolley_count=("trolley_count", "max"),
            mean_ward_occupancy_pct=("ward_occupancy_pct", "mean"),
            max_ward_occupancy_pct=("ward_occupancy_pct", "max"),
            mean_icu_occupancy_pct=("icu_occupancy_pct", "mean"),
            max_icu_occupancy_pct=("icu_occupancy_pct", "max"),
            mean_staff_stress=("staff_stress", "mean"),
            peak_ed_crowding_pct=("ed_crowding_pct", "max"),
        )
        .reset_index()
    )

    daily = base.merge(arrivals, on=["hospital_id", "day"], how="left")
    daily = daily.merge(time_to_bed, on=["hospital_id", "day"], how="left")
    daily = daily.merge(discharges, on=["hospital_id", "day"], how="left")
    daily = daily.merge(hourly, on=["hospital_id", "day"], how="left")

    count_columns = ["ed_arrivals", "admissions", "icu_admissions", "discharges", "max_trolley_count"]
    value_columns = [
        "mean_ed_wait_hours",
        "p90_ed_wait_hours",
        "mean_time_to_bed_hours",
        "p90_time_to_bed_hours",
        "mean_ward_occupancy_pct",
        "max_ward_occupancy_pct",
        "mean_icu_occupancy_pct",
        "max_icu_occupancy_pct",
        "mean_staff_stress",
        "peak_ed_crowding_pct",
    ]
    daily[count_columns] = daily[count_columns].fillna(0).astype(int)
    daily[value_columns] = daily[value_columns].fillna(0.0)
    daily["risk_score"] = daily.apply(
        lambda row: calculate_risk_score(
            ed_crowding_pct=float(row["peak_ed_crowding_pct"]),
            ward_occupancy_pct=float(row["max_ward_occupancy_pct"]),
            icu_occupancy_pct=float(row["max_icu_occupancy_pct"]),
            trolley_count=int(row["max_trolley_count"]),
            mean_staff_stress=float(row["mean_staff_stress"]),
            mean_ed_wait_hours=float(row["mean_ed_wait_hours"]),
        ),
        axis=1,
    )
    daily["risk_score"] = daily["risk_score"].round(2)
    daily["risk_label"] = daily["risk_score"].map(risk_label)

    ordered_columns = [
        "date",
        "day",
        "hospital_id",
        "ed_arrivals",
        "admissions",
        "icu_admissions",
        "discharges",
        "mean_ed_wait_hours",
        "p90_ed_wait_hours",
        "mean_time_to_bed_hours",
        "p90_time_to_bed_hours",
        "max_trolley_count",
        "mean_ward_occupancy_pct",
        "max_ward_occupancy_pct",
        "mean_icu_occupancy_pct",
        "max_icu_occupancy_pct",
        "mean_staff_stress",
        "risk_score",
        "risk_label",
        "peak_ed_crowding_pct",
    ]
    return daily[ordered_columns]


def _percentile(values: pd.Series, percentile: int) -> float:
    if values.empty:
        return 0.0
    return float(np.percentile(values, percentile))


def _empty_patient_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient_id",
            "hospital_id",
            "arrival_time",
            "arrival_hour",
            "arrival_day",
            "exit_time",
            "exit_hour",
            "exit_day",
            "needs_admission",
            "needs_icu",
            "bed_type",
            "triage_wait_hours",
            "ed_wait_hours",
            "time_to_bed_hours",
            "inpatient_stay_hours",
            "discharge_delay_hours",
            "total_los_hours",
        ]
    )


def _empty_hourly_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "hour",
            "day",
            "hospital_id",
            "ed_occupied",
            "ed_queue",
            "ed_cubicles",
            "ed_crowding_pct",
            "trolley_count",
            "ward_occupied",
            "ward_beds",
            "ward_occupancy_pct",
            "icu_occupied",
            "icu_beds",
            "icu_occupancy_pct",
            "staff_absence_rate",
            "staff_stress",
        ]
    )


def _empty_daily_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "day",
            "hospital_id",
            "ed_arrivals",
            "admissions",
            "icu_admissions",
            "discharges",
            "mean_ed_wait_hours",
            "p90_ed_wait_hours",
            "mean_time_to_bed_hours",
            "p90_time_to_bed_hours",
            "max_trolley_count",
            "mean_ward_occupancy_pct",
            "max_ward_occupancy_pct",
            "mean_icu_occupancy_pct",
            "max_icu_occupancy_pct",
            "mean_staff_stress",
            "risk_score",
            "risk_label",
            "peak_ed_crowding_pct",
        ]
    )


def _empty_event_log() -> pd.DataFrame:
    return pd.DataFrame(columns=["hour", "day", "hospital_id", "patient_id", "event"])
