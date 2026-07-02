from __future__ import annotations

from dataclasses import dataclass

import simpy


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the SimPy patient-flow engine."""

    hours_per_day: int = 24
    initial_ward_occupancy_pct: float = 0.82
    initial_icu_occupancy_pct: float = 0.70
    initial_occupancy_hold_hours: float = 96.0
    monitor_interval_hours: float = 1.0


@dataclass(frozen=True)
class HospitalResourcePlan:
    """Synthetic hospital resource capacities used by the simulation."""

    hospital_id: str
    triage_nurses: int
    ed_cubicles: int
    ed_doctors: int
    ward_beds: int
    icu_beds: int
    inpatient_nurses: int
    discharge_team_capacity: int


@dataclass
class HospitalResources:
    """Live SimPy resources for one hospital."""

    plan: HospitalResourcePlan
    triage_nurses: simpy.Resource
    ed_cubicles: simpy.Resource
    ed_doctors: simpy.Resource
    ward_beds: simpy.Resource
    icu_beds: simpy.Resource
    inpatient_nurses: simpy.Resource
    discharge_team_capacity: simpy.Resource


def build_resource_plan(hospital: dict[str, object]) -> HospitalResourcePlan:
    """Translate a hospital profile into resource capacities."""

    ed_cubicles = max(1, int(hospital["ed_cubicles"]))
    general_beds = max(1, int(hospital["general_beds"]))
    icu_beds = max(1, int(hospital["icu_beds"]))
    nurses = max(1, int(hospital["nurses"]))

    return HospitalResourcePlan(
        hospital_id=str(hospital["hospital_id"]),
        triage_nurses=max(1, round(ed_cubicles / 8)),
        ed_cubicles=ed_cubicles,
        ed_doctors=max(1, round(ed_cubicles / 4)),
        ward_beds=general_beds,
        icu_beds=icu_beds,
        inpatient_nurses=max(1, round(nurses / 8)),
        discharge_team_capacity=max(1, round(general_beds / 75)),
    )


def build_simpy_resources(env: simpy.Environment, plan: HospitalResourcePlan) -> HospitalResources:
    """Create live SimPy resources from a resource plan."""

    return HospitalResources(
        plan=plan,
        triage_nurses=simpy.Resource(env, capacity=plan.triage_nurses),
        ed_cubicles=simpy.Resource(env, capacity=plan.ed_cubicles),
        ed_doctors=simpy.Resource(env, capacity=plan.ed_doctors),
        ward_beds=simpy.Resource(env, capacity=plan.ward_beds),
        icu_beds=simpy.Resource(env, capacity=plan.icu_beds),
        inpatient_nurses=simpy.Resource(env, capacity=plan.inpatient_nurses),
        discharge_team_capacity=simpy.Resource(env, capacity=plan.discharge_team_capacity),
    )

