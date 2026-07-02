from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Hospital:
    """Synthetic hospital profile used by the command-center demo."""

    hospital_id: str
    name: str
    type: str
    region: str
    latitude: float
    longitude: float
    regional_population: int
    ed_cubicles: int
    general_beds: int
    icu_beds: int
    baseline_ed_arrivals_per_day: float
    baseline_admission_rate: float
    baseline_icu_admission_rate: float
    nurses: int
    doctors: int
    healthcare_assistants: int
    discharge_delay_rate: float
    average_discharge_delay_days: float
    elective_beds_per_day: int
    transfer_partners: list[str]


_FICTIONAL_HOSPITAL_SITES = (
    ("North Dublin General", "Dublin North", 53.36, -6.25),
    ("Cork Harbour University Hospital", "Cork-Kerry", 51.90, -8.47),
    ("Galway Bay Regional Hospital", "West", 53.27, -9.05),
    ("Limerick Shannon General", "Mid-West", 52.66, -8.63),
    ("Waterford Estuary Hospital", "South East", 52.26, -7.11),
    ("Sligo Atlantic Hospital", "North West", 54.27, -8.48),
    ("Athlone Midlands Medical Centre", "Midlands", 53.42, -7.94),
    ("Drogheda Boyne Valley Hospital", "North East", 53.72, -6.35),
    ("Kerry Lakes District Hospital", "South West", 52.06, -9.51),
    ("Mayo Greenway General", "West", 53.85, -9.30),
)

_TYPE_PROFILES = {
    "small": {
        "regional_population": (85_000, 180_000),
        "ed_cubicles": (12, 22),
        "general_beds": (90, 180),
        "icu_beds": (4, 10),
        "arrivals": (55, 105),
        "admission_rate": (0.16, 0.24),
        "icu_rate": (0.015, 0.030),
        "elective_beds": (4, 10),
        "delay_rate": (0.08, 0.17),
        "delay_days": (1.0, 2.4),
    },
    "medium": {
        "regional_population": (190_000, 390_000),
        "ed_cubicles": (24, 42),
        "general_beds": (210, 420),
        "icu_beds": (12, 24),
        "arrivals": (115, 210),
        "admission_rate": (0.21, 0.30),
        "icu_rate": (0.025, 0.045),
        "elective_beds": (12, 24),
        "delay_rate": (0.10, 0.21),
        "delay_days": (1.4, 3.0),
    },
    "tertiary": {
        "regional_population": (430_000, 820_000),
        "ed_cubicles": (46, 76),
        "general_beds": (520, 900),
        "icu_beds": (32, 58),
        "arrivals": (230, 390),
        "admission_rate": (0.26, 0.36),
        "icu_rate": (0.040, 0.070),
        "elective_beds": (25, 46),
        "delay_rate": (0.12, 0.24),
        "delay_days": (1.8, 3.6),
    },
}

_DEFAULT_TYPE_MIX = (
    "tertiary",
    "medium",
    "medium",
    "small",
    "medium",
    "tertiary",
    "small",
    "medium",
    "small",
    "medium",
)


def generate_hospital_network(n_hospitals: int = 8, seed: int = 42) -> list[Hospital]:
    """Generate a deterministic fictional Irish-style hospital network."""

    if not 5 <= n_hospitals <= 10:
        raise ValueError("n_hospitals must be between 5 and 10.")

    rng = np.random.default_rng(seed)
    site_indices = rng.choice(len(_FICTIONAL_HOSPITAL_SITES), size=n_hospitals, replace=False)
    hospital_types = list(_DEFAULT_TYPE_MIX[:n_hospitals])
    rng.shuffle(hospital_types)

    hospitals: list[Hospital] = []
    for index, (site_index, hospital_type) in enumerate(zip(site_indices, hospital_types), start=1):
        name, region, base_latitude, base_longitude = _FICTIONAL_HOSPITAL_SITES[int(site_index)]
        profile = _TYPE_PROFILES[hospital_type]

        regional_population = _integer_in_range(rng, profile["regional_population"])
        ed_cubicles = _integer_in_range(rng, profile["ed_cubicles"])
        general_beds = _integer_in_range(rng, profile["general_beds"])
        icu_beds = _integer_in_range(rng, profile["icu_beds"])
        baseline_ed_arrivals = float(round(rng.uniform(*profile["arrivals"]), 1))
        baseline_admission_rate = float(round(rng.uniform(*profile["admission_rate"]), 3))
        baseline_icu_admission_rate = float(round(rng.uniform(*profile["icu_rate"]), 3))

        nurses = int(round(general_beds * rng.uniform(1.65, 2.15) + icu_beds * rng.uniform(4.0, 5.8)))
        doctors = int(round(general_beds * rng.uniform(0.23, 0.34) + icu_beds * rng.uniform(0.9, 1.4)))
        healthcare_assistants = int(round(general_beds * rng.uniform(0.42, 0.68)))

        hospitals.append(
            Hospital(
                hospital_id=f"H{index:03d}",
                name=name,
                type=hospital_type,
                region=region,
                latitude=float(round(base_latitude + rng.normal(0, 0.035), 4)),
                longitude=float(round(base_longitude + rng.normal(0, 0.045), 4)),
                regional_population=regional_population,
                ed_cubicles=ed_cubicles,
                general_beds=general_beds,
                icu_beds=icu_beds,
                baseline_ed_arrivals_per_day=baseline_ed_arrivals,
                baseline_admission_rate=baseline_admission_rate,
                baseline_icu_admission_rate=baseline_icu_admission_rate,
                nurses=nurses,
                doctors=doctors,
                healthcare_assistants=healthcare_assistants,
                discharge_delay_rate=float(round(rng.uniform(*profile["delay_rate"]), 3)),
                average_discharge_delay_days=float(round(rng.uniform(*profile["delay_days"]), 2)),
                elective_beds_per_day=_integer_in_range(rng, profile["elective_beds"]),
                transfer_partners=[],
            )
        )

    return _with_transfer_partners(hospitals, rng)


def hospitals_to_dataframe(hospitals: list[Hospital]) -> pd.DataFrame:
    """Convert synthetic hospital profiles to a stable tabular representation."""

    return pd.DataFrame([asdict(hospital) for hospital in hospitals])


def get_default_hospitals(seed: int = 42) -> pd.DataFrame:
    """Return the default synthetic hospital network as a DataFrame."""

    return hospitals_to_dataframe(generate_hospital_network(seed=seed))


def _integer_in_range(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    low, high = bounds
    return int(rng.integers(low, high + 1))


def _with_transfer_partners(hospitals: list[Hospital], rng: np.random.Generator) -> list[Hospital]:
    hospital_ids = [hospital.hospital_id for hospital in hospitals]
    tertiary_ids = [hospital.hospital_id for hospital in hospitals if hospital.type == "tertiary"]
    updated: list[Hospital] = []

    for hospital in hospitals:
        candidate_ids = [hospital_id for hospital_id in hospital_ids if hospital_id != hospital.hospital_id]
        preferred_ids = [hospital_id for hospital_id in tertiary_ids if hospital_id != hospital.hospital_id]
        partner_pool = preferred_ids + [hospital_id for hospital_id in candidate_ids if hospital_id not in preferred_ids]
        partner_count = min(len(candidate_ids), 3 if hospital.type == "small" else 2)
        transfer_partners = list(rng.choice(partner_pool, size=partner_count, replace=False))
        updated.append(
            Hospital(
                **{
                    **asdict(hospital),
                    "transfer_partners": sorted(str(partner) for partner in transfer_partners),
                }
            )
        )

    return updated

