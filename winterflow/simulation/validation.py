from __future__ import annotations

import pandas as pd


REQUIRED_HOSPITAL_COLUMNS = {
    "hospital_id",
    "ed_cubicles",
    "general_beds",
    "icu_beds",
    "nurses",
}

REQUIRED_PATIENT_COLUMNS = {
    "patient_id",
    "hospital_id",
    "arrival_time",
    "arrival_day",
    "needs_admission",
    "needs_icu",
    "ed_service_time_hours",
    "ward_los_days",
    "icu_los_days",
    "discharge_delay_days",
}


def validate_simulation_inputs(hospitals_df: pd.DataFrame, patient_arrivals_df: pd.DataFrame) -> None:
    """Validate required columns before starting the SimPy engine."""

    missing_hospital_columns = REQUIRED_HOSPITAL_COLUMNS - set(hospitals_df.columns)
    missing_patient_columns = REQUIRED_PATIENT_COLUMNS - set(patient_arrivals_df.columns)
    if missing_hospital_columns:
        raise ValueError(f"Hospital data is missing columns: {sorted(missing_hospital_columns)}")
    if missing_patient_columns:
        raise ValueError(f"Patient arrivals are missing columns: {sorted(missing_patient_columns)}")
    if hospitals_df["hospital_id"].duplicated().any():
        raise ValueError("Hospital IDs must be unique for simulation.")
    if (patient_arrivals_df["ed_service_time_hours"] < 0).any():
        raise ValueError("ED service times must be non-negative.")
    if patient_arrivals_df.loc[patient_arrivals_df["needs_icu"], "needs_admission"].eq(False).any():
        raise ValueError("Patients needing ICU must also need admission.")

