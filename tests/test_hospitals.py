from winterflow.data.hospitals import generate_hospital_network, hospitals_to_dataframe


REQUIRED_COLUMNS = {
    "hospital_id",
    "name",
    "type",
    "region",
    "latitude",
    "longitude",
    "regional_population",
    "ed_cubicles",
    "general_beds",
    "icu_beds",
    "baseline_ed_arrivals_per_day",
    "baseline_admission_rate",
    "baseline_icu_admission_rate",
    "nurses",
    "doctors",
    "healthcare_assistants",
    "discharge_delay_rate",
    "average_discharge_delay_days",
    "elective_beds_per_day",
    "transfer_partners",
}


def test_required_columns_exist() -> None:
    df = hospitals_to_dataframe(generate_hospital_network(seed=7))

    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_hospital_count_is_between_five_and_ten() -> None:
    hospitals = generate_hospital_network(seed=7)

    assert 5 <= len(hospitals) <= 10


def test_hospital_ids_are_unique() -> None:
    df = hospitals_to_dataframe(generate_hospital_network(seed=7))

    assert df["hospital_id"].is_unique


def test_capacity_values_are_positive() -> None:
    df = hospitals_to_dataframe(generate_hospital_network(seed=7))
    capacity_columns = [
        "regional_population",
        "ed_cubicles",
        "general_beds",
        "icu_beds",
        "baseline_ed_arrivals_per_day",
        "baseline_admission_rate",
        "baseline_icu_admission_rate",
        "nurses",
        "doctors",
        "healthcare_assistants",
        "average_discharge_delay_days",
        "elective_beds_per_day",
    ]

    assert (df[capacity_columns] > 0).all().all()


def test_tertiary_hospitals_have_larger_capacity_than_small_hospitals_when_both_exist() -> None:
    df = hospitals_to_dataframe(generate_hospital_network(n_hospitals=10, seed=7))

    if {"tertiary", "small"}.issubset(set(df["type"])):
        tertiary_capacity = df.loc[df["type"] == "tertiary", "general_beds"].mean()
        small_capacity = df.loc[df["type"] == "small", "general_beds"].mean()

        assert tertiary_capacity > small_capacity

