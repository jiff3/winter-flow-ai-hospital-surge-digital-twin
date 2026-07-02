from winterflow.data.hospitals import get_default_hospitals
from winterflow.forecasting.evaluation import evaluate_models
from winterflow.forecasting.features import create_features, generate_synthetic_historical_demand
from winterflow.forecasting.models import forecast_next_days, train_forecast_models


def _history():
    hospitals = get_default_hospitals(seed=12).head(2)
    return generate_synthetic_historical_demand(hospitals, years=2, seed=12)


def test_historical_data_generation_returns_expected_columns() -> None:
    history = _history()
    expected_columns = {
        "date",
        "hospital_id",
        "ed_arrivals",
        "admissions",
        "icu_admissions",
        "risk_score",
        "trolley_count",
    }

    assert expected_columns.issubset(history.columns)
    assert not history.empty


def test_feature_generation_works() -> None:
    features = create_features(_history())

    assert "ed_arrivals_lag_1" in features.columns
    assert "sin_day_of_year" in features.columns
    assert features.attrs["feature_columns"]


def test_model_trains() -> None:
    models = train_forecast_models(_history(), targets=["ed_arrivals"])

    assert "ed_arrivals" in models["models"]
    assert "p50" in models["models"]["ed_arrivals"]


def test_forecast_has_correct_horizon() -> None:
    history = _history()
    models = train_forecast_models(history, targets=["ed_arrivals"])
    recent = history.loc[history["hospital_id"] == history["hospital_id"].iloc[0]].tail(90)

    forecast = forecast_next_days(models, recent, horizon=7)

    assert len(forecast) == 7
    assert forecast["day"].max() == 7


def test_prediction_intervals_are_ordered() -> None:
    history = _history()
    models = train_forecast_models(history, targets=["ed_arrivals"])
    recent = history.loc[history["hospital_id"] == history["hospital_id"].iloc[0]].tail(90)

    forecast = forecast_next_days(models, recent, horizon=7)

    assert (forecast["p10"] <= forecast["p50"]).all()
    assert (forecast["p50"] <= forecast["p90"]).all()


def test_evaluation_metrics_are_returned() -> None:
    metrics = evaluate_models(_history(), targets=["ed_arrivals"])

    assert {"target", "model", "mae", "rmse", "mape"}.issubset(metrics.columns)
    assert not metrics.empty

