from __future__ import annotations

import numpy as np
import pandas as pd

from winterflow.forecasting.features import FORECAST_TARGETS, create_features, get_feature_columns
from winterflow.forecasting.models import _fit_quantile_model


def evaluate_models(
    history_df: pd.DataFrame,
    targets: list[str] | tuple[str, ...] = FORECAST_TARGETS,
) -> pd.DataFrame:
    """Evaluate baseline rolling averages against ML median forecasts."""

    feature_df = create_features(history_df).sort_values("date")
    feature_columns = get_feature_columns(feature_df)
    split_date = feature_df["date"].quantile(0.80)
    train_df = feature_df[feature_df["date"] <= split_date]
    test_df = feature_df[feature_df["date"] > split_date]
    if test_df.empty:
        test_df = feature_df.tail(max(14, len(feature_df) // 5))
        train_df = feature_df.drop(test_df.index)

    rows: list[dict[str, object]] = []
    for target in targets:
        model = _fit_quantile_model(train_df[feature_columns], train_df[target], quantile=0.50, seed=31)
        ml_predictions = np.clip(model.predict(test_df[feature_columns]), 0, None)
        baseline_column = f"{target}_rolling_14"
        baseline_predictions = np.clip(test_df[baseline_column].to_numpy(), 0, None)
        actual = test_df[target].to_numpy()
        rows.append(_metrics_row(target, "rolling_average", actual, baseline_predictions))
        rows.append(_metrics_row(target, "hist_gradient_boosting_p50", actual, ml_predictions))
    return pd.DataFrame(rows)


def _metrics_row(target: str, model_name: str, actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    error = actual - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = np.where(actual == 0, 1, actual)
    mape = float(np.mean(np.abs(error) / denominator) * 100)
    return {
        "target": target,
        "model": model_name,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 3),
    }

