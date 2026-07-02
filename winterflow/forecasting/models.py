from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from winterflow.forecasting.features import FORECAST_TARGETS, create_features, get_feature_columns


@dataclass
class RollingAverageModel:
    target: str
    window: int = 14
    fallback_: float = 0.0

    def fit(self, history_df: pd.DataFrame) -> "RollingAverageModel":
        self.fallback_ = float(history_df[self.target].tail(self.window).mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.repeat(self.fallback_, horizon)


def train_forecast_models(
    history_df: pd.DataFrame,
    targets: list[str] | tuple[str, ...] = FORECAST_TARGETS,
) -> dict[str, object]:
    """Train baseline and quantile forecasting models for the requested targets."""

    feature_df = create_features(history_df)
    feature_columns = get_feature_columns(feature_df)
    models: dict[str, object] = {
        "targets": list(targets),
        "feature_columns": feature_columns,
        "models": {},
        "feature_importance": {},
    }
    train_df = feature_df.sort_values("date")
    X = train_df[feature_columns]

    for target in targets:
        y = train_df[target]
        target_models = {
            "baseline": RollingAverageModel(target=target).fit(train_df),
            "p10": _fit_quantile_model(X, y, quantile=0.10, seed=11),
            "p50": _fit_quantile_model(X, y, quantile=0.50, seed=17),
            "p90": _fit_quantile_model(X, y, quantile=0.90, seed=23),
        }
        models["models"][target] = target_models
        models["feature_importance"][target] = _calculate_feature_importance(target_models["p50"], X, y, feature_columns)
    return models


def forecast_next_days(
    models: dict[str, object],
    recent_df: pd.DataFrame,
    horizon: int = 14,
) -> pd.DataFrame:
    """Forecast the next horizon days for each trained target."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    targets = list(models["targets"])
    feature_columns = list(models["feature_columns"])
    model_lookup = models["models"]
    recent_sorted = recent_df.copy()
    recent_sorted["date"] = pd.to_datetime(recent_sorted["date"])
    recent_sorted = recent_sorted.sort_values("date")
    extended_df = recent_sorted.copy()
    forecast_rows: list[dict[str, object]] = []

    for step in range(1, horizon + 1):
        next_row = _future_template_row(extended_df, step)
        temp_df = pd.concat([extended_df, pd.DataFrame([next_row])], ignore_index=True)
        feature_df = create_features(temp_df)
        future_features = _align_features(feature_df.iloc[[-1]], feature_columns)

        predicted_values: dict[str, float] = {}
        for target in targets:
            target_models = model_lookup[target]
            raw_p10 = float(target_models["p10"].predict(future_features)[0])
            raw_p50 = float(target_models["p50"].predict(future_features)[0])
            raw_p90 = float(target_models["p90"].predict(future_features)[0])
            p10, p50, p90 = np.sort([raw_p10, raw_p50, raw_p90])
            p10 = max(0.0, float(p10))
            p50 = max(p10, float(p50))
            p90 = max(p50, float(p90))
            predicted_values[target] = p50
            forecast_rows.append(
                {
                    "date": next_row["date"],
                    "day": step,
                    "hospital_id": next_row["hospital_id"],
                    "target": target,
                    "baseline": float(target_models["baseline"].predict(horizon)[step - 1]),
                    "p10": round(p10, 2),
                    "p50": round(p50, 2),
                    "p90": round(p90, 2),
                }
            )

        for target, value in predicted_values.items():
            next_row[target] = value
        extended_df = pd.concat([extended_df, pd.DataFrame([next_row])], ignore_index=True)

    return pd.DataFrame(forecast_rows)


def get_feature_importance(models: dict[str, object], target: str) -> pd.DataFrame:
    return models["feature_importance"][target].copy()


def _fit_quantile_model(
    X: pd.DataFrame,
    y: pd.Series,
    quantile: float,
    seed: int,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=55,
        learning_rate=0.06,
        l2_regularization=0.04,
        min_samples_leaf=18,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def _calculate_feature_importance(
    model: HistGradientBoostingRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:
    sample_size = min(350, len(X))
    X_sample = X.tail(sample_size)
    y_sample = y.tail(sample_size)
    result = permutation_importance(model, X_sample, y_sample, n_repeats=2, random_state=19)
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": result.importances_mean,
        }
    )
    return importance_df.sort_values("importance", ascending=False).head(15).reset_index(drop=True)


def _future_template_row(extended_df: pd.DataFrame, step: int) -> dict[str, object]:
    last_row = extended_df.iloc[-1].to_dict()
    next_date = pd.Timestamp(last_row["date"]) + pd.Timedelta(days=1)
    new_row = last_row.copy()
    new_row["date"] = next_date
    new_row["day"] = int(last_row.get("day", 0)) + 1
    new_row["history_day"] = int(last_row.get("history_day", 0)) + 1
    for target in FORECAST_TARGETS:
        new_row[target] = float(extended_df[target].tail(14).mean()) if target in extended_df else 0.0
    for column in [
        "combined_demand_multiplier",
        "flu_multiplier",
        "rsv_multiplier",
        "covid_multiplier",
        "severity_shift",
        "admission_pressure_multiplier",
        "icu_pressure_multiplier",
        "staff_absence_rate",
    ]:
        if column in extended_df:
            new_row[column] = float(extended_df[column].tail(14).mean())
    return new_row


def _align_features(feature_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = feature_df.copy()
    for column in feature_columns:
        if column not in aligned:
            aligned[column] = 0
    return aligned[feature_columns]

