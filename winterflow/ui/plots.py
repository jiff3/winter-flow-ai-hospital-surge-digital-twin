from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from winterflow.ui.theme import CHART_COLORS, RAG_COLORS


def apply_command_center_layout(fig: go.Figure, title: str, yaxis_title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#0F172A"},
        margin={"l": 20, "r": 20, "t": 58, "b": 28},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.18},
    )
    if yaxis_title:
        fig.update_yaxes(title_text=yaxis_title, gridcolor="#E2E8F0")
    fig.update_xaxes(gridcolor="#E2E8F0")
    return fig


def plot_scenario_curves(scenario_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    curves = [
        ("flu_multiplier", "Flu", CHART_COLORS["steel"]),
        ("rsv_multiplier", "RSV", CHART_COLORS["teal"]),
        ("covid_multiplier", "COVID", CHART_COLORS["amber"]),
        ("combined_demand_multiplier", "Combined demand", CHART_COLORS["red"]),
    ]
    for column, label, color in curves:
        fig.add_trace(go.Scatter(x=scenario_df["date"], y=scenario_df[column], mode="lines", name=label, line={"color": color}))
    fig.update_xaxes(title_text="Date")
    return apply_command_center_layout(fig, "Winter Virus Waves", "Multiplier")


def plot_staff_absence(scenario_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scenario_df["date"],
            y=scenario_df["staff_absence_rate"],
            mode="lines",
            name="Staff absence rate",
            line={"color": CHART_COLORS["amber"]},
        )
    )
    fig.update_xaxes(title_text="Date")
    return apply_command_center_layout(fig, "Staff Absence Pressure", "Rate")


def plot_daily_demand(daily_demand_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for hospital_id, hospital_df in daily_demand_df.groupby("hospital_id"):
        fig.add_trace(go.Scatter(x=hospital_df["date"], y=hospital_df["synthetic_arrivals"], mode="lines", name=hospital_id))
    fig.update_xaxes(title_text="Date")
    return apply_command_center_layout(fig, "Generated Daily ED Demand By Hospital", "Synthetic arrivals")


def plot_simulation_occupancy(daily_metrics_df: pd.DataFrame) -> go.Figure:
    regional = (
        daily_metrics_df.groupby("day")
        .agg(
            peak_ed_crowding_pct=("peak_ed_crowding_pct", "max"),
            peak_ward_occupancy_pct=("max_ward_occupancy_pct", "max"),
            peak_icu_occupancy_pct=("max_icu_occupancy_pct", "max"),
        )
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=regional["day"], y=regional["peak_ed_crowding_pct"], mode="lines", name="ED crowding"))
    fig.add_trace(go.Scatter(x=regional["day"], y=regional["peak_ward_occupancy_pct"], mode="lines", name="Ward occupancy"))
    fig.add_trace(go.Scatter(x=regional["day"], y=regional["peak_icu_occupancy_pct"], mode="lines", name="ICU occupancy"))
    fig.update_xaxes(title_text="Simulation day")
    return apply_command_center_layout(fig, "Daily Peak Occupancy And Crowding", "Percent")


def plot_trolley_counts(daily_metrics_df: pd.DataFrame) -> go.Figure:
    regional = daily_metrics_df.groupby("day").agg(regional_trolley_count=("max_trolley_count", "sum")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=regional["day"], y=regional["regional_trolley_count"], name="Regional trolley count", marker_color=CHART_COLORS["red"]))
    fig.update_xaxes(title_text="Simulation day")
    return apply_command_center_layout(fig, "Regional Daily Trolley Count", "Patients waiting for beds")


def plot_risk_heatmap(daily_metrics_df: pd.DataFrame) -> go.Figure:
    pivot = daily_metrics_df.pivot(index="hospital_id", columns="day", values="risk_score").fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0.0, RAG_COLORS["Green"]],
                [0.40, "#FACC15"],
                [0.70, "#F97316"],
                [1.0, RAG_COLORS["Red"]],
            ],
            zmin=0,
            zmax=100,
            colorbar={"title": "Risk"},
        )
    )
    fig.update_xaxes(title_text="Simulation day")
    fig.update_yaxes(title_text="Hospital")
    return apply_command_center_layout(fig, "Operational Risk Heatmap")


def plot_hospital_network(hospitals_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    hospital_lookup = hospitals_df.set_index("hospital_id").to_dict("index")
    for hospital in hospitals_df.to_dict("records"):
        partners = hospital.get("transfer_partners", [])
        if not isinstance(partners, (list, tuple)):
            partners = []
        for partner_id in partners:
            if partner_id in hospital_lookup:
                partner = hospital_lookup[partner_id]
                fig.add_trace(
                    go.Scattergeo(
                        lon=[hospital["longitude"], partner["longitude"]],
                        lat=[hospital["latitude"], partner["latitude"]],
                        mode="lines",
                        line={"width": 1, "color": "#94A3B8"},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    type_colors = {"small": CHART_COLORS["teal"], "medium": CHART_COLORS["steel"], "tertiary": CHART_COLORS["red"]}
    for hospital_type, group_df in hospitals_df.groupby("type"):
        fig.add_trace(
            go.Scattergeo(
                lon=group_df["longitude"],
                lat=group_df["latitude"],
                text=group_df["name"],
                customdata=group_df[["hospital_id", "general_beds", "icu_beds"]],
                mode="markers+text",
                textposition="top center",
                name=hospital_type.title(),
                marker={
                    "size": (group_df["general_beds"] / group_df["general_beds"].max() * 22 + 8),
                    "color": type_colors.get(hospital_type, CHART_COLORS["slate"]),
                    "line": {"width": 1, "color": "#FFFFFF"},
                },
                hovertemplate="<b>%{text}</b><br>ID: %{customdata[0]}<br>Ward beds: %{customdata[1]}<br>ICU beds: %{customdata[2]}<extra></extra>",
            )
        )

    fig.update_geos(
        scope="europe",
        lataxis_range=[51.2, 55.4],
        lonaxis_range=[-10.8, -5.4],
        showland=True,
        landcolor="#F8FAFC",
        showcountries=True,
        countrycolor="#CBD5E1",
        showocean=True,
        oceancolor="#E0F2FE",
    )
    fig.update_layout(height=560, legend={"orientation": "h", "y": -0.08})
    return apply_command_center_layout(fig, "Synthetic Irish-Style Hospital Network")


def plot_before_after_timeseries(
    baseline_daily_df: pd.DataFrame,
    policy_daily_df: pd.DataFrame,
    metric: str = "max_trolley_count",
) -> go.Figure:
    baseline = _regional_series(baseline_daily_df, metric)
    policy = _regional_series(policy_daily_df, metric)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=baseline["day"], y=baseline["value"], mode="lines", name="Baseline", line={"color": CHART_COLORS["slate"]}))
    fig.add_trace(go.Scatter(x=policy["day"], y=policy["value"], mode="lines", name="Policy", line={"color": CHART_COLORS["green"]}))
    fig.update_xaxes(title_text="Simulation day")
    return apply_command_center_layout(fig, f"Before/After: {metric.replace('_', ' ').title()}", "Regional value")


def plot_comparison_bars(comparison_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comparison_df["metric"], y=comparison_df["baseline"], name="Baseline", marker_color=CHART_COLORS["slate"]))
    fig.add_trace(go.Bar(x=comparison_df["metric"], y=comparison_df["policy"], name="Policy", marker_color=CHART_COLORS["green"]))
    fig.update_layout(barmode="group")
    return apply_command_center_layout(fig, "Before/After Metrics")


def plot_forecast_interval(forecast_df: pd.DataFrame, target_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["p90"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["p10"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",
            line={"width": 0},
            name="P10-P90 interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["p50"],
            mode="lines+markers",
            name="Median forecast",
            line={"color": CHART_COLORS["steel"], "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["baseline"],
            mode="lines",
            name="Rolling baseline",
            line={"color": CHART_COLORS["slate"], "dash": "dash"},
        )
    )
    fig.update_xaxes(title_text="Date")
    return apply_command_center_layout(fig, f"{target_label} Forecast", "Daily value")


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    ranked = importance_df.sort_values("importance", ascending=True).tail(12)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ranked["importance"],
            y=ranked["feature"],
            orientation="h",
            marker_color=CHART_COLORS["teal"],
            name="Importance",
        )
    )
    fig.update_xaxes(title_text="Permutation importance")
    return apply_command_center_layout(fig, "Feature Importance")


def plot_optimizer_scores(recommendations_df: pd.DataFrame) -> go.Figure:
    top = recommendations_df.head(5).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["actions"], y=top["baseline_score"], name="Baseline", marker_color=CHART_COLORS["slate"]))
    fig.add_trace(go.Bar(x=top["actions"], y=top["projected_score"], name="Projected", marker_color=CHART_COLORS["green"]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Candidate intervention", tickangle=-25)
    return apply_command_center_layout(fig, "Optimizer Before/After Objective Score", "Objective score")


def _regional_series(daily_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "max_trolley_count":
        return daily_df.groupby("day")[metric].sum().reset_index(name="value")
    if metric in {"risk_score", "mean_staff_stress"}:
        return daily_df.groupby("day")[metric].mean().reset_index(name="value")
    return daily_df.groupby("day")[metric].max().reset_index(name="value")
