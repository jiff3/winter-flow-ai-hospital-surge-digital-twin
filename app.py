from __future__ import annotations

import pandas as pd
import streamlit as st

from winterflow.constants import APP_SUBTITLE, APP_TITLE, SYNTHETIC_DATA_DISCLAIMER
from winterflow.data.hospitals import get_default_hospitals
from winterflow.data.scenarios import ScenarioConfig, generate_winter_scenario, list_available_scenarios
from winterflow.data.synthetic_history import generate_daily_hospital_demand, generate_patient_arrivals
from winterflow.forecasting.evaluation import evaluate_models
from winterflow.forecasting.features import FORECAST_TARGETS, generate_synthetic_historical_demand
from winterflow.forecasting.models import forecast_next_days, get_feature_importance, train_forecast_models
from winterflow.optimization.actions import ACTION_TYPES
from winterflow.optimization.optimizer import optimize_resource_allocation
from winterflow.optimization.policy import (
    POLICY_PRESETS,
    PolicyControls,
    apply_policy_to_hospitals,
    apply_policy_to_scenario_config,
    build_before_after_comparison,
    build_policy_resource_overrides,
    describe_policy_controls,
    interpret_policy_impact,
    summarize_simulation_outputs,
)
from winterflow.reporting.report import create_report_bundle
from winterflow.simulation.engine import run_simulation
from winterflow.simulation.entities import SimulationConfig
from winterflow.ui.components import (
    command_center_header,
    comparison_metric_cards,
    info_box,
    kpi_cards,
    render_risk_badge,
    styled_impact_table,
)
from winterflow.ui.plots import (
    plot_before_after_timeseries,
    plot_comparison_bars,
    plot_daily_demand,
    plot_feature_importance,
    plot_forecast_interval,
    plot_hospital_network,
    plot_optimizer_scores,
    plot_risk_heatmap,
    plot_scenario_curves,
    plot_simulation_occupancy,
    plot_staff_absence,
    plot_trolley_counts,
)
from winterflow.ui.text import METHODOLOGY_TEXT, POLICY_SANDBOX_TEXT, format_scenario_name
from winterflow.ui.theme import apply_theme, configure_page


POLICY_WIDGET_DEFAULTS = {
    "policy_virus_intensity_multiplier": 1.0,
    "policy_rsv_peak_shift_days": 0,
    "policy_flu_peak_shift_days": 0,
    "policy_covid_peak_shift_days": 0,
    "policy_staff_availability_reduction_pct": 0.0,
    "policy_open_surge_beds": 0,
    "policy_open_surge_icu_beds": 0,
    "policy_temporary_nurses": 0,
    "policy_temporary_doctors": 0,
    "policy_reduce_elective_admissions_pct": 0.0,
    "policy_discharge_acceleration_pct": 0.0,
    "policy_transfer_capacity": 0,
}

TARGET_LABELS = {
    "ed_arrivals": "ED arrivals",
    "admissions": "Admissions",
    "icu_admissions": "ICU admissions",
    "risk_score": "Risk score",
    "trolley_count": "Trolley count",
}

ACTION_LABELS = {
    "do_nothing": "Do nothing",
    "open_surge_general_beds": "Open surge general beds",
    "open_surge_icu_beds": "Open surge ICU beds",
    "add_temporary_nurses": "Add temporary nurses",
    "add_temporary_doctors": "Add temporary doctors",
    "reduce_elective_admissions": "Reduce elective admissions",
    "accelerate_discharge": "Accelerate discharge",
    "create_ed_overflow_spaces": "Create ED overflow spaces",
    "transfer_patients": "Transfer patients",
}


@st.cache_data
def load_hospital_network(seed: int = 42) -> pd.DataFrame:
    hospitals_df = get_default_hospitals(seed=seed)
    hospitals_df["transfer_partners"] = hospitals_df["transfer_partners"].apply(tuple)
    return hospitals_df


@st.cache_data
def load_baseline_scenario(scenario: str, n_days: int = 90) -> pd.DataFrame:
    return generate_winter_scenario(ScenarioConfig(scenario=scenario, n_days=n_days))


@st.cache_data
def load_policy_scenario(scenario: str, controls: PolicyControls, n_days: int = 90) -> pd.DataFrame:
    config = apply_policy_to_scenario_config(ScenarioConfig(scenario=scenario, n_days=n_days), controls)
    return generate_winter_scenario(config)


@st.cache_data
def load_daily_demand(hospitals_df: pd.DataFrame, scenario_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    return generate_daily_hospital_demand(hospitals_df, scenario_df, seed=seed)


@st.cache_data
def load_patient_arrivals(
    hospitals_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    quick_mode: bool,
    seed: int = 42,
) -> pd.DataFrame:
    return generate_patient_arrivals(hospitals_df, scenario_df, seed=seed, quick_mode=quick_mode)


@st.cache_data
def load_forecast_history(hospitals_df: pd.DataFrame, years: int = 3, seed: int = 42) -> pd.DataFrame:
    return generate_synthetic_historical_demand(hospitals_df, years=years, seed=seed)


@st.cache_resource(show_spinner="Training forecasting models...")
def train_cached_forecast_models(history_df: pd.DataFrame, targets: tuple[str, ...]):
    return train_forecast_models(history_df, targets=targets)


@st.cache_data(show_spinner="Evaluating forecasting models...")
def evaluate_cached_forecast_models(history_df: pd.DataFrame, targets: tuple[str, ...]) -> pd.DataFrame:
    return evaluate_models(history_df, targets=targets)


@st.cache_data(show_spinner="Running current SimPy simulation...")
def run_cached_current_simulation(
    hospitals_df: pd.DataFrame,
    patient_arrivals_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    controls: PolicyControls,
    initial_ward_occupancy_pct: float,
    initial_icu_occupancy_pct: float,
):
    config = SimulationConfig(
        initial_ward_occupancy_pct=initial_ward_occupancy_pct,
        initial_icu_occupancy_pct=initial_icu_occupancy_pct,
    )
    resource_overrides = build_policy_resource_overrides(hospitals_df, controls)
    return run_simulation(
        hospitals_df,
        patient_arrivals_df,
        scenario_df,
        config=config,
        resource_overrides=resource_overrides,
    )


@st.cache_data(show_spinner="Running baseline and policy simulations...")
def run_cached_policy_pair(
    hospitals_df: pd.DataFrame,
    baseline_patient_arrivals_df: pd.DataFrame,
    baseline_scenario_df: pd.DataFrame,
    policy_hospitals_df: pd.DataFrame,
    policy_patient_arrivals_df: pd.DataFrame,
    policy_scenario_df: pd.DataFrame,
    controls: PolicyControls,
    initial_ward_occupancy_pct: float,
    initial_icu_occupancy_pct: float,
) -> dict[str, object]:
    config = SimulationConfig(
        initial_ward_occupancy_pct=initial_ward_occupancy_pct,
        initial_icu_occupancy_pct=initial_icu_occupancy_pct,
    )
    baseline_outputs = run_simulation(hospitals_df, baseline_patient_arrivals_df, baseline_scenario_df, config=config)
    baseline_summary = summarize_simulation_outputs(baseline_outputs[0], baseline_outputs[2])
    policy_overrides = build_policy_resource_overrides(hospitals_df, controls, baseline_outputs[2])
    policy_outputs = run_simulation(
        policy_hospitals_df,
        policy_patient_arrivals_df,
        policy_scenario_df,
        config=config,
        resource_overrides=policy_overrides,
    )
    policy_summary = summarize_simulation_outputs(policy_outputs[0], policy_outputs[2])
    comparison_df = build_before_after_comparison(baseline_summary, policy_summary)
    interpretation = interpret_policy_impact(controls, comparison_df, baseline_summary, policy_summary)
    return {
        "baseline_outputs": baseline_outputs,
        "policy_outputs": policy_outputs,
        "baseline_summary": baseline_summary,
        "policy_summary": policy_summary,
        "comparison_df": comparison_df,
        "interpretation": interpretation,
    }


def main() -> None:
    configure_page()
    apply_pending_policy_preset()
    apply_theme()

    selected_scenario, quick_mode, initial_ward_occupancy_pct, initial_icu_occupancy_pct = render_sidebar_controls()
    policy_controls = policy_controls_from_state()

    hospitals_df = load_hospital_network()
    baseline_scenario_df = load_baseline_scenario(selected_scenario)
    policy_scenario_df = load_policy_scenario(selected_scenario, policy_controls)
    policy_hospitals_df = apply_policy_to_hospitals(hospitals_df, policy_controls)
    baseline_daily_demand_df = load_daily_demand(hospitals_df, baseline_scenario_df)
    policy_daily_demand_df = load_daily_demand(policy_hospitals_df, policy_scenario_df)
    baseline_patient_arrivals_df = load_patient_arrivals(hospitals_df, baseline_scenario_df, quick_mode=quick_mode)
    policy_patient_arrivals_df = load_patient_arrivals(policy_hospitals_df, policy_scenario_df, quick_mode=quick_mode)
    active_signature = (
        selected_scenario,
        quick_mode,
        initial_ward_occupancy_pct,
        initial_icu_occupancy_pct,
        policy_controls,
        len(policy_patient_arrivals_df),
    )
    if st.session_state.get("active_signature") != active_signature:
        st.session_state["active_signature"] = active_signature
        st.session_state.pop("current_simulation_outputs", None)
        st.session_state.pop("policy_comparison_outputs", None)

    current_risk = get_current_risk_label()
    command_center_header(
        APP_TITLE,
        APP_SUBTITLE,
        format_scenario_name(selected_scenario),
        len(policy_scenario_df),
        len(policy_patient_arrivals_df),
        current_risk,
    )

    tabs = st.tabs(
        [
            "Command Center",
            "Hospital Network",
            "Policy Sandbox",
            "Forecasting",
            "Optimizer",
            "Report",
            "Methodology",
        ]
    )

    with tabs[0]:
        render_command_center_tab(
            selected_scenario,
            hospitals_df,
            policy_scenario_df,
            policy_daily_demand_df,
            policy_patient_arrivals_df,
            policy_controls,
            initial_ward_occupancy_pct,
            initial_icu_occupancy_pct,
        )

    with tabs[1]:
        render_hospital_network_tab(hospitals_df)

    with tabs[2]:
        render_policy_sandbox_tab(
            hospitals_df,
            baseline_scenario_df,
            policy_scenario_df,
            policy_hospitals_df,
            baseline_patient_arrivals_df,
            policy_patient_arrivals_df,
            policy_controls,
            initial_ward_occupancy_pct,
            initial_icu_occupancy_pct,
        )

    with tabs[3]:
        render_forecasting_tab(hospitals_df)

    with tabs[4]:
        render_optimizer_tab(policy_daily_demand_df, hospitals_df)

    with tabs[5]:
        render_report_tab(
            selected_scenario,
            policy_controls,
            hospitals_df,
            policy_daily_demand_df,
            policy_patient_arrivals_df,
        )

    with tabs[6]:
        render_methodology_tab()


def render_sidebar_controls() -> tuple[str, bool, float, float]:
    st.sidebar.header("Scenario Controls")
    available_scenarios = list_available_scenarios()
    selected_scenario = st.sidebar.selectbox(
        "Scenario selector",
        available_scenarios,
        index=available_scenarios.index("combined_winter_surge"),
        format_func=format_scenario_name,
    )
    quick_mode = st.sidebar.toggle("Quick mode", value=True)
    initial_ward_occupancy_pct = st.sidebar.slider("Starting ward occupancy", 0.50, 0.95, 0.82, 0.01)
    initial_icu_occupancy_pct = st.sidebar.slider("Starting ICU occupancy", 0.40, 0.90, 0.70, 0.01)

    st.sidebar.header("Policy Levers")
    with st.sidebar:
        render_policy_preset_buttons("sidebar")
    st.sidebar.slider("Virus intensity multiplier", 0.70, 1.50, key="policy_virus_intensity_multiplier", step=0.05)
    st.sidebar.slider("RSV peak shift days", -21, 21, key="policy_rsv_peak_shift_days", step=1)
    st.sidebar.slider("Flu peak shift days", -21, 21, key="policy_flu_peak_shift_days", step=1)
    st.sidebar.slider("COVID peak shift days", -21, 21, key="policy_covid_peak_shift_days", step=1)
    st.sidebar.slider(
        "Staff availability reduction",
        0.0,
        40.0,
        key="policy_staff_availability_reduction_pct",
        step=1.0,
    )
    st.sidebar.number_input("Open surge beds", min_value=0, max_value=200, key="policy_open_surge_beds", step=1)
    st.sidebar.number_input("Open surge ICU beds", min_value=0, max_value=80, key="policy_open_surge_icu_beds", step=1)
    st.sidebar.number_input("Add temporary nurses", min_value=0, max_value=300, key="policy_temporary_nurses", step=5)
    st.sidebar.number_input("Add temporary doctors", min_value=0, max_value=120, key="policy_temporary_doctors", step=1)
    st.sidebar.slider(
        "Reduce elective admissions",
        0.0,
        80.0,
        key="policy_reduce_elective_admissions_pct",
        step=5.0,
    )
    st.sidebar.slider(
        "Discharge acceleration",
        -50.0,
        50.0,
        key="policy_discharge_acceleration_pct",
        step=5.0,
    )
    st.sidebar.number_input("Transfer capacity", min_value=0, max_value=100, key="policy_transfer_capacity", step=1)
    return selected_scenario, quick_mode, initial_ward_occupancy_pct, initial_icu_occupancy_pct


def render_command_center_tab(
    selected_scenario: str,
    hospitals_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    daily_demand_df: pd.DataFrame,
    patient_arrivals_df: pd.DataFrame,
    policy_controls: PolicyControls,
    initial_ward_occupancy_pct: float,
    initial_icu_occupancy_pct: float,
) -> None:
    info_box(
        "Synthetic command center",
        "Use the sidebar to shape the winter wave and policy package. Run the simulation to convert demand into occupancy, waits, trolley counts, and risk.",
    )
    kpi_cards(
        [
            {"label": "Hospitals", "value": f"{len(hospitals_df)}"},
            {"label": "Scenario", "value": format_scenario_name(selected_scenario)},
            {"label": "Synthetic arrivals", "value": f"{len(patient_arrivals_df):,}"},
            {"label": "Expected admissions", "value": f"{int(patient_arrivals_df['needs_admission'].sum()):,}"},
            {"label": "Expected ICU", "value": f"{int(patient_arrivals_df['needs_icu'].sum()):,}"},
            {"label": "Peak demand", "value": f"{scenario_df['combined_demand_multiplier'].max():.2f}x"},
            {"label": "Peak absence", "value": f"{scenario_df['staff_absence_rate'].max():.1%}"},
            {"label": "Policy package", "value": describe_policy_controls(policy_controls)},
        ],
        columns=4,
    )

    if st.button("Run Current Simulation", type="primary"):
        outputs = run_cached_current_simulation(
            hospitals_df,
            patient_arrivals_df,
            scenario_df,
            policy_controls,
            initial_ward_occupancy_pct,
            initial_icu_occupancy_pct,
        )
        st.session_state["current_simulation_outputs"] = outputs

    if "current_simulation_outputs" in st.session_state:
        patient_results_df, hourly_metrics_df, daily_metrics_df, event_log_df = st.session_state[
            "current_simulation_outputs"
        ]
        summary = summarize_simulation_outputs(patient_results_df, daily_metrics_df)
        render_risk_badge(str(summary["regional_risk_label"]))
        kpi_cards(
            [
                {"label": "Regional risk score", "value": f"{summary['regional_risk_score']:.1f}"},
                {"label": "Peak trolley count", "value": f"{summary['peak_trolley_count']:.0f}"},
                {"label": "Peak ward occupancy", "value": f"{summary['peak_ward_occupancy']:.0f}%"},
                {"label": "Peak ICU occupancy", "value": f"{summary['peak_icu_occupancy']:.0f}%"},
                {"label": "Mean ED wait", "value": f"{summary['mean_ed_wait']:.1f}h"},
                {"label": "P90 ED wait", "value": f"{summary['p90_ed_wait']:.1f}h"},
                {"label": "Mean staff stress", "value": f"{summary['mean_staff_stress']:.2f}"},
                {"label": "Event log rows", "value": f"{len(event_log_df):,}"},
            ],
            columns=4,
        )
        st.plotly_chart(plot_simulation_occupancy(daily_metrics_df), use_container_width=True, key="current_occupancy_chart")
        st.plotly_chart(plot_trolley_counts(daily_metrics_df), use_container_width=True, key="current_trolley_chart")
        st.plotly_chart(plot_risk_heatmap(daily_metrics_df), use_container_width=True, key="current_risk_heatmap")
        st.dataframe(daily_metrics_df.tail(60), use_container_width=True, hide_index=True)
    else:
        info_box("Simulation not run", "The demand and scenario charts below are ready; run the simulation when you want operational metrics.", "warning")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(plot_scenario_curves(scenario_df), use_container_width=True, key="command_scenario_curves")
    with chart_right:
        st.plotly_chart(plot_staff_absence(scenario_df), use_container_width=True, key="command_staff_absence")
    st.plotly_chart(plot_daily_demand(daily_demand_df), use_container_width=True, key="command_daily_demand")


def render_hospital_network_tab(hospitals_df: pd.DataFrame) -> None:
    info_box(
        "Synthetic hospital network",
        "Locations, capacities, and transfer partners are fictional but shaped to resemble a small Irish-style regional network.",
    )
    st.plotly_chart(plot_hospital_network(hospitals_df), use_container_width=True, key="hospital_network_chart")
    st.dataframe(hospitals_df, use_container_width=True, hide_index=True)


def render_policy_sandbox_tab(
    hospitals_df: pd.DataFrame,
    baseline_scenario_df: pd.DataFrame,
    policy_scenario_df: pd.DataFrame,
    policy_hospitals_df: pd.DataFrame,
    baseline_patient_arrivals_df: pd.DataFrame,
    policy_patient_arrivals_df: pd.DataFrame,
    policy_controls: PolicyControls,
    initial_ward_occupancy_pct: float,
    initial_icu_occupancy_pct: float,
) -> None:
    info_box("Policy Sandbox", POLICY_SANDBOX_TEXT)
    st.subheader("Scenario Playbooks")
    render_policy_preset_buttons("sandbox")

    st.subheader("Active Policy Package")
    st.dataframe(policy_controls_table(policy_controls), use_container_width=True, hide_index=True)

    before_1, before_2, before_3 = st.columns(3)
    before_1.metric("Baseline arrivals", f"{len(baseline_patient_arrivals_df):,}")
    before_2.metric("Policy arrivals", f"{len(policy_patient_arrivals_df):,}")
    before_3.metric("Policy", describe_policy_controls(policy_controls))

    if st.button("Apply Policy And Compare", type="primary"):
        st.session_state["policy_comparison_outputs"] = run_cached_policy_pair(
            hospitals_df,
            baseline_patient_arrivals_df,
            baseline_scenario_df,
            policy_hospitals_df,
            policy_patient_arrivals_df,
            policy_scenario_df,
            policy_controls,
            initial_ward_occupancy_pct,
            initial_icu_occupancy_pct,
        )

    if "policy_comparison_outputs" not in st.session_state:
        info_box("Awaiting comparison", "Apply a policy to run the baseline and policy simulations side by side.", "warning")
        st.plotly_chart(plot_scenario_curves(policy_scenario_df), use_container_width=True, key="policy_pending_scenario_curves")
        return

    outputs = st.session_state["policy_comparison_outputs"]
    comparison_df = outputs["comparison_df"]
    baseline_outputs = outputs["baseline_outputs"]
    policy_outputs = outputs["policy_outputs"]
    baseline_summary = outputs["baseline_summary"]
    policy_summary = outputs["policy_summary"]

    info_box("Plain-English Interpretation", str(outputs["interpretation"]), "success")
    risk_cols = st.columns(2)
    with risk_cols[0]:
        st.write("Baseline risk")
        render_risk_badge(str(baseline_summary["regional_risk_label"]))
    with risk_cols[1]:
        st.write("Policy risk")
        render_risk_badge(str(policy_summary["regional_risk_label"]))

    comparison_metric_cards(comparison_df)
    st.plotly_chart(
        plot_before_after_timeseries(baseline_outputs[2], policy_outputs[2], "max_trolley_count"),
        use_container_width=True,
        key="policy_trolley_before_after",
    )
    st.plotly_chart(
        plot_before_after_timeseries(baseline_outputs[2], policy_outputs[2], "risk_score"),
        use_container_width=True,
        key="policy_risk_before_after",
    )
    st.plotly_chart(plot_comparison_bars(comparison_df), use_container_width=True, key="policy_comparison_bars")
    styled_impact_table(comparison_df)


def render_forecasting_tab(hospitals_df: pd.DataFrame) -> None:
    info_box(
        "AI demand forecasting",
        "Models train on 2-3 years of synthetic hospital-day history. Forecast intervals are generated with quantile gradient boosting and compared with a rolling-average baseline.",
    )
    history_df = load_forecast_history(hospitals_df, years=3)
    target = st.selectbox(
        "Forecast target",
        list(FORECAST_TARGETS),
        format_func=lambda value: TARGET_LABELS[value],
    )
    hospital_options = history_df[["hospital_id", "hospital_name"]].drop_duplicates().sort_values("hospital_id")
    hospital_label_lookup = {
        f"{row.hospital_id} - {row.hospital_name}": row.hospital_id for row in hospital_options.itertuples()
    }
    selected_hospital_label = st.selectbox("Hospital", list(hospital_label_lookup))
    selected_hospital_id = hospital_label_lookup[selected_hospital_label]
    horizon = st.selectbox("Forecast horizon", [7, 14], index=1)

    models = train_cached_forecast_models(history_df, targets=(target,))
    selected_history = history_df.loc[history_df["hospital_id"] == selected_hospital_id].tail(180)
    forecast_df = forecast_next_days(models, selected_history, horizon=horizon)
    forecast_target_df = forecast_df.loc[forecast_df["target"] == target]
    metrics_df = evaluate_cached_forecast_models(history_df, targets=(target,))
    importance_df = get_feature_importance(models, target)

    median_peak = forecast_target_df["p50"].max()
    p90_peak = forecast_target_df["p90"].max()
    highest_risk_date = forecast_target_df.loc[forecast_target_df["p90"].idxmax(), "date"]
    kpi_cards(
        [
            {"label": "Hospital", "value": selected_hospital_id},
            {"label": "Target", "value": TARGET_LABELS[target]},
            {"label": "Median peak", "value": f"{median_peak:,.0f}"},
            {"label": "P90 peak", "value": f"{p90_peak:,.0f}"},
            {"label": "Highest-risk date", "value": pd.Timestamp(highest_risk_date).strftime("%Y-%m-%d")},
            {"label": "Training rows", "value": f"{len(history_df):,}"},
        ],
        columns=3,
    )
    st.plotly_chart(plot_forecast_interval(forecast_target_df, TARGET_LABELS[target]), use_container_width=True, key="forecast_interval_chart")
    left, right = st.columns(2)
    with left:
        st.subheader("Model evaluation")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Feature importance")
        st.plotly_chart(plot_feature_importance(importance_df), use_container_width=True, key="forecast_feature_importance")
    st.dataframe(forecast_target_df, use_container_width=True, hide_index=True)


def render_optimizer_tab(daily_demand_df: pd.DataFrame, hospitals_df: pd.DataFrame) -> None:
    info_box(
        "Resource allocation optimizer",
        "Candidate intervention packages are screened with fast pressure approximations and ranked by projected risk reduction per cost point.",
    )
    budget = st.slider("Optimization budget", min_value=0, max_value=180, value=80, step=5)
    max_actions = st.slider("Maximum actions per package", min_value=1, max_value=4, value=3, step=1)
    allowed_labels = st.multiselect(
        "Action constraints",
        [ACTION_LABELS[action_type] for action_type in ACTION_TYPES if action_type != "do_nothing"],
        default=[ACTION_LABELS[action_type] for action_type in ACTION_TYPES if action_type != "do_nothing"],
    )
    reverse_action_lookup = {label: action_type for action_type, label in ACTION_LABELS.items()}
    allowed_action_types = ["do_nothing"] + [reverse_action_lookup[label] for label in allowed_labels]
    result = optimize_resource_allocation(
        daily_demand_df,
        hospitals_df,
        budget=budget,
        allowed_action_types=allowed_action_types,
        max_actions=max_actions,
    )
    recommendations_df = result["recommendations"]

    kpi_cards(
        [
            {"label": "Baseline score", "value": f"{result['baseline_score']:.1f}"},
            {"label": "Projected score", "value": f"{result['projected_score']:.1f}"},
            {"label": "Risk reduction", "value": f"{result['risk_reduction']:.1f}"},
            {"label": "Cost points", "value": f"{result['estimated_cost_points']:.0f} / {budget}"},
        ],
        columns=4,
    )
    info_box("Recommended plan rationale", str(result["rationale_text"]), "success")
    st.plotly_chart(plot_optimizer_scores(recommendations_df), use_container_width=True, key="optimizer_scores_chart")
    st.subheader("Recommended intervention plan")
    st.dataframe(recommendations_df, use_container_width=True, hide_index=True)


def render_report_tab(
    selected_scenario: str,
    policy_controls: PolicyControls,
    hospitals_df: pd.DataFrame,
    daily_demand_df: pd.DataFrame,
    patient_arrivals_df: pd.DataFrame,
) -> None:
    info_box(
        "Executive report",
        "Preview and download a synthetic executive report plus CSV extracts for metrics, optimizer recommendations, and patient-level samples.",
    )
    optimizer_result = optimize_resource_allocation(daily_demand_df, hospitals_df, budget=80)
    optimizer_recommendations = optimizer_result["recommendations"]
    daily_metrics_df, regional_kpis, before_after_df = current_report_metrics(daily_demand_df)
    highest_risk_df = highest_risk_hospitals_for_report(daily_metrics_df, daily_demand_df, hospitals_df)
    forecast_summary = forecast_summary_for_report(daily_demand_df)
    policy_table = policy_controls_table(policy_controls)

    bundle = create_report_bundle(
        scenario_name=format_scenario_name(selected_scenario),
        policy_assumptions=policy_table,
        regional_kpis=regional_kpis,
        highest_risk_hospitals=highest_risk_df,
        forecast_summary=forecast_summary,
        optimizer_recommendations=optimizer_recommendations,
        before_after_comparison=before_after_df,
        daily_metrics=daily_metrics_df,
        patient_sample=patient_arrivals_df.head(1000),
    )

    st.subheader("Preview executive report")
    st.markdown(bundle["markdown"])

    download_1, download_2, download_3, download_4 = st.columns(4)
    with download_1:
        st.download_button(
            "Download Markdown Report",
            data=bundle["markdown"],
            file_name="winter_flow_executive_report.md",
            mime="text/markdown",
        )
    with download_2:
        st.download_button(
            "Download Daily Metrics CSV",
            data=bundle["daily_metrics_csv"],
            file_name="winter_flow_daily_metrics.csv",
            mime="text/csv",
        )
    with download_3:
        st.download_button(
            "Download Optimizer Plan CSV",
            data=bundle["optimizer_plan_csv"],
            file_name="winter_flow_optimizer_plan.csv",
            mime="text/csv",
        )
    with download_4:
        st.download_button(
            "Download Patient Sample CSV",
            data=bundle["patient_sample_csv"],
            file_name="winter_flow_patient_sample.csv",
            mime="text/csv",
        )


def render_methodology_tab() -> None:
    st.markdown(METHODOLOGY_TEXT)
    info_box("Disclaimer", SYNTHETIC_DATA_DISCLAIMER, "warning")


def current_report_metrics(daily_demand_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame | None]:
    if "policy_comparison_outputs" in st.session_state:
        outputs = st.session_state["policy_comparison_outputs"]
        policy_outputs = outputs["policy_outputs"]
        policy_summary = outputs["policy_summary"]
        return policy_outputs[2], _format_summary_for_report(policy_summary), outputs["comparison_df"]
    if "current_simulation_outputs" in st.session_state:
        outputs = st.session_state["current_simulation_outputs"]
        summary = summarize_simulation_outputs(outputs[0], outputs[2])
        return outputs[2], _format_summary_for_report(summary), None

    preview_metrics = (
        daily_demand_df.groupby(["date", "day"], as_index=False)
        .agg(
            ed_arrivals=("synthetic_arrivals", "sum"),
            admissions=("expected_admissions", "sum"),
            icu_admissions=("expected_icu_admissions", "sum"),
            peak_demand_multiplier=("combined_demand_multiplier", "max"),
            mean_staff_absence_rate=("staff_absence_rate", "mean"),
        )
        .round(3)
    )
    regional_kpis = {
        "simulation_status": "Preview only - run simulation for operational KPIs",
        "synthetic_arrivals": f"{int(daily_demand_df['synthetic_arrivals'].sum()):,}",
        "expected_admissions": f"{daily_demand_df['expected_admissions'].sum():,.0f}",
        "expected_icu_admissions": f"{daily_demand_df['expected_icu_admissions'].sum():,.0f}",
        "peak_daily_arrivals": f"{int(preview_metrics['ed_arrivals'].max()):,}",
    }
    return preview_metrics, regional_kpis, None


def highest_risk_hospitals_for_report(
    daily_metrics_df: pd.DataFrame,
    daily_demand_df: pd.DataFrame,
    hospitals_df: pd.DataFrame,
) -> pd.DataFrame:
    if "risk_score" in daily_metrics_df.columns and "hospital_id" in daily_metrics_df.columns:
        ranked = (
            daily_metrics_df.groupby("hospital_id", as_index=False)
            .agg(
                peak_risk_score=("risk_score", "max"),
                peak_trolley_count=("max_trolley_count", "max"),
                peak_ward_occupancy=("max_ward_occupancy_pct", "max"),
                peak_icu_occupancy=("max_icu_occupancy_pct", "max"),
            )
            .merge(hospitals_df[["hospital_id", "name"]], on="hospital_id", how="left")
            .sort_values("peak_risk_score", ascending=False)
        )
        return ranked[["hospital_id", "name", "peak_risk_score", "peak_trolley_count", "peak_ward_occupancy", "peak_icu_occupancy"]]

    ranked = (
        daily_demand_df.groupby("hospital_id", as_index=False)
        .agg(
            total_arrivals=("synthetic_arrivals", "sum"),
            peak_daily_arrivals=("synthetic_arrivals", "max"),
            expected_admissions=("expected_admissions", "sum"),
            expected_icu_admissions=("expected_icu_admissions", "sum"),
        )
        .merge(hospitals_df[["hospital_id", "name"]], on="hospital_id", how="left")
        .sort_values("expected_admissions", ascending=False)
        .round(1)
    )
    return ranked[["hospital_id", "name", "total_arrivals", "peak_daily_arrivals", "expected_admissions", "expected_icu_admissions"]]


def forecast_summary_for_report(daily_demand_df: pd.DataFrame) -> dict[str, object]:
    daily = (
        daily_demand_df.groupby("date", as_index=False)
        .agg(
            synthetic_arrivals=("synthetic_arrivals", "sum"),
            expected_admissions=("expected_admissions", "sum"),
            expected_icu_admissions=("expected_icu_admissions", "sum"),
        )
        .sort_values("date")
        .head(14)
    )
    peak_row = daily.loc[daily["synthetic_arrivals"].idxmax()]
    return {
        "14_day_synthetic_arrivals": f"{int(daily['synthetic_arrivals'].sum()):,}",
        "14_day_expected_admissions": f"{daily['expected_admissions'].sum():,.0f}",
        "14_day_expected_icu_admissions": f"{daily['expected_icu_admissions'].sum():,.0f}",
        "median_daily_arrivals": f"{daily['synthetic_arrivals'].median():,.0f}",
        "peak_forecast_date": pd.Timestamp(peak_row["date"]).strftime("%Y-%m-%d"),
        "peak_forecast_arrivals": f"{int(peak_row['synthetic_arrivals']):,}",
    }


def _format_summary_for_report(summary: dict[str, object]) -> dict[str, object]:
    return {
        "regional_risk_label": summary.get("regional_risk_label", "Unavailable"),
        "regional_risk_score": f"{float(summary.get('regional_risk_score', 0)):.1f}",
        "mean_ed_wait_hours": f"{float(summary.get('mean_ed_wait', 0)):.1f}",
        "p90_ed_wait_hours": f"{float(summary.get('p90_ed_wait', 0)):.1f}",
        "peak_trolley_count": f"{float(summary.get('peak_trolley_count', 0)):.0f}",
        "peak_ward_occupancy": f"{float(summary.get('peak_ward_occupancy', 0)):.0f}%",
        "peak_icu_occupancy": f"{float(summary.get('peak_icu_occupancy', 0)):.0f}%",
        "mean_staff_stress": f"{float(summary.get('mean_staff_stress', 0)):.2f}",
    }


def render_policy_preset_buttons(location: str) -> None:
    for preset_name in POLICY_PRESETS:
        if st.button(preset_name, key=f"{location}_{preset_name}"):
            st.session_state["pending_policy_preset"] = preset_name
            st.rerun()


def apply_pending_policy_preset() -> None:
    for key, value in POLICY_WIDGET_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    preset_name = st.session_state.pop("pending_policy_preset", None)
    if preset_name:
        controls = POLICY_PRESETS[preset_name]
        set_policy_controls_in_state(controls)


def set_policy_controls_in_state(controls: PolicyControls) -> None:
    st.session_state["policy_virus_intensity_multiplier"] = controls.virus_intensity_multiplier
    st.session_state["policy_rsv_peak_shift_days"] = controls.rsv_peak_shift_days
    st.session_state["policy_flu_peak_shift_days"] = controls.flu_peak_shift_days
    st.session_state["policy_covid_peak_shift_days"] = controls.covid_peak_shift_days
    st.session_state["policy_staff_availability_reduction_pct"] = controls.staff_availability_reduction_pct
    st.session_state["policy_open_surge_beds"] = controls.open_surge_beds
    st.session_state["policy_open_surge_icu_beds"] = controls.open_surge_icu_beds
    st.session_state["policy_temporary_nurses"] = controls.temporary_nurses
    st.session_state["policy_temporary_doctors"] = controls.temporary_doctors
    st.session_state["policy_reduce_elective_admissions_pct"] = controls.reduce_elective_admissions_pct
    st.session_state["policy_discharge_acceleration_pct"] = controls.discharge_acceleration_pct
    st.session_state["policy_transfer_capacity"] = controls.transfer_capacity
    st.session_state.pop("current_simulation_outputs", None)
    st.session_state.pop("policy_comparison_outputs", None)


def policy_controls_from_state() -> PolicyControls:
    return PolicyControls(
        virus_intensity_multiplier=float(st.session_state["policy_virus_intensity_multiplier"]),
        rsv_peak_shift_days=int(st.session_state["policy_rsv_peak_shift_days"]),
        flu_peak_shift_days=int(st.session_state["policy_flu_peak_shift_days"]),
        covid_peak_shift_days=int(st.session_state["policy_covid_peak_shift_days"]),
        staff_availability_reduction_pct=float(st.session_state["policy_staff_availability_reduction_pct"]),
        open_surge_beds=int(st.session_state["policy_open_surge_beds"]),
        open_surge_icu_beds=int(st.session_state["policy_open_surge_icu_beds"]),
        temporary_nurses=int(st.session_state["policy_temporary_nurses"]),
        temporary_doctors=int(st.session_state["policy_temporary_doctors"]),
        reduce_elective_admissions_pct=float(st.session_state["policy_reduce_elective_admissions_pct"]),
        discharge_acceleration_pct=float(st.session_state["policy_discharge_acceleration_pct"]),
        transfer_capacity=int(st.session_state["policy_transfer_capacity"]),
    )


def policy_controls_table(controls: PolicyControls) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"lever": "Virus intensity multiplier", "value": controls.virus_intensity_multiplier},
            {"lever": "RSV peak shift days", "value": controls.rsv_peak_shift_days},
            {"lever": "Flu peak shift days", "value": controls.flu_peak_shift_days},
            {"lever": "COVID peak shift days", "value": controls.covid_peak_shift_days},
            {"lever": "Staff availability reduction", "value": f"{controls.staff_availability_reduction_pct:.0f}%"},
            {"lever": "Open surge beds", "value": controls.open_surge_beds},
            {"lever": "Open surge ICU beds", "value": controls.open_surge_icu_beds},
            {"lever": "Add temporary nurses", "value": controls.temporary_nurses},
            {"lever": "Add temporary doctors", "value": controls.temporary_doctors},
            {"lever": "Reduce elective admissions", "value": f"{controls.reduce_elective_admissions_pct:.0f}%"},
            {"lever": "Discharge acceleration", "value": f"{controls.discharge_acceleration_pct:.0f}%"},
            {"lever": "Transfer capacity", "value": controls.transfer_capacity},
        ]
    )


def get_current_risk_label() -> str | None:
    if "policy_comparison_outputs" in st.session_state:
        return str(st.session_state["policy_comparison_outputs"]["policy_summary"]["regional_risk_label"])
    if "current_simulation_outputs" in st.session_state:
        outputs = st.session_state["current_simulation_outputs"]
        return str(summarize_simulation_outputs(outputs[0], outputs[2])["regional_risk_label"])
    return None


if __name__ == "__main__":
    main()
