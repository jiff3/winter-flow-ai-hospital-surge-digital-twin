from __future__ import annotations

from collections.abc import Iterable
from html import escape

import pandas as pd
import streamlit as st

from winterflow.ui.theme import RAG_COLORS


def command_center_header(
    title: str,
    subtitle: str,
    scenario_label: str,
    horizon_days: int,
    synthetic_arrivals: int,
    risk: str | None = None,
) -> None:
    risk_markup = risk_badge(risk) if risk else ""
    header_html = (
        '<div class="wf-header">'
        f"<h1>{escape(title)}</h1>"
        f"<p>{escape(subtitle)}</p>"
        '<div class="wf-header-meta">'
        f'<span class="wf-pill">Scenario: {escape(scenario_label)}</span>'
        f'<span class="wf-pill">Horizon: {horizon_days} days</span>'
        f'<span class="wf-pill">Synthetic arrivals: {synthetic_arrivals:,}</span>'
        f"{risk_markup}"
        "</div>"
        "</div>"
    )
    st.html(header_html)


def kpi_cards(items: Iterable[dict[str, object]], columns: int = 4) -> None:
    item_list = list(items)
    for start in range(0, len(item_list), columns):
        cols = st.columns(columns)
        for column, item in zip(cols, item_list[start : start + columns]):
            with column:
                delta_html = ""
                if item.get("delta"):
                    delta_html = f"<div class='wf-card-help'>{item['delta']}</div>"
                st.markdown(
                    f"""
                    <div class="wf-card">
                        <div class="wf-card-label">{item.get("label", "")}</div>
                        <div class="wf-card-value">{item.get("value", "")}</div>
                        {delta_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def risk_badge(label: str | None) -> str:
    if not label:
        return ""
    normalized = str(label).title()
    color = RAG_COLORS.get(normalized, "#64748B")
    return f"<span class='wf-badge' style='background:{color}'>{normalized}</span>"


def render_risk_badge(label: str | None) -> None:
    st.markdown(risk_badge(label), unsafe_allow_html=True)


def info_box(title: str, body: str, level: str = "info") -> None:
    css_class = {
        "info": "wf-info",
        "warning": "wf-warning",
        "success": "wf-success",
    }.get(level, "wf-info")
    st.markdown(
        f"<div class='{css_class}'><strong>{title}</strong><br>{body}</div>",
        unsafe_allow_html=True,
    )


def comparison_metric_cards(comparison_df: pd.DataFrame) -> None:
    cards = []
    for row in comparison_df.to_dict("records"):
        change = float(row["percent_change"])
        prefix = "+" if change > 0 else ""
        cards.append(
            {
                "label": row["metric"],
                "value": f"{row['policy']}",
                "delta": f"{prefix}{change:.1f}% vs baseline",
            }
        )
    kpi_cards(cards, columns=4)


def styled_impact_table(comparison_df: pd.DataFrame) -> None:
    display_df = comparison_df.copy()
    display_df["percent_change"] = display_df["percent_change"].map(lambda value: f"{value:+.1f}%")
    display_df["absolute_change"] = display_df["absolute_change"].map(lambda value: f"{value:+.2f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
