from __future__ import annotations

import streamlit as st

from winterflow.constants import APP_TITLE


RAG_COLORS = {
    "Green": "#16A34A",
    "Amber": "#D97706",
    "Red": "#DC2626",
}

CHART_COLORS = {
    "navy": "#0F172A",
    "steel": "#2563EB",
    "teal": "#0F766E",
    "amber": "#D97706",
    "red": "#DC2626",
    "green": "#16A34A",
    "slate": "#64748B",
    "cyan": "#0891B2",
}


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="WF",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --wf-bg: #F6F8FB;
            --wf-ink: #0F172A;
            --wf-muted: #64748B;
            --wf-border: #D8E1EA;
            --wf-panel: #FFFFFF;
            --wf-header: #101827;
        }
        .stApp {
            background: var(--wf-bg);
            color: var(--wf-ink);
        }
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--wf-border);
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--wf-border);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetric"] label {
            color: var(--wf-muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .wf-header {
            background: linear-gradient(135deg, #101827 0%, #17324D 55%, #0F766E 100%);
            color: #FFFFFF;
            padding: 22px 26px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.10);
            margin-bottom: 16px;
        }
        .wf-header h1 {
            margin: 0;
            font-size: 2.05rem;
            letter-spacing: 0;
        }
        .wf-header p {
            margin: 6px 0 0 0;
            color: #D6E4F0;
            font-size: 1.02rem;
        }
        .wf-header-meta {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 14px;
        }
        .wf-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.18);
            color: #FFFFFF;
            font-size: 0.82rem;
        }
        .wf-card {
            background: #FFFFFF;
            border: 1px solid var(--wf-border);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        .wf-card-label {
            color: var(--wf-muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 6px;
        }
        .wf-card-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--wf-ink);
            line-height: 1.2;
        }
        .wf-card-help {
            color: var(--wf-muted);
            font-size: 0.84rem;
            margin-top: 5px;
        }
        .wf-badge {
            display: inline-flex;
            padding: 5px 9px;
            border-radius: 999px;
            color: #FFFFFF;
            font-weight: 700;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .wf-info {
            border-left: 4px solid #2563EB;
            background: #EFF6FF;
            color: #1E3A8A;
            padding: 12px 14px;
            border-radius: 8px;
            margin: 8px 0 14px 0;
        }
        .wf-warning {
            border-left: 4px solid #D97706;
            background: #FFF7ED;
            color: #7C2D12;
            padding: 12px 14px;
            border-radius: 8px;
            margin: 8px 0 14px 0;
        }
        .wf-success {
            border-left: 4px solid #16A34A;
            background: #F0FDF4;
            color: #14532D;
            padding: 12px 14px;
            border-radius: 8px;
            margin: 8px 0 14px 0;
        }
        .block-container {
            padding-top: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

