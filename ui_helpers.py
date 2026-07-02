"""
Reusable UI components for the chatbot dashboard.

Eliminates repeated inline HTML across app.py — one function call
renders a styled card, badge, or test-result block.

Usage:
    from ui_helpers import info_card, result_card, section_header, meta_badge
"""

import streamlit as st
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Generic cards
# ---------------------------------------------------------------------------

def info_card(title: str, body: str, *, variant: str = "info") -> None:
    """
    Render a styled information card.

    Variants:
        - "info"    : cyan left border (default)
        - "warning" : amber left border
        - "tip"     : purple background tint
    """
    border_colors = {"info": "#06B6D4", "warning": "#F59E0B"}
    bg_colors = {"info": "rgba(6,182,212,0.05)", "warning": "rgba(245,158,11,0.05)", "tip": "rgba(124,58,237,0.05)"}

    border = border_colors.get(variant, "transparent")
    bg = bg_colors.get(variant, bg_colors["tip"])

    border_css = f"border-left: 4px solid {border};" if variant in border_colors else ""

    st.markdown(
        f"""
        <div style="background: {bg}; padding: 1.25rem; border-radius: 12px;
                    {border_css} margin: 1.5rem 0;">
            <div style="font-weight: 600; color: #1E1B4B; margin-bottom: 0.5rem;">{title}</div>
            <div style="color: #6B7280; font-size: 0.875rem; line-height: 1.7;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(passed: int, total: int, label: str) -> None:
    """Render a centered pass/total result badge with a green gradient."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(6,182,212,0.1) 100%);
                    padding: 1.25rem; border-radius: 12px; margin: 1.5rem 0; text-align: center;">
            <div style="font-weight: 700; color: #10B981; font-size: 1.5rem;">{passed}/{total}</div>
            <div style="color: #6B7280; font-size: 0.875rem; margin-top: 0.25rem;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def success_banner(text: str) -> None:
    """Render a centered success message (e.g. 'Model terpilih')."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(6,182,212,0.1) 100%);
                    padding: 1.25rem; border-radius: 12px; margin: 1.5rem 0; text-align: center;">
            <div style="font-weight: 700; color: #10B981; font-size: 1.125rem;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def meta_badge(text: str) -> None:
    """Render a small monospace badge for debug metadata (intent, confidence, method)."""
    st.markdown(
        f"""
        <div style="margin-top: 0.5rem; padding: 0.5rem 0.75rem; background: rgba(124,58,237,0.05);
                    border-radius: 8px; font-size: 0.75rem; color: #6B7280; font-family: monospace;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def subtitle(text: str) -> None:
    """Render a muted subtitle under a section heading."""
    st.markdown(
        f'<div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 1rem;">{text}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def section_header(emoji: str, title: str) -> None:
    """Render a section heading with an emoji prefix."""
    st.markdown(f"## {emoji} {title}")


def divider() -> None:
    """Render a themed horizontal divider."""
    st.markdown("---")


# ---------------------------------------------------------------------------
# Test-section shorthand
# ---------------------------------------------------------------------------

def render_test_section(emoji: str, title: str, description: str, df, conf_col: str = "confidence", key: str = None) -> int:
    """
    Render a complete test section: header + description card + dataframe + visual gauge.

    Returns the number of PASS cases.
    """
    section_header(emoji, title)
    info_card("🎯 Tujuan Uji", description, variant="tip")

    passed = int((df["status"] == "PASS").sum())
    total = len(df)
    pass_rate = passed / total if total > 0 else 0

    # Visual layout: dataframe + donut gauge
    df_col, gauge_col = st.columns([3, 1])
    with df_col:
        fmt = {conf_col: "{:.2f}"} if conf_col in df.columns else {}
        st.dataframe(df.style.format(fmt), use_container_width=True)
    with gauge_col:
        # Donut gauge
        color = "#10B981" if pass_rate >= 0.8 else ("#F59E0B" if pass_rate >= 0.6 else "#F87171")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pass_rate * 100,
            number=dict(suffix="%", font=dict(size=28, color="#1E1B4B")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#6B7280", tickfont=dict(size=9)),
                bar=dict(color=color, thickness=0.6),
                bgcolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[0, 60], color="rgba(248,113,113,0.15)"),
                    dict(range=[60, 80], color="rgba(251,191,36,0.15)"),
                    dict(range=[80, 100], color="rgba(16,185,129,0.15)"),
                ],
                threshold=dict(line=dict(color="white", width=3), thickness=0.8, value=pass_rate * 100),
            ),
            title=dict(text=f"{passed}/{total} Lolos", font=dict(size=14, color="#6B7280")),
            domain=dict(x=[0, 1], y=[0, 1]),
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"gauge_{key or title}")

    return passed
