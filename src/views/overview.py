"""
Vue d'ensemble — KPIs, distribution de l'attrition, statistiques globales.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def show_overview(df: pd.DataFrame, df_processed: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-header"><h2>📈 Vue d\'ensemble des données</h2></div>',
        unsafe_allow_html=True,
    )

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total = len(df)
    departed = (df["Attrition"] == "Yes").sum()
    rate = departed / total * 100
    avg_sat = df["JobSatisfaction"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Employés", f"{total:,}")
    col2.metric("🚪 Départs", f"{departed}", delta=f"-{rate:.1f}%", delta_color="inverse")
    col3.metric("📊 Taux d'Attrition", f"{rate:.1f}%")
    col4.metric("😊 Satisfaction Moyenne", f"{avg_sat:.2f}/4")

    st.markdown("---")

    # ── Charts row ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        counts = df["Attrition"].value_counts()
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Restés", "Partis"],
                    values=[counts.get("No", 0), counts.get("Yes", 0)],
                    hole=0.6,
                    marker_colors=["#2a9d8f", "#e63946"],
                    textinfo="percent+value",
                    textfont_size=14,
                )
            ]
        )
        fig.update_layout(
            title="<b>Distribution de l'Attrition</b>",
            title_font_size=18,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            height=400,
        )
        fig.add_annotation(
            text=f"<b>{rate:.1f}%</b><br>Taux d'attrition",
            x=0.5,
            y=0.5,
            font_size=16,
            showarrow=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dept_attr = (
            df.groupby("Department")["Attrition"]
            .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
            .reset_index(name="Attrition_Rate")
        )
        fig = px.bar(
            dept_attr,
            x="Department",
            y="Attrition_Rate",
            color="Attrition_Rate",
            color_continuous_scale=["#2a9d8f", "#f4a261", "#e63946"],
            title="<b>Taux d'Attrition par Département</b>",
        )
        fig.update_layout(
            xaxis_title="Département",
            yaxis_title="Taux d'Attrition (%)",
            coloraxis_showscale=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Stats boxes ────────────────────────────────────────────────────────────
    st.markdown("### 📊 Statistiques Clés")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="insight-box">
                <h4>💰 Rémunération</h4>
                <p><b>Salaire moyen :</b> ${df['MonthlyIncome'].mean():,.0f}</p>
                <p><b>Médiane :</b> ${df['MonthlyIncome'].median():,.0f}</p>
                <p><b>Écart-type :</b> ${df['MonthlyIncome'].std():,.0f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="insight-box">
                <h4>📅 Ancienneté</h4>
                <p><b>Moyenne :</b> {df['YearsAtCompany'].mean():.1f} ans</p>
                <p><b>Médiane :</b> {df['YearsAtCompany'].median():.1f} ans</p>
                <p><b>Maximum :</b> {df['YearsAtCompany'].max():.0f} ans</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        ot_rate = (df["OverTime"] == "Yes").mean() * 100
        st.markdown(
            f"""
            <div class="insight-box">
                <h4>⏰ Heures Supplémentaires</h4>
                <p><b>Taux d'overtime :</b> {ot_rate:.1f}%</p>
                <p><b>Distance moyenne :</b> {df['DistanceFromHome'].mean():.1f} km</p>
                <p><b>Équilibre vie/travail :</b> {df['WorkLifeBalance'].mean():.2f}/4</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
