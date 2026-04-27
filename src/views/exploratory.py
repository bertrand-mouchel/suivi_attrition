"""
Analyse exploratoire — facteurs de risque, distributions, corrélations.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def show_exploratory_analysis(df: pd.DataFrame, df_processed: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-header"><h2>🔍 Analyse Exploratoire</h2></div>',
        unsafe_allow_html=True,
    )

    analysis_type = st.selectbox(
        "Choisissez le type d'analyse :",
        [
            "Facteurs de Risque Principaux",
            "Analyse par Variables Continues",
            "Analyse par Variables Catégorielles",
            "Corrélations",
        ],
    )

    if analysis_type == "Facteurs de Risque Principaux":
        _risk_factors(df)
    elif analysis_type == "Analyse par Variables Continues":
        _continuous_vars(df)
    elif analysis_type == "Analyse par Variables Catégorielles":
        _categorical_vars(df)
    else:
        _correlations(df, df_processed)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _risk_factors(df: pd.DataFrame) -> None:
    st.markdown("### 🎯 Facteurs de Risque les Plus Impactants")

    col1, col2 = st.columns(2)

    with col1:
        ot_attr = df.groupby(["OverTime", "Attrition"]).size().unstack(fill_value=0)
        ot_pct = ot_attr.div(ot_attr.sum(axis=1), axis=0) * 100

        fig = go.Figure()
        fig.add_trace(
            go.Bar(name="Restés", x=ot_pct.index, y=ot_pct["No"], marker_color="#2a9d8f")
        )
        fig.add_trace(
            go.Bar(name="Partis", x=ot_pct.index, y=ot_pct["Yes"], marker_color="#e63946")
        )
        fig.update_layout(
            title="<b>Impact des Heures Supplémentaires</b>",
            barmode="stack",
            yaxis_title="Pourcentage (%)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        ot_yes = df[df["OverTime"] == "Yes"]["Attrition"].value_counts(normalize=True).get("Yes", 0) * 100
        ot_no  = df[df["OverTime"] == "No"]["Attrition"].value_counts(normalize=True).get("Yes", 0) * 100
        st.warning(
            f"⚠️ Avec heures sup. : **{ot_yes:.1f}%** vs sans : **{ot_no:.1f}%** d'attrition."
        )

    with col2:
        sat_attr = (
            df.groupby("JobSatisfaction")["Attrition"]
            .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
            .reset_index(name="Attrition_Rate")
        )
        fig = px.bar(
            sat_attr,
            x="JobSatisfaction",
            y="Attrition_Rate",
            color="Attrition_Rate",
            color_continuous_scale=["#2a9d8f", "#f4a261", "#e63946"],
            title="<b>Impact de la Satisfaction au Travail</b>",
        )
        fig.update_layout(
            xaxis_title="Niveau de Satisfaction (1-4)",
            yaxis_title="Taux d'Attrition (%)",
            coloraxis_showscale=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        low = sat_attr.loc[sat_attr["JobSatisfaction"] == 1, "Attrition_Rate"].values[0]
        high = sat_attr.loc[sat_attr["JobSatisfaction"] == 4, "Attrition_Rate"].values[0]
        st.info(f"📊 Satisfaction 1 : **{low:.1f}%** vs Satisfaction 4 : **{high:.1f}%**")

    st.markdown("### 💰 Impact du Salaire")
    df = df.copy()
    df["Income_Bracket"] = pd.cut(
        df["MonthlyIncome"],
        bins=[0, 3000, 6000, 10000, 20000],
        labels=["<3K", "3K-6K", "6K-10K", ">10K"],
    )
    inc_attr = (
        df.groupby("Income_Bracket", observed=True)["Attrition"]
        .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
        .reset_index(name="Attrition_Rate")
    )
    fig = px.bar(
        inc_attr,
        x="Income_Bracket",
        y="Attrition_Rate",
        color="Attrition_Rate",
        color_continuous_scale=["#2a9d8f", "#f4a261", "#e63946"],
        title="<b>Taux d'Attrition par Tranche de Salaire</b>",
    )
    fig.update_layout(
        xaxis_title="Tranche de Salaire ($)",
        yaxis_title="Taux d'Attrition (%)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _continuous_vars(df: pd.DataFrame) -> None:
    st.markdown("### 📊 Distribution des Variables Continues")
    options = [
        "Age", "MonthlyIncome", "YearsAtCompany",
        "DistanceFromHome", "TotalWorkingYears", "YearsInCurrentRole",
    ]
    var = st.selectbox("Sélectionnez une variable :", options)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x=var, color="Attrition",
            color_discrete_map={"Yes": "#e63946", "No": "#2a9d8f"},
            barmode="overlay", opacity=0.7,
            title=f"<b>Distribution de {var}</b>",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(
            df, x="Attrition", y=var, color="Attrition",
            color_discrete_map={"Yes": "#e63946", "No": "#2a9d8f"},
            title=f"<b>Boxplot de {var} par Attrition</b>",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Restés :**")
        st.dataframe(df[df["Attrition"] == "No"][var].describe().round(2))
    with col2:
        st.markdown("**📊 Partis :**")
        st.dataframe(df[df["Attrition"] == "Yes"][var].describe().round(2))


def _categorical_vars(df: pd.DataFrame) -> None:
    st.markdown("### 📊 Analyse par Variables Catégorielles")
    options = ["Department", "JobRole", "MaritalStatus", "BusinessTravel", "EducationField", "Gender"]
    cat = st.selectbox("Sélectionnez une variable :", options)

    cat_attr = (
        df.groupby(cat)["Attrition"]
        .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
        .reset_index(name="Attrition_Rate")
        .sort_values("Attrition_Rate")
    )
    fig = px.bar(
        cat_attr, y=cat, x="Attrition_Rate", orientation="h",
        color="Attrition_Rate",
        color_continuous_scale=["#2a9d8f", "#f4a261", "#e63946"],
        title=f"<b>Taux d'Attrition par {cat}</b>",
    )
    fig.update_layout(
        xaxis_title="Taux d'Attrition (%)",
        yaxis_title=cat,
        coloraxis_showscale=False,
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    detailed = df.groupby(cat).agg(
        Départs=("Attrition", lambda x: (x == "Yes").sum()),
        Total=("EmployeeNumber", "count"),
        Salaire_Moyen=("MonthlyIncome", "mean"),
    ).round(2)
    detailed["Taux Attrition (%)"] = (detailed["Départs"] / detailed["Total"] * 100).round(2)
    st.dataframe(
        detailed.style.background_gradient(cmap="RdYlGn_r", subset=["Taux Attrition (%)"])
    )


def _correlations(df: pd.DataFrame, df_processed: pd.DataFrame) -> None:
    st.markdown("### 🔗 Matrice de Corrélations")
    numeric_cols = [
        "Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears",
        "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
        "DistanceFromHome", "NumCompaniesWorked", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager", "PercentSalaryHike",
    ]
    corr = df_processed[numeric_cols + ["Attrition_Binary"]].corr()

    fig = px.imshow(
        corr,
        labels=dict(color="Corrélation"),
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="<b>Matrice de Corrélations</b>",
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Top Corrélations avec l'Attrition")
    attr_corr = corr["Attrition_Binary"].drop("Attrition_Binary").sort_values(key=abs, ascending=False)
    fig = px.bar(
        x=attr_corr.values, y=attr_corr.index, orientation="h",
        color=attr_corr.values,
        color_continuous_scale="RdBu_r",
        title="<b>Corrélations avec l'Attrition</b>",
    )
    fig.update_layout(
        xaxis_title="Coefficient de Corrélation",
        yaxis_title="Variable",
        coloraxis_showscale=False,
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
