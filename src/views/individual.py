"""
Prédiction individuelle — évaluation du risque d'un employé saisi manuellement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.models.classifier import train_models


def show_individual_prediction(
    df: pd.DataFrame,
    df_processed: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    le_dict: dict,
) -> None:
    st.markdown(
        '<div class="section-header"><h2>⚠️ Prédiction de Risque Individuel</h2></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Renseignez le profil d'un employé pour estimer son risque d'attrition "
        "et obtenir des recommandations personnalisées."
    )

    with st.spinner("Chargement du modèle…"):
        results, _, _, _, _, scaler = train_models(X, y)
    model = results["Random Forest"]["model"]

    # ── Input form ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 👤 Informations Personnelles")
        age              = st.slider("Âge", 18, 60, 35)
        gender           = st.selectbox("Genre", ["Male", "Female"])
        marital_status   = st.selectbox("Statut Marital", ["Single", "Married", "Divorced"])
        distance         = st.slider("Distance Domicile-Travail (km)", 1, 30, 10)
        education        = st.slider("Niveau d'Éducation (1-5)", 1, 5, 3)
        education_field  = st.selectbox(
            "Domaine d'Études",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        )

    with col2:
        st.markdown("### 💼 Informations Professionnelles")
        department      = st.selectbox("Département", ["Sales", "Research & Development", "Human Resources"])
        job_role        = st.selectbox("Poste", sorted(df["JobRole"].unique()))
        job_level       = st.slider("Niveau Hiérarchique (1-5)", 1, 5, 2)
        monthly_income  = st.slider("Salaire Mensuel ($)", 1000, 20000, 5000)
        years_at_company= st.slider("Années dans l'entreprise", 0, 40, 5)
        years_in_role   = st.slider("Années dans le poste actuel", 0, 20, 3)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 😊 Satisfaction & Performance")
        job_satisfaction = st.slider("Satisfaction au Travail (1-4)", 1, 4, 3)
        env_satisfaction = st.slider("Satisfaction Environnement (1-4)", 1, 4, 3)
        work_life        = st.slider("Équilibre Vie/Travail (1-4)", 1, 4, 3)
        job_involvement  = st.slider("Implication (1-4)", 1, 4, 3)
        performance      = st.slider("Performance (3-4)", 3, 4, 3)

    with col4:
        st.markdown("### ⏰ Autres Facteurs")
        overtime        = st.selectbox("Heures Supplémentaires", ["No", "Yes"])
        business_travel = st.selectbox(
            "Déplacements Professionnels",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        )
        num_companies   = st.slider("Entreprises précédentes", 0, 10, 2)
        training_times  = st.slider("Formations l'an dernier", 0, 6, 2)
        stock_option    = st.slider("Stock Options (0-3)", 0, 3, 1)

    if st.button("🔮 Prédire le Risque d'Attrition", type="primary"):
        risk_score = _predict(
            model, scaler, feature_cols, le_dict,
            age, gender, marital_status, distance, education, education_field,
            department, job_role, job_level, monthly_income,
            years_at_company, years_in_role,
            job_satisfaction, env_satisfaction, work_life, job_involvement, performance,
            overtime, business_travel, num_companies, training_times, stock_option,
        )
        _display_result(risk_score)
        _display_risk_factors(
            overtime, monthly_income, job_satisfaction, work_life,
            distance, years_at_company, env_satisfaction, stock_option,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _predict(
    model,
    scaler,
    feature_cols: list[str],
    le_dict: dict,
    age, gender, marital_status, distance, education, education_field,
    department, job_role, job_level, monthly_income,
    years_at_company, years_in_role,
    job_satisfaction, env_satisfaction, work_life, job_involvement, performance,
    overtime, business_travel, num_companies, training_times, stock_option,
) -> float:
    total_working_years = years_at_company + 5

    base = {
        "Age": age,
        "DistanceFromHome": distance,
        "Education": education,
        "EnvironmentSatisfaction": env_satisfaction,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "NumCompaniesWorked": num_companies,
        "PercentSalaryHike": 15,
        "PerformanceRating": performance,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": stock_option,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times,
        "WorkLifeBalance": work_life,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": min(years_in_role, 5),
        "OT_Binary": int(overtime == "Yes"),
        "Single": int(marital_status == "Single"),
        "Income_per_Year": monthly_income / (total_working_years + 1),
        "Years_Since_Promo_Ratio": 2 / (years_at_company + 1),
        "Satisfaction_Avg": (job_satisfaction + env_satisfaction + 3 + work_life) / 4,
        "Young_New": int(age < 30 and years_at_company < 3),
        "No_StockOption": int(stock_option == 0),
        "Travel_Freq": int(business_travel == "Travel_Frequently"),
    }

    # Encoded categoricals
    cat_values = {
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "OverTime": overtime,
    }
    for col, val in cat_values.items():
        if col in le_dict:
            base[f"{col}_Encoded"] = le_dict[col].transform([val])[0]

    vector = np.array([base.get(c, 0) for c in feature_cols]).reshape(1, -1)
    vector_sc = scaler.transform(vector)
    return float(model.predict_proba(vector_sc)[0][1] * 100)


def _display_result(risk_score: float) -> None:
    st.markdown("---")
    st.markdown("## 🎯 Résultat de la Prédiction")

    _, col, _ = st.columns([1, 2, 1])
    with col:
        if risk_score > 50:
            st.error(
                f"### 🔴 RISQUE ÉLEVÉ\n\n"
                f"**Score de risque : {risk_score:.1f}%**\n\n"
                "Des actions préventives sont recommandées en urgence."
            )
        elif risk_score > 25:
            st.warning(
                f"### 🟡 RISQUE MODÉRÉ\n\n"
                f"**Score de risque : {risk_score:.1f}%**\n\n"
                "Une surveillance et des discussions régulières sont conseillées."
            )
        else:
            st.success(
                f"### 🟢 RISQUE FAIBLE\n\n"
                f"**Score de risque : {risk_score:.1f}%**\n\n"
                "Cet employé semble engagé et stable."
            )


def _display_risk_factors(
    overtime, monthly_income, job_satisfaction, work_life,
    distance, years_at_company, env_satisfaction, stock_option,
) -> None:
    st.markdown("### ⚠️ Facteurs de Risque Identifiés")

    factors: list[tuple[str, str, str]] = []

    if overtime == "Yes":
        factors.append(("🔴", "Heures supplémentaires fréquentes", "Réduire la charge de travail"))
    if monthly_income < 4000:
        factors.append(("🔴", "Salaire inférieur à la moyenne", "Envisager une révision salariale"))
    if job_satisfaction < 3:
        factors.append(("🟡", "Satisfaction au travail faible", "Organiser un entretien de feedback"))
    if work_life < 3:
        factors.append(("🟡", "Équilibre vie/travail dégradé", "Proposer télétravail ou horaires flexibles"))
    if distance > 20:
        factors.append(("🟡", "Distance domicile-travail importante", "Envisager le télétravail partiel"))
    if years_at_company < 2:
        factors.append(("🟡", "Employé récent", "Renforcer l'accompagnement et l'intégration"))
    if env_satisfaction < 3:
        factors.append(("🟡", "Insatisfaction de l'environnement", "Améliorer les conditions de travail"))
    if stock_option == 0:
        factors.append(("🟢", "Pas de stock options", "Envisager un plan d'intéressement"))

    if not factors:
        st.info("✅ Aucun facteur de risque majeur identifié.")
        return

    for severity, factor, action in factors:
        st.markdown(
            f"""
            <div class="recommendation-card">
                <b>{severity} {factor}</b><br>
                <i>💡 Action recommandée : {action}</i>
            </div>
            """,
            unsafe_allow_html=True,
        )
