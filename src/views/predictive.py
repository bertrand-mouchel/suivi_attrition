"""
Modèles prédictifs — entraînement, métriques, courbes ROC / PR, matrices de confusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import cross_val_score

from src.models.classifier import train_models

_COLORS = ["#1d3557", "#e63946", "#2a9d8f"]


def show_predictive_models(X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> None:
    st.markdown(
        '<div class="section-header"><h2>🤖 Modèles Prédictifs</h2></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="insight-box">
            <h4>🔬 Méthodologie</h4>
            <p>Trois algorithmes ML sont comparés pour prédire le risque d'attrition :</p>
            <ul>
                <li><b>Random Forest</b> — Ensemble d'arbres, robuste au surapprentissage</li>
                <li><b>Gradient Boosting</b> — Construction séquentielle, optimise les erreurs résiduelles</li>
                <li><b>Régression Logistique</b> — Modèle linéaire interprétable (baseline)</li>
            </ul>
            <p>Découpage <b>80 % entraînement / 20 % test</b> stratifié + rééquilibrage SMOTE.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Entraînement des modèles en cours…"):
        results, X_train, X_test, y_train, y_test, scaler = train_models(X, y)

    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Performances Globales")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Meilleur Modèle", best_name)
    c2.metric("🎯 F1-Score", f"{best['f1']:.3f}")
    c3.metric("🔍 Recall", f"{best['recall']:.3f}")
    c4.metric("✅ Précision", f"{best['precision']:.3f}")

    st.markdown("---")

    # ── Comparison ───────────────────────────────────────────────────────────
    st.markdown("### 📊 Comparaison des Modèles")
    cmp = pd.DataFrame(
        [
            {"Modèle": n, "Accuracy": r["accuracy"], "Precision": r["precision"],
             "Recall": r["recall"], "F1-Score": r["f1"]}
            for n, r in results.items()
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for i, m in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"]):
            fig.add_trace(go.Bar(
                name=m, x=cmp["Modèle"], y=cmp[m],
                marker_color=["#1d3557", "#457b9d", "#a8dadc", "#2a9d8f"][i],
                text=cmp[m].apply(lambda v: f"{v:.2f}"),
                textposition="outside",
            ))
        fig.update_layout(title="<b>Comparaison des Performances</b>",
                          barmode="group", yaxis_range=[0, 1.15], height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for idx, (name, res) in enumerate(results.items()):
            vals = [res["accuracy"], res["precision"], res["recall"], res["f1"], res["accuracy"]]
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=["Accuracy", "Precision", "Recall", "F1-Score", "Accuracy"],
                fill="toself", name=name,
                line_color=_COLORS[idx], opacity=0.6,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="<b>Profil Radar des Modèles</b>", height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Tableau Récapitulatif")
    st.dataframe(
        cmp.set_index("Modèle").style.format("{:.3f}").background_gradient(cmap="Greens")
    )

    st.markdown("---")

    # ── Cross-validation ─────────────────────────────────────────────────────
    st.markdown("### 🔄 Validation Croisée (5-Fold)")
    cv_rows = []
    X_sc = scaler.transform(X)
    for name, res in results.items():
        for fold_i, sc in enumerate(cross_val_score(res["model"], X_sc, y, cv=5, scoring="f1")):
            cv_rows.append({"Modèle": name, "Fold": fold_i + 1, "F1-Score": sc})
    cv_df = pd.DataFrame(cv_rows)

    fig = px.box(cv_df, x="Modèle", y="F1-Score", color="Modèle",
                 color_discrete_sequence=_COLORS,
                 title="<b>Distribution F1-Score — 5-Fold CV</b>", points="all")
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    cv_sum = cv_df.groupby("Modèle")["F1-Score"].agg(["mean", "std"]).round(3)
    cv_sum.columns = ["F1 Moyen", "Écart-type"]
    st.dataframe(cv_sum.style.background_gradient(cmap="Greens", subset=["F1 Moyen"]))

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("### 🎯 Importance des Variables")
    col1, col2 = st.columns(2)

    def _importance_chart(model, title, palette):
        imp = (
            pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
            .sort_values("Importance")
            .tail(15)
        )
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=palette,
                     title=f"<b>Top 15 — {title}</b>")
        fig.update_layout(xaxis_title="Importance", yaxis_title="",
                          coloraxis_showscale=False, height=500)
        return fig

    with col1:
        st.plotly_chart(
            _importance_chart(results["Random Forest"]["model"], "Random Forest",
                               ["#a8dadc", "#1d3557"]),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            _importance_chart(results["Gradient Boosting"]["model"], "Gradient Boosting",
                               ["#f4a261", "#e63946"]),
            use_container_width=True,
        )

    lr = results["Logistic Regression"]["model"]
    lr_imp = (
        pd.DataFrame({"Feature": feature_cols, "Importance": np.abs(lr.coef_[0])})
        .sort_values("Importance")
        .tail(15)
    )
    fig = px.bar(lr_imp, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale=["#2a9d8f", "#264653"],
                 title="<b>Top 15 — Régression Logistique (|coefficients|)</b>")
    fig.update_layout(xaxis_title="Importance", yaxis_title="",
                      coloraxis_showscale=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    rf_top = set(pd.DataFrame({"Feature": feature_cols,
                                "I": results["Random Forest"]["model"].feature_importances_})
                 .nlargest(10, "I")["Feature"])
    gb_top = set(pd.DataFrame({"Feature": feature_cols,
                                "I": results["Gradient Boosting"]["model"].feature_importances_})
                 .nlargest(10, "I")["Feature"])
    lr_top = set(lr_imp.tail(10)["Feature"])
    common = rf_top & gb_top & lr_top
    st.info(f"📌 **Variables communes aux 3 modèles (Top 10) :** {', '.join(sorted(common))}")

    st.markdown("---")

    # ── ROC & PR curves ───────────────────────────────────────────────────────
    st.markdown("### 📈 Courbes d'Évaluation")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for idx, (name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={roc_auc:.3f})",
                                     mode="lines", line=dict(color=_COLORS[idx], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Aléatoire (AUC=0.500)",
                                  mode="lines", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="<b>Courbes ROC</b>",
                          xaxis_title="Faux Positifs (FPR)",
                          yaxis_title="Vrais Positifs (TPR)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for idx, (name, res) in enumerate(results.items()):
            prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
            pr_auc = auc(rec, prec)
            fig.add_trace(go.Scatter(x=rec, y=prec, name=f"{name} (AUC={pr_auc:.3f})",
                                     mode="lines", line=dict(color=_COLORS[idx], width=2)))
        fig.update_layout(title="<b>Courbes Précision-Rappel</b>",
                          xaxis_title="Rappel", yaxis_title="Précision", height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown("### 🎯 Matrices de Confusion")
    cols = st.columns(3)
    cm_palettes = ["Blues", "Reds", "Greens"]

    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        fig = px.imshow(
            cm, labels=dict(x="Prédit", y="Réel", color="Nombre"),
            x=["Resté", "Parti"], y=["Resté", "Parti"],
            color_continuous_scale=cm_palettes[idx], text_auto=True,
        )
        fig.update_layout(title=f"<b>{name}</b>", height=350, coloraxis_showscale=False)
        cols[idx].plotly_chart(fig, use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        cols[idx].caption(f"VP={tp} | FP={fp} | FN={fn} | VN={tn}")

    st.markdown("---")

    # ── Risk score distribution ───────────────────────────────────────────────
    st.markdown("### 📊 Distribution des Scores de Risque")
    proba_df = pd.DataFrame({
        "Probabilité de départ": results[best_name]["y_proba"],
        "Réalité": ["Parti" if v == 1 else "Resté" for v in y_test],
    })
    fig = px.histogram(
        proba_df, x="Probabilité de départ", color="Réalité",
        color_discrete_map={"Parti": "#e63946", "Resté": "#2a9d8f"},
        barmode="overlay", opacity=0.7, nbins=30,
        title=f"<b>Distribution des Probabilités Prédites ({best_name})</b>",
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="Seuil = 0.5")
    fig.update_layout(xaxis_title="Probabilité prédite", yaxis_title="Nombre d'employés", height=400)
    st.plotly_chart(fig, use_container_width=True)

    proba = results[best_name]["y_proba"]
    c1, c2 = st.columns(2)
    c1.metric("Employés à risque (>50%)", f"{(proba > 0.5).sum()} / {len(proba)}")
    c2.metric("Zone de vigilance (25-50%)", f"{((proba > 0.25) & (proba <= 0.5)).sum()} employés")

    st.markdown("---")

    # ── Synthesis ─────────────────────────────────────────────────────────────
    st.success(
        f"**🏆 Meilleur modèle : {best_name}** — "
        f"F1 {best['f1']:.3f} · Recall {best['recall']:.3f} · Précision {best['precision']:.3f}"
    )
    st.markdown(
        """
        <div class="insight-box">
            <h4>📌 Points clés</h4>
            <ul>
                <li><b>OverTime</b> est le facteur prédictif n°1</li>
                <li><b>MonthlyIncome</b> — salaires bas fortement corrélés à l'attrition</li>
                <li><b>Âge & Ancienneté</b> — les jeunes employés récents sont les plus volatils</li>
                <li><b>JobSatisfaction</b> — signal d'alerte précoce</li>
                <li><b>StockOptionLevel</b> — absence de stock options réduit l'engagement</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "⚠️ **Limite :** données déséquilibrées (~16 % de départs). "
        "Le Recall est la métrique prioritaire : mieux vaut une fausse alarme qu'un départ non détecté."
    )
