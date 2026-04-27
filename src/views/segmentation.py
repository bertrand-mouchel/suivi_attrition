"""
Segmentation avancée — K-Means, visualisations PCA, clustering hiérarchique.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from src.models.clustering import (
    find_optimal_clusters,
    perform_clustering,
    perform_hierarchical_clustering,
)


def show_segmentation(df_processed: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
    st.markdown(
        '<div class="section-header"><h2>🎯 Segmentation Avancée des Employés</h2></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="insight-box">
            <p>La segmentation identifie des groupes homogènes d'employés avec des caractéristiques
            et des niveaux de risque similaires. Des métriques avancées optimisent le nombre
            de segments et évaluent la qualité du clustering.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 Optimisation & Clustering",
            "📊 Visualisations Avancées",
            "🔍 Analyse Détaillée",
            "🌳 Clustering Hiérarchique",
        ]
    )

    with tab1:
        n_clusters, cluster_results = _tab_optimisation(X)

    with tab2:
        _tab_visualisations(df_processed, cluster_results)

    with tab3:
        _tab_detailed(df_processed, cluster_results, n_clusters)

    with tab4:
        _tab_hierarchical(df_processed, X, cluster_results, n_clusters)


# ---------------------------------------------------------------------------
# Tab 1 — Optimisation & clustering
# ---------------------------------------------------------------------------

def _tab_optimisation(X: pd.DataFrame) -> tuple[int, dict]:
    st.markdown("### 🔬 Détermination du Nombre Optimal de Clusters")

    with st.spinner("Calcul des métriques d'optimisation…"):
        opt = find_optimal_clusters(X)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(
            go.Scatter(
                x=opt["k_range"], y=opt["inertias"],
                mode="lines+markers",
                marker=dict(size=10, color="#1d3557"),
                line=dict(width=3, color="#1d3557"),
            )
        )
        fig.update_layout(
            title="<b>Méthode du Coude (Elbow)</b>",
            xaxis_title="Nombre de Clusters (k)",
            yaxis_title="Inertie",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Chercher le 'coude' dans la courbe pour identifier k optimal.")

    with col2:
        sil_max = max(opt["silhouette_scores"])
        fig = go.Figure(
            go.Scatter(
                x=opt["k_range"], y=opt["silhouette_scores"],
                mode="lines+markers",
                marker=dict(size=10, color="#2a9d8f"),
                line=dict(width=3, color="#2a9d8f"),
                fill="tozeroy", fillcolor="rgba(42,157,143,0.2)",
            )
        )
        fig.add_hline(
            y=sil_max, line_dash="dash", line_color="red",
            annotation_text=f"Max : {sil_max:.3f}",
        )
        fig.update_layout(
            title="<b>Silhouette Score</b>",
            xaxis_title="Nombre de Clusters (k)",
            yaxis_title="Score",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Score 0→1. Plus élevé = meilleure séparation entre clusters.")

    st.markdown("### 💡 Recommandations")
    col1, col2 = st.columns(2)
    col1.info(f"**Silhouette optimal :** k = {opt['optimal_k_silhouette']}")
    col2.success(f"**Recommandé :** k = {opt['optimal_k_silhouette']}")

    st.markdown("---")
    st.markdown("### ⚙️ Configuration du Clustering")
    col1, col2 = st.columns([3, 1])
    with col1:
        n_clusters = st.slider(
            "Nombre de segments (k) :",
            2, 10, opt["optimal_k_silhouette"],
            help="Ajustez selon les métriques ci-dessus.",
        )
    with col2:
        st.metric("K sélectionné", n_clusters)

    with st.spinner("Clustering en cours…"):
        cluster_results = perform_clustering(X, n_clusters)

    st.markdown("### 📈 Métriques de Qualité")
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette Score", f"{cluster_results['silhouette_avg']:.3f}", help=">0.5 = bon, >0.7 = excellent")
    c2.metric("Variance PCA 2D", f"{cluster_results['variance_explained_2d'].sum()*100:.1f}%")
    c3.metric("Variance PCA 3D", f"{cluster_results['variance_explained_3d'].sum()*100:.1f}%")

    # Cumulative variance
    cv = cluster_results["cumulative_variance"]
    n_comp = min(20, len(cv))
    fig = go.Figure(
        go.Scatter(
            x=list(range(1, n_comp + 1)), y=cv[:n_comp] * 100,
            mode="lines+markers",
            marker=dict(size=8, color="#457b9d"),
            line=dict(width=3, color="#457b9d"),
            fill="tozeroy", fillcolor="rgba(69,123,157,0.2)",
        )
    )
    fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="90%")
    fig.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="95%")
    fig.update_layout(
        title="<b>Variance Cumulée par Composante PCA</b>",
        xaxis_title="Composantes",
        yaxis_title="Variance Expliquée (%)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Silhouette per cluster
    st.markdown("### 🎯 Silhouette Score par Cluster")
    sil_df = pd.DataFrame(
        [
            {
                "Cluster": f"Cluster {i}",
                "Silhouette": cluster_results["silhouette_vals"][cluster_results["clusters"] == i].mean(),
                "Taille": int((cluster_results["clusters"] == i).sum()),
            }
            for i in range(n_clusters)
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(sil_df, x="Cluster", y="Silhouette", color="Silhouette",
                     color_continuous_scale="RdYlGn",
                     title="<b>Silhouette Score par Cluster</b>")
        fig.add_hline(y=cluster_results["silhouette_avg"], line_dash="dash",
                      line_color="black", annotation_text="Moyenne")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(sil_df, x="Cluster", y="Taille", color="Taille",
                     color_continuous_scale="Blues",
                     title="<b>Taille des Clusters</b>")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sil_df.style.background_gradient(cmap="RdYlGn", subset=["Silhouette"]))

    return n_clusters, cluster_results


# ---------------------------------------------------------------------------
# Tab 2 — Visualisations avancées
# ---------------------------------------------------------------------------

def _tab_visualisations(df_processed: pd.DataFrame, cluster_results: dict) -> None:
    df_c = df_processed.copy()
    df_c["Cluster"] = cluster_results["clusters"]
    df_c["PCA1"]    = cluster_results["X_pca_2d"][:, 0]
    df_c["PCA2"]    = cluster_results["X_pca_2d"][:, 1]
    df_c["PCA3"]    = cluster_results["X_pca_3d"][:, 2]

    var2d = cluster_results["variance_explained_2d"].sum() * 100
    var3d = cluster_results["variance_explained_3d"].sum() * 100

    st.markdown("### 🎨 Projections PCA")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df_c, x="PCA1", y="PCA2",
            color="Cluster", symbol="Attrition",
            color_continuous_scale="viridis",
            hover_data=["Age", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany"],
            title=f"<b>PCA 2D — {var2d:.1f}% variance</b>",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter_3d(
            df_c, x="PCA1", y="PCA2", z="PCA3",
            color="Cluster", symbol="Attrition",
            color_continuous_scale="viridis",
            hover_data=["Age", "MonthlyIncome", "JobSatisfaction"],
            title=f"<b>PCA 3D — {var3d:.1f}% variance</b>",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.caption("💡 Pivotez la vue 3D avec la souris pour explorer les clusters sous différents angles.")

    st.markdown("### 📦 Distributions Comparatives par Variable")
    var = st.selectbox(
        "Sélectionnez une variable :",
        ["Age", "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance",
         "YearsAtCompany", "DistanceFromHome", "NumCompaniesWorked"],
    )
    df_c["Cluster_str"] = df_c["Cluster"].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df_c, x="Cluster_str", y=var, color="Cluster_str",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title=f"<b>Distribution de {var} par Cluster</b>")
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title="Cluster")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.violin(df_c, x="Cluster_str", y=var, color="Cluster_str",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        title=f"<b>Violin Plot — {var}</b>", box=True)
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title="Cluster")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Analyse détaillée
# ---------------------------------------------------------------------------

def _tab_detailed(df_processed: pd.DataFrame, cluster_results: dict, n_clusters: int) -> None:
    df_c = df_processed.copy()
    df_c["Cluster"] = cluster_results["clusters"]

    st.markdown("### 📋 Profil Détaillé des Clusters")
    agg = df_c.groupby("Cluster").agg(
        Taux_Attrition=("Attrition_Binary", "mean"),
        Nb_Departs=("Attrition_Binary", "sum"),
        Total=("Attrition_Binary", "count"),
        Salaire_Moyen=("MonthlyIncome", "mean"),
        Satisfaction=("JobSatisfaction", "mean"),
        Anciennete=("YearsAtCompany", "mean"),
        Age_Moyen=("Age", "mean"),
        WorkLife=("WorkLifeBalance", "mean"),
        Distance=("DistanceFromHome", "mean"),
    ).round(2)
    agg["Taux_Attrition"] = (agg["Taux_Attrition"] * 100).round(1)

    fig = px.bar(
        agg.reset_index(), x="Cluster", y="Taux_Attrition",
        color="Taux_Attrition",
        color_continuous_scale=["#2a9d8f", "#f4a261", "#e63946"],
        title="<b>Taux d'Attrition par Cluster</b>",
        text="Taux_Attrition",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_title="Cluster", yaxis_title="Taux d'Attrition (%)",
                      coloraxis_showscale=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Tableau Récapitulatif")
    st.dataframe(
        agg.style
        .background_gradient(cmap="RdYlGn_r", subset=["Taux_Attrition"])
        .background_gradient(cmap="Greens", subset=["Satisfaction", "WorkLife"])
        .format({
            "Taux_Attrition": "{:.1f}%",
            "Salaire_Moyen": "${:,.0f}",
            "Satisfaction": "{:.2f}",
            "Anciennete": "{:.1f}",
            "Age_Moyen": "{:.1f}",
            "WorkLife": "{:.2f}",
            "Distance": "{:.1f}",
        }),
        use_container_width=True,
    )

    st.markdown("### 🔍 Analyse par Cluster")
    for i in range(n_clusters):
        data = df_c[df_c["Cluster"] == i]
        rate = data["Attrition_Binary"].mean() * 100
        risk = "🔴 ÉLEVÉ" if rate > 25 else ("🟡 MODÉRÉ" if rate > 15 else "🟢 FAIBLE")

        with st.expander(
            f"📊 **Cluster {i}** — Risque {risk} ({len(data)} employés, {rate:.1f}% attrition)",
            expanded=(rate > 20),
        ):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Âge Moyen", f"{data['Age'].mean():.1f} ans")
            c2.metric("Salaire Moyen", f"${data['MonthlyIncome'].mean():,.0f}")
            c3.metric("Satisfaction", f"{data['JobSatisfaction'].mean():.2f}/4")
            c4.metric("Ancienneté", f"{data['YearsAtCompany'].mean():.1f} ans")
            c5.metric("Work-Life", f"{data['WorkLifeBalance'].mean():.2f}/4")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Heures Supplémentaires :**")
                for val, pct in (data["OverTime"].value_counts(normalize=True) * 100).items():
                    st.markdown(f"- {val} : {pct:.1f}%")
            with col2:
                if "Department" in data.columns:
                    st.markdown("**Départements :**")
                    for val, pct in (data["Department"].value_counts(normalize=True) * 100).head(3).items():
                        st.markdown(f"- {val} : {pct:.1f}%")
            with col3:
                if "MaritalStatus" in data.columns:
                    st.markdown("**Statut Marital :**")
                    for val, pct in (data["MaritalStatus"].value_counts(normalize=True) * 100).items():
                        st.markdown(f"- {val} : {pct:.1f}%")

            chars = _build_characteristics(data, df_processed)
            st.markdown("**✨ Caractéristiques Principales :**")
            for c in chars:
                st.markdown(c)


def _build_characteristics(cluster_data: pd.DataFrame, ref: pd.DataFrame) -> list[str]:
    chars = []
    if cluster_data["OverTime"].value_counts(normalize=True).get("Yes", 0) > 0.4:
        chars.append("• 🔴 **Fort taux d'heures supplémentaires** (>40%)")
    if cluster_data["MonthlyIncome"].mean() < ref["MonthlyIncome"].quantile(0.25):
        chars.append("• 💰 **Salaires dans le quartile inférieur**")
    if cluster_data["MonthlyIncome"].mean() > ref["MonthlyIncome"].quantile(0.75):
        chars.append("• 💎 **Salaires dans le quartile supérieur**")
    if cluster_data["JobSatisfaction"].mean() < 2.5:
        chars.append("• 😞 **Faible satisfaction au travail** (<2.5/4)")
    if cluster_data["JobSatisfaction"].mean() > 3.2:
        chars.append("• 😊 **Haute satisfaction au travail** (>3.2/4)")
    if cluster_data["YearsAtCompany"].mean() < 3:
        chars.append("• 🆕 **Employés récents** (< 3 ans)")
    if cluster_data["YearsAtCompany"].mean() > 10:
        chars.append("• 🏆 **Employés expérimentés** (> 10 ans)")
    if cluster_data["Age"].mean() < 30:
        chars.append("• 👶 **Population jeune** (< 30 ans)")
    if cluster_data["Age"].mean() > 45:
        chars.append("• 👴 **Population senior** (> 45 ans)")
    if cluster_data["DistanceFromHome"].mean() > 15:
        chars.append("• 🚗 **Distance domicile-travail élevée** (>15 km)")
    if cluster_data["WorkLifeBalance"].mean() < 2.5:
        chars.append("• ⚖️ **Mauvais équilibre vie/travail** (<2.5/4)")
    if cluster_data["StockOptionLevel"].mean() < 0.5:
        chars.append("• 📉 **Peu ou pas de stock options**")
    if cluster_data["NumCompaniesWorked"].mean() > 4:
        chars.append("• 🔄 **Forte mobilité professionnelle** (>4 entreprises)")
    if not chars:
        chars.append("• ✅ **Profil équilibré** sans facteur de risque majeur")
    return chars


# ---------------------------------------------------------------------------
# Tab 4 — Clustering hiérarchique
# ---------------------------------------------------------------------------

def _tab_hierarchical(
    df_processed: pd.DataFrame,
    X: pd.DataFrame,
    cluster_results: dict,
    n_clusters: int,
) -> None:
    st.markdown("### 🌳 Clustering Hiérarchique")
    st.markdown(
        """
        <div class="insight-box">
            <p>Le clustering hiérarchique construit une arborescence de groupes visualisable
            par un dendrogramme. Le nombre de clusters n'est pas requis à l'avance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        n_hier = st.slider("Nombre de clusters (hiérarchique) :", 2, 10, n_clusters,
                           key="hierarchical_k")
    with col2:
        st.metric("Clusters Hiérarchiques", n_hier)

    with st.spinner("Clustering hiérarchique en cours…"):
        hier = perform_hierarchical_clustering(X, n_hier)

    col1, col2 = st.columns(2)
    col1.metric("Silhouette Score", f"{hier['silhouette_avg']:.3f}")
    delta = hier["silhouette_avg"] - cluster_results["silhouette_avg"]
    col2.metric("vs K-Means", f"{delta:+.3f}", delta=f"{delta:+.3f}")

    # Dendrogramme
    st.markdown("### 🌲 Dendrogramme")
    MAX_SAMPLES = 100
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    if len(X) > MAX_SAMPLES:
        st.warning(f"⚠️ Dendrogramme limité à {MAX_SAMPLES} échantillons aléatoires.")
        idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X_sample = StandardScaler().fit_transform(X.iloc[idx])
        from scipy.cluster.hierarchy import linkage as _linkage
        lm = _linkage(X_sample, method="ward")
    else:
        lm = hier["linkage_matrix"]

    dendro = dendrogram(lm, no_plot=True)
    fig = go.Figure()
    for xi, yi in zip(dendro["icoord"], dendro["dcoord"]):
        fig.add_trace(
            go.Scatter(x=xi, y=yi, mode="lines",
                       line=dict(color="#1d3557", width=1.5),
                       hoverinfo="skip", showlegend=False)
        )
    fig.update_layout(
        title="<b>Dendrogramme du Clustering Hiérarchique</b>",
        xaxis_title="Échantillons", yaxis_title="Distance",
        height=600, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparison matrix
    df_cmp = df_processed.copy()
    df_cmp["Cluster_KMeans"]       = cluster_results["clusters"]
    df_cmp["Cluster_Hierarchical"] = hier["clusters"]

    confusion = pd.crosstab(
        df_cmp["Cluster_KMeans"], df_cmp["Cluster_Hierarchical"], normalize="index"
    ) * 100

    fig = px.imshow(
        confusion,
        labels=dict(x="Cluster Hiérarchique", y="Cluster K-Means", color="% Overlap"),
        color_continuous_scale="Blues",
        title="<b>Matrice de Correspondance K-Means vs Hiérarchique</b>",
        text_auto=".1f",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown("---")
    st.markdown("### 💾 Export des Résultats")
    col1, col2 = st.columns(2)
    with col1:
        exp = df_processed.copy()
        exp["Cluster_KMeans"]      = cluster_results["clusters"]
        exp["Silhouette_Score"]    = cluster_results["silhouette_vals"]
        csv = exp[["EmployeeNumber", "Cluster_KMeans", "Silhouette_Score",
                   "Attrition", "Age", "MonthlyIncome", "JobSatisfaction"]].to_csv(index=False)
        st.download_button("📥 Segments K-Means", csv,
                           file_name=f"kmeans_{n_clusters}.csv", mime="text/csv")
    with col2:
        exp2 = df_processed.copy()
        exp2["Cluster_Hierarchical"] = hier["clusters"]
        csv2 = exp2[["EmployeeNumber", "Cluster_Hierarchical",
                     "Attrition", "Age", "MonthlyIncome", "JobSatisfaction"]].to_csv(index=False)
        st.download_button("📥 Segments Hiérarchiques", csv2,
                           file_name=f"hierarchical_{n_hier}.csv", mime="text/csv")
