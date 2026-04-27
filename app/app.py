"""
Employee Attrition Analytics — Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import warnings

import streamlit as st

warnings.filterwarnings("ignore")

from src.config import CUSTOM_CSS
from src.data.loader import load_data, preprocess_data
from src.features.engineering import create_feature_matrix
from src.views.exploratory import show_exploratory_analysis
from src.views.individual import show_individual_prediction
from src.views.overview import show_overview
from src.views.predictive import show_predictive_models
from src.views.recommendations import show_recommendations
from src.views.segmentation import show_segmentation

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analyse Attrition RH",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load():
    df = load_data()
    df_processed, le_dict = preprocess_data(df)
    X, y, feature_cols = create_feature_matrix(df_processed)
    return df, df_processed, le_dict, X, y, feature_cols


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🎯 Analyse de l'Attrition des Employés</h1>
            <p>Tableau de bord analytique pour la prédiction et la prévention du turnover</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Chargement des données…"):
        df, df_processed, le_dict, X, y, feature_cols = _load()

    # Sidebar navigation
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/000000/user-group-man-woman.png",
            width=80,
        )
        st.markdown("### 📊 Navigation")

        page = st.radio(
            "Sélectionnez une section :",
            [
                "📈 Vue d'ensemble",
                "🔍 Analyse Exploratoire",
                "🎯 Segmentation",
                "🤖 Modèles Prédictifs",
                "⚠️ Prédiction Individuelle",
                "📋 Recommandations",
            ],
        )

        st.markdown("---")
        st.markdown("### 📌 À propos")
        st.info(
            "Cette application analyse les facteurs d'attrition "
            "et prédit le risque de départ des employés."
        )
        st.markdown(
            "<small>Dataset : IBM HR Analytics · "
            "1 470 employés · 35 variables</small>",
            unsafe_allow_html=True,
        )

    # Page routing
    if page == "📈 Vue d'ensemble":
        show_overview(df, df_processed)
    elif page == "🔍 Analyse Exploratoire":
        show_exploratory_analysis(df, df_processed)
    elif page == "🎯 Segmentation":
        show_segmentation(df_processed, X, y)
    elif page == "🤖 Modèles Prédictifs":
        show_predictive_models(X, y, feature_cols)
    elif page == "⚠️ Prédiction Individuelle":
        show_individual_prediction(df, df_processed, X, y, feature_cols, le_dict)
    elif page == "📋 Recommandations":
        show_recommendations(df, df_processed)


if __name__ == "__main__":
    main()
