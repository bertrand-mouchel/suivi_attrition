# Project Structure

```
employee-attrition-analytics/
├── app.py                  # Streamlit entry point & routing
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # IBM HR dataset
│
├── src/
│   ├── config.py           # CSS styles & app-wide constants
│   │
│   ├── data/
│   │   └── loader.py       # load_data(), preprocess_data()
│   │
│   ├── features/
│   │   └── engineering.py  # create_feature_matrix()
│   │
│   ├── models/
│   │   ├── classifier.py   # train_models()
│   │   └── clustering.py   # find_optimal_clusters(), perform_clustering(), perform_hierarchical_clustering()
│   │
│   └── views/
│       ├── overview.py         # Vue d'ensemble
│       ├── exploratory.py      # Analyse exploratoire
│       ├── segmentation.py     # Segmentation avancée
│       ├── predictive.py       # Modèles prédictifs
│       ├── individual.py       # Prédiction individuelle
│       └── recommendations.py  # Recommandations stratégiques
│
└── docs/
    └── project_structure.md
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/config.py` | Custom CSS, color palette, page constants |
| `src/data/loader.py` | Raw data ingestion, binary encoding, feature derivation |
| `src/features/engineering.py` | Assembles the feature matrix used by ML models |
| `src/models/classifier.py` | Trains RF / GBM / LR with SMOTE + optimised thresholds |
| `src/models/clustering.py` | K-Means, Hierarchical clustering & evaluation metrics |
| `src/views/` | One module per Streamlit page, pure presentation logic |
