"""
Unsupervised-learning pipeline — K-Means and Hierarchical clustering.

Public API
----------
find_optimal_clusters(X, max_k)         -> dict
perform_clustering(X, n_clusters)       -> dict
perform_hierarchical_clustering(X, n_clusters) -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_CLUSTERS, MAX_CLUSTERS, RANDOM_STATE


def find_optimal_clusters(
    X: pd.DataFrame,
    max_k: int = MAX_CLUSTERS,
) -> dict:
    """
    Sweep k from 2 to *max_k* and collect Elbow, Silhouette and Davies-Bouldin metrics.

    Returns
    -------
    dict with keys:
        k_range, inertias, silhouette_scores, davies_bouldin_scores,
        optimal_k_silhouette, optimal_k_davies, X_scaled
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_range = range(2, max_k + 1)
    inertias: list[float] = []
    silhouette_scores: list[float] = []
    davies_bouldin_scores: list[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": davies_bouldin_scores,
        "optimal_k_silhouette": list(k_range)[int(np.argmax(silhouette_scores))],
        "optimal_k_davies": list(k_range)[int(np.argmin(davies_bouldin_scores))],
        "X_scaled": X_scaled,
    }


def perform_clustering(
    X: pd.DataFrame,
    n_clusters: int = DEFAULT_CLUSTERS,
) -> dict:
    """
    Run K-Means clustering and compute quality metrics + PCA projections.

    Returns
    -------
    dict with keys:
        clusters, X_pca_2d, X_pca_3d, kmeans, pca_2d, pca_3d, scaler,
        silhouette_avg, silhouette_vals, davies_bouldin,
        variance_explained_2d, variance_explained_3d, cumulative_variance, X_scaled
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = km.fit_predict(X_scaled)

    sil_avg = silhouette_score(X_scaled, clusters)
    sil_vals = silhouette_samples(X_scaled, clusters)
    db_score = davies_bouldin_score(X_scaled, clusters)

    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)

    pca_full = PCA()
    pca_full.fit(X_scaled)

    return {
        "clusters": clusters,
        "X_pca_2d": X_pca_2d,
        "X_pca_3d": X_pca_3d,
        "kmeans": km,
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "scaler": scaler,
        "silhouette_avg": sil_avg,
        "silhouette_vals": sil_vals,
        "davies_bouldin": db_score,
        "variance_explained_2d": pca_2d.explained_variance_ratio_,
        "variance_explained_3d": pca_3d.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca_full.explained_variance_ratio_),
        "X_scaled": X_scaled,
    }


def perform_hierarchical_clustering(
    X: pd.DataFrame,
    n_clusters: int = DEFAULT_CLUSTERS,
) -> dict:
    """
    Run Ward hierarchical clustering and compute quality metrics + linkage matrix.

    Returns
    -------
    dict with keys:
        clusters, linkage_matrix, silhouette_avg, davies_bouldin, model
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    clusters = model.fit_predict(X_scaled)

    linkage_matrix = linkage(X_scaled, method="ward")

    return {
        "clusters": clusters,
        "linkage_matrix": linkage_matrix,
        "silhouette_avg": silhouette_score(X_scaled, clusters),
        "davies_bouldin": davies_bouldin_score(X_scaled, clusters),
        "model": model,
    }
