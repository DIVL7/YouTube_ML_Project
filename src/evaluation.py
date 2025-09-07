import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

def kmeans_silhouette(X, labels):
    """Silhouette como criterio de K (alineado al temario)."""
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        # Si la matriz es esparsa y falla, densificamos en casos pequeños
        return float(silhouette_score(np.asarray(X.todense()), labels))

def top_terms_by_groups(vec, X_tfidf, groups, topn=12):
    """
    vec: TfidfVectorizer ya 'fit'
    X_tfidf: matriz tf-idf (n_samples x vocab)
    groups: asignaciones de cluster/comunidad
    """
    vocab = np.array(vec.get_feature_names_out())
    groups = np.asarray(groups)
    rows = []
    for g in pd.Series(groups).unique():
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            continue
        mean_vec = np.asarray(X_tfidf[idx].mean(axis=0)).ravel()
        top_idx = np.argsort(-mean_vec)[:topn]
        rows.append({"group": int(g), "size": int(len(idx)), "top_terms": ", ".join(vocab[top_idx])})
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
