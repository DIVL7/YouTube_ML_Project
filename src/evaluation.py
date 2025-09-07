import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, roc_auc_score, average_precision_score, precision_recall_curve

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

def bin_metrics(y_true, y_prob):
    """AUC-ROC y AUC-PR."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
    }

def precision_at_k(y_true, y_prob, k=0.1):
    """Precision@k: k fracción (0.1 = top 10%)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    top = max(1, int(round(k * n)))
    idx = np.argsort(-y_prob)[:top]
    return float(np.mean(y_true[idx]))

def evaluate_cv_prob(clf, X, y, cv):
    """
    Evalúa con validación cruzada estratificada devolviendo
    AUC-PR, AUC-ROC y Precision@k (k=10%).
    """
    from sklearn.model_selection import cross_val_predict
    y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    out = bin_metrics(y, y_prob)
    out["p_at_10"] = precision_at_k(y, y_prob, k=0.10)
    return out
