from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from src.utils import derive_eda_columns, duration_bucket

# Columnas base desde EDA
NUM_COLS = ["duration_sec", "title_len", "title_words", "tag_count"]
CAT_COLS = ["category", "duration_bucket", "title_has_question", "title_has_exclaim"]
TEXT_COLS = ["title"]

# --- TF-IDF alineado al temario (stopwords EN/ES + filtro numérico simple) ---
def _tfidf_title(min_df=10):
    return TfidfVectorizer(
        min_df=min_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
        stop_words=["english", "spanish"],
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )

def _tfidf_hashtags(min_df=8):
    return TfidfVectorizer(
        min_df=min_df,
        ngram_range=(1, 1),
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b[#]?[A-Za-z][\w-]*\b"
    )

# ---------- Frame base para modelar ----------
def build_model_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = derive_eda_columns(df_raw)
    df["duration_bucket"] = df["duration_sec"].apply(duration_bucket)
    return df

# ---------- Etiqueta (para supervisado; no usar como feature) ----------
def make_target_hit_er(df: pd.DataFrame, p: float = 0.90) -> pd.Series:
    tmp = df.copy()
    denom = tmp["views"].replace(0, np.nan)
    tmp["er"] = (tmp["likes"] + tmp["comments"]) / denom
    tmp["duration_bucket"] = tmp["duration_sec"].apply(duration_bucket)
    p90 = tmp.groupby(["category", "duration_bucket"])["er"].transform(lambda s: s.quantile(p))
    return (tmp["er"] >= p90).astype(int)

# ---------- Preprocesador PRE-PUBLICACIÓN (supervisado) ----------
def make_preprocessor_prepub(use_hashtags: bool = False) -> Tuple[ColumnTransformer, Dict]:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = [
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
        ("tfidf_title", _tfidf_title(min_df=10), "title"),
    ]
    if use_hashtags:
        transformers.append(("tfidf_hashtags", _tfidf_hashtags(min_df=8), "hashtags"))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    manifest = {"numericas": NUM_COLS, "categoricas": CAT_COLS, "texto": ["title"] + (["hashtags"] if use_hashtags else [])}
    return pre, manifest

# ---------- Preprocesador NO SUPERVISADO (K-Means) ----------
def make_preprocessor_unsupervised(
    include_numeric: bool = True,
    include_categorical: bool = True,
    include_title_tfidf: bool = True,
    include_hashtags_tfidf: bool = False,
    svd_components: Optional[int] = 120
) -> Pipeline:
    transformers = []
    if include_numeric:
        transformers.append(("num",
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]),
            NUM_COLS))
    if include_categorical:
        transformers.append(("cat",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
            CAT_COLS))
    if include_title_tfidf:
        transformers.append(("tfidf_title", _tfidf_title(min_df=10), "title"))
    if include_hashtags_tfidf:
        transformers.append(("tfidf_hashtags", _tfidf_hashtags(min_df=8), "hashtags"))

    ct = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    steps = [("ct", ct)]
    if svd_components is not None and svd_components > 0:
        steps.append(("svd", TruncatedSVD(n_components=svd_components, random_state=42)))
    return Pipeline(steps)

# ---------- TF-IDF puro (solo texto) ----------
def make_tfidf_only(column: str = "title", min_df: int = 10) -> TfidfVectorizer:
    if column == "title":
        return _tfidf_title(min_df=min_df)
    return TfidfVectorizer(min_df=min_df, ngram_range=(1, 1), sublinear_tf=True, norm="l2")

# ---------- Grafo kNN por similitud coseno (para Greedy) ----------
def cosine_knn_graph(X, k: int = 15, sim_threshold: float = 0.25) -> List[tuple]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    edges = []
    n = X.shape[0]
    for i in range(n):
        for d, j in zip(distances[i, 1:], indices[i, 1:]):  # omite self
            sim = 1.0 - float(d)
            if sim >= sim_threshold:
                edges.append((int(i), int(j), sim))
    return edges

def prepare_graph_for_greedy(
    df: pd.DataFrame,
    text_col: str = "title",
    min_df: int = 10,
    k: int = 15,
    sim_threshold: float = 0.25
):
    vec = make_tfidf_only(column=text_col, min_df=min_df)
    X = vec.fit_transform(df[text_col].fillna("").astype(str))
    edges = cosine_knn_graph(X, k=k, sim_threshold=sim_threshold)
    index_to_row = dict(enumerate(df.index.tolist()))
    return edges, index_to_row
