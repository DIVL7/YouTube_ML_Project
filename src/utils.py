import pandas as pd
import numpy as np

def derive_eda_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Casting numérico básico
    for col in ["views", "likes", "comments", "duration"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Duración a segundos (si viene con otra unidad, ya vendrá numérico)
    if "duration" in out.columns:
        out["duration_sec"] = out["duration"].astype(float)
    else:
        out["duration_sec"] = np.nan

    # Rasgos de título
    if "title" in out.columns:
        s = out["title"].astype(str)
        out["title_len"] = s.str.len()
        out["title_words"] = s.str.split().apply(len)
        out["title_has_question"] = s.str.contains(r"\?")
        out["title_has_exclaim"] = s.str.contains(r"!")
    else:
        out["title_len"] = out["title_words"] = np.nan
        out["title_has_question"] = out["title_has_exclaim"] = np.nan

    # Conteo simple de hashtags (si vienen separados por coma)
    if "hashtags" in out.columns:
        out["tag_count"] = out["hashtags"].astype(str).apply(
            lambda x: len([t for t in [s.strip() for s in x.split(",")] if t]) if isinstance(x, str) and x.strip() else 0
        )
    else:
        out["tag_count"] = np.nan

    # Engagement rate (solo para análisis; NO usar como feature si es target)
    if {"views","likes","comments"} <= set(out.columns):
        denom = out["views"].replace(0, np.nan)
        out["engagement_rate"] = (out["likes"] + out["comments"]) / denom
    else:
        out["engagement_rate"] = np.nan

    return out

def duration_bucket(sec):
    if pd.isna(sec): return "unknown"
    sec = float(sec)
    if sec < 120: return "<2m"
    if sec < 300: return "2-5m"
    if sec < 600: return "5-10m"
    if sec < 1200: return "10-20m"
    return "20m+"
