from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def make_classification_pipeline(preprocessor, model: str = "logreg") -> Pipeline:
    if model == "logreg":
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        return Pipeline([("pre", preprocessor), ("model", clf)])

    elif model == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        return Pipeline([("pre", preprocessor), ("model", clf)])

    elif model == "xgb":
        # Paso de seguridad: castear la matriz a float32 (CSR/CSC soportado)
        to_float32 = FunctionTransformer(lambda X: X.astype("float32"), accept_sparse=True, check_inverse=False)
        clf = XGBClassifier(
            clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="approx",      # ← en lugar de 'hist'
            predictor="cpu_predictor",
            n_jobs=1, eval_metric="logloss", random_state=42, verbosity=0
            )
        )
        return Pipeline([("pre", preprocessor), ("to_float32", to_float32), ("model", clf)])

    else:
        raise ValueError("model must be one of: logreg, rf, xgb")
