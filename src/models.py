from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

def make_regression_pipeline(preprocessor, model: str = "linear") -> Pipeline:
    if model == "linear":
        clf = LinearRegression()
    elif model == "xgb":
        clf = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.8, n_jobs=-1
        )
    elif model == "rf":
        clf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    else:
        raise ValueError("model must be one of: linear, xgb, rf")
    return Pipeline([("pre", preprocessor), ("model", clf)])

def make_classification_pipeline(preprocessor, model: str = "logreg") -> Pipeline:
    if model == "logreg":
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
    elif model == "xgb":
        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.8, n_jobs=-1,
            eval_metric="logloss", tree_method="hist"
        )
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")
    else:
        raise ValueError("model must be one of: logreg, xgb, rf")
    return Pipeline([("pre", preprocessor), ("model", clf)])