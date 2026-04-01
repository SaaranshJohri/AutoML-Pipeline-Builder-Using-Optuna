from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_model(trial):
    model_type = trial.suggest_categorical("model", ["rf", "xgb", "lr"])

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            random_state=42
        )

    elif model_type == "xgb":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("lr", 0.01, 0.3),
            eval_metric='logloss'
        )

    elif model_type == "lr":
        return LogisticRegression(
            C=trial.suggest_float("C", 0.01, 10),
            max_iter=5000
        )