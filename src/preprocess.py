from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def get_preprocessing(trial, X):

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # ---- Numerical Pipeline ----
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
    ])

    use_scaler = trial.suggest_categorical("scaler", [True, False])
    if use_scaler:
        num_pipeline.steps.append(("scaler", StandardScaler()))

    # ---- Categorical Pipeline ----
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ---- Combine ----
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    return preprocessor