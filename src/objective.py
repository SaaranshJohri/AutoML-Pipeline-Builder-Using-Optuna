from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from src.models import get_model
from src.preprocess import get_preprocessing

def objective(trial, X, y):

    preprocessor = get_preprocessing(trial, X)
    model = get_model(trial)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    score = cross_val_score(pipeline, X, y, cv=3).mean()

    return score