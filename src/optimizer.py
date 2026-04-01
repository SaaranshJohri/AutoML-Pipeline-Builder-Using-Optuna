import optuna
from src.objective import objective

def run_optimization(X, y, n_trials=30):

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    return study