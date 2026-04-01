from src.data_loader import load_data
from src.optimizer import run_optimization
import optuna.visualization as vis


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("data/mountains_vs_beaches_preferences.csv", "Preference")

    study = run_optimization(X_train, y_train, n_trials=20)

    print("Best Trial:")
    print(study.best_trial.params)
    print("Best Score:", study.best_value)
    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()