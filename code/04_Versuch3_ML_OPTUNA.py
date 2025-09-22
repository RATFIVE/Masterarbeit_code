
# ## Import Libaries
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, recall_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from utils.dl_helper_functions import (
    create_sequences,
    load_picture_lagged_data,
    scale_data,
)
import argparse
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

HORIZON = 24  # 1 day of forecast
SEQUENCE_LENGTH = 24 # 1 days of data
DTYPE_NUMPY = np.float32  # Use float32 for numpy arrays

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trainiere Forecast-Modell mit Optuna-Hyperparameter-Tuning."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="XGBoost",
        choices=["RandomForest", "XGBoost", "SVR", "LGBM", "Linear"],
        help="Modelltyp für das Forecasting"
    )
    return parser.parse_args()

args = parse_arguments()
n_jobs = -1  # Use all available CPU cores for parallel processing
model_name = args.model  # z.B. "RandomForest", "SVR", "XGBoost", "LGBM", "Linear"
storage = f"sqlite:///Versuch3_{model_name}.db"  # SQLite-Datenbank für Optuna
study_name = f"{HORIZON}"  # Name der Optuna-Studie
n_trials = 1000  # Anzahl der Versuche für die Hyperparameter-Optimierung
print(f"Run Model: {model_name}, with {n_trials} trials and HORIZON of {HORIZON} hours and SEQUENCE_LENGTH of {SEQUENCE_LENGTH} hours.")


# # Load Data
X, y_lagged, y, common_time = load_picture_lagged_data(return_common_time=True, verbose=False, grid_size=25, n_jobs=n_jobs, dtype=DTYPE_NUMPY, pca=True)


# Annahmen (vorab gesetzt)
DTYPE_NUMPY = np.float32
n_jobs = -1

# Daten vorbereiten
X = X.astype(DTYPE_NUMPY)
y_lagged = y_lagged.astype(DTYPE_NUMPY)
y = y.astype(DTYPE_NUMPY)

# Cross-Validation Zeitpunkte
folds = {
    "Surge1": pd.Timestamp("2023-02-25 16:00:00"),
    "Surge2": pd.Timestamp("2023-04-01 09:00:00"),
    "Surge3": pd.Timestamp("2023-10-07 20:00:00"),
    "Surge4": pd.Timestamp("2023-10-20 21:00:00"),
    "Surge5": pd.Timestamp("2024-01-03 01:00:00"),
    "Surge6": pd.Timestamp("2024-02-09 15:00:00"),
    "Surge7": pd.Timestamp("2024-12-09 10:00:00"),
    "normal1": pd.Timestamp("2023-07-01 14:00:00"),
    "normal2": pd.Timestamp("2024-04-01 18:00:00"),
    "normal3": pd.Timestamp("2025-01-01 12:00:00"),
}


def custom_score(y_true=None, y_pred=None, bins=[1, 2.00], alpha=0.7):
    
    # Initialisiere Recall- und MSE-Werte
    recalls = []
    for i in range(y_true.shape[1]):  # Iteriere über jede Spalte
        y_true_class = np.digitize(y_true[:, i], bins=bins)
        y_pred_class = np.digitize(y_pred[:, i], bins=bins)
        recalls.append(recall_score(y_true_class, y_pred_class, average="macro"))
    
    mean_recall = np.mean(recalls)  # Durchschnittlicher Recall
    mse = mean_squared_error(y_true, y_pred)
    return alpha * (1 - mean_recall) + (1 - alpha) * mse


def get_model(name, trial_params=None):
    if trial_params is None:
        trial_params = {}

    if name == "RandomForest":
        return RandomForestRegressor(random_state=42, n_jobs=n_jobs, **trial_params)
    elif name == "SVR":
        return MultiOutputRegressor(SVR(**trial_params), n_jobs=n_jobs)
    elif name == "XGBoost":
        return MultiOutputRegressor(XGBRegressor(random_state=42, n_jobs=n_jobs, **trial_params), n_jobs=n_jobs)
    elif name == "LGBM":
        return MultiOutputRegressor(LGBMRegressor(random_state=42, n_jobs=n_jobs, **trial_params), n_jobs=n_jobs)
    elif name == "Linear":
        return MultiOutputRegressor(LinearRegression(n_jobs=n_jobs), n_jobs=n_jobs)
    else:
        raise ValueError(f"Unbekanntes Modell: {name}")


def cross_validation_loop(model_name, folds, X, y_lagged, y, common_time, time_delta, trial_params=None, trial:optuna.Trial=None):
    fold_results = []

    for surge_name, fold in folds.items():
        start_cutoff = fold - time_delta
        end_cutoff = fold + time_delta
        idx_start_cutoff = np.where(common_time == start_cutoff)[0][0]
        idx_end_cutoff = np.where(common_time == end_cutoff)[0][0]

        X_test = X[idx_start_cutoff:idx_end_cutoff]
        y_lagged_test = y_lagged[idx_start_cutoff:idx_end_cutoff]
        y_test = y[idx_start_cutoff:idx_end_cutoff]

        X_train = X.copy()
        y_lagged_train = y_lagged.copy()
        y_train = y.copy()

        X_train[idx_start_cutoff:idx_end_cutoff] = np.nan
        y_lagged_train[idx_start_cutoff:idx_end_cutoff] = np.nan
        y_train[idx_start_cutoff:idx_end_cutoff] = np.nan

        X_train, y_lagged_train, y_train = create_sequences(X_train, y_lagged_train, y_train, SEQUENCE_LENGTH, HORIZON)
        X_test, y_lagged_test, y_test = create_sequences(X_test, y_lagged_test, y_test, SEQUENCE_LENGTH, HORIZON)

        gap = 168
        X_test = X_test[gap:-gap]
        y_lagged_test = y_lagged_test[gap:-gap]
        y_test = y_test[gap:-gap]

        
        data = scale_data(X_train, y_lagged_train, y_train,
                          None, None, None,
                          X_test, y_lagged_test, y_test,
                          dtype=DTYPE_NUMPY, verbose=True)

        X_train, y_lagged_train, y_train, _, _, _, X_test, y_lagged_test, y_test, X_scalers, y_lagged_scalers  = data

        X_train = np.hstack([X_train.reshape(X_train.shape[0], -1), y_lagged_train.reshape(y_lagged_train.shape[0], -1)])
        X_test = np.hstack([X_test.reshape(X_test.shape[0], -1), y_lagged_test.reshape(y_lagged_test.shape[0], -1)])

        model = get_model(model_name, trial_params=trial_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = custom_score(y_test, y_pred)
        fold_results.append(score)
        # Report intermediate results to Optuna
        mean_score_so_far = np.mean(fold_results)
        trial.report(mean_score_so_far, step=len(fold_results))
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at fold {surge_name} with score {mean_score_so_far}.")
            raise optuna.TrialPruned()

    return fold_results


def objective(trial: optuna.trial.Trial):
    

    if model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]), 
        }
    elif model_name == "SVR":
        params = {
            "C": trial.suggest_float("C", 0.01, 50, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            "max_iter": 50000
        }

        if params["kernel"] == "poly":
            params["degree"] = trial.suggest_int("degree", 1, 2)
            params["coef0"] = trial.suggest_float("coef0", 0.0, 1.0)
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])

        if params["kernel"] in ["rbf"]:
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        
            


    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 20.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "verbosity": 0
        }
    elif model_name == "LGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 20.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 20.0),
            "verbosity": 0
        }
    else:
        params = {}

    try:
        scores = cross_validation_loop(model_name, folds, X, y_lagged, y, common_time, pd.Timedelta(hours=168 * 4), params, trial=trial)
        score = np.mean(scores)
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float("inf")  # Markiere den Versuch als schlecht
    
    print(f"Finished trial with model {model_name} and params {params}, score: {score}")
    return score




# Optuna Study starten
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
study = optuna.create_study(direction="minimize", study_name=f"{study_name}", storage=storage, load_if_exists=True, pruner=pruner)
study.optimize(objective, n_trials=n_trials, n_jobs=1)

print("Beste Parameter:", study.best_params)
print("Bester Score:", study.best_value)



