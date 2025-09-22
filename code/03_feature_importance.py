# ========== Imports & Settings ==========
import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from utils.config import *

# Eigene Hilfsfunktionen & Config

# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Matplotlib Einstellungen
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "figure.titlesize": 18, "figure.labelsize": 14, "savefig.dpi": 300, "figure.dpi": 100,
})


parser = argparse.ArgumentParser(description="Trainiere Forecast-Modell mit Optuna-Hyperparameter-Tuning.")

parser.add_argument(
    "--ocean_points",
    type=int,
    default=5,
    help="Anzahl der Ozean-Punkte ."
)

parser.add_argument(
    "--n_trials",
    type=int,
    default=100,
    help="Anzahl der Trials für das Hyperparameter-Tuning."
)

args = parser.parse_args()
OCEAN_POINTS = args.ocean_points

N_TRIALS = args.n_trials

print(f"\nAnzahl der Ozean-Punkte: {OCEAN_POINTS}")
print(f"Anzahl der Trials: {N_TRIALS}\n")

# ========== Daten laden ==========
ocean_data_path = Path(f"../data/numerical_data/points{OCEAN_POINTS}")
print(ocean_data_path)
weather_data_path = Path(f"../data/numerical_data/points{WEATHER_POINTS}")
print(weather_data_path)

# save df_merged to ../data/tabular_data_FI/
file_name = f'df_merged{OCEAN_POINTS}_FI.tsv'
output_path = Path('../data/tabular_data_FI/')

# Load data
df_merged = pd.read_csv(output_path / file_name, sep='\t')

# ========== Hyperparameter-Tuning mit Optuna ==========
n_splits = 10


# Features und Zielvariable
X = df_merged.drop(columns=["time", "slev"])
y = df_merged["slev"]
tscv = TimeSeriesSplit(n_splits=n_splits)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1
    }
    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
    return np.mean(scores)

# Storage-URL für SQLite
storage_url = "sqlite:///./optuna_study.db"
study_name = f"water_level_rf_points{OCEAN_POINTS}_fi"

# 2) Studie anlegen oder laden
study = optuna.create_study(
    study_name=study_name,     # optional
    storage=storage_url,             # hier die DB-Verbindung
    direction="maximize",
    load_if_exists=True              # falls bereits vorhanden, weiterführen
)

study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

print("\nBeste Hyperparameter:", study.best_params)
print("\nBestes (negatives) MSE:", study.best_value)

