# Standard‐Library
import argparse
import warnings
from pathlib import Path
from pprint import pprint

# Drittanbieter
import numpy as np
import optuna
import pandas as pd
import xarray as xr
import yaml
from lightgbm import LGBMRegressor
from mlforecast import MLForecast

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsforecast.models import ARIMA
from statsforecast import StatsForecast
from mlforecast.feature_engineering import transform_exog


# Eigene Module (utils)
from utils.config import (
    OCEAN_POINTS,
    WEATHER_POINTS,
)
from utils.eda_helper_functions import (
    group_data_hourly,
    load_insitu_data,
    load_ocean_data,
    load_weather_data,
    process_df,
    process_flensburg_data,
    
)
from utils.ml_helper_functions import (
    feature_engineering,
    run_eof_reduction,
    convert_to_xarray,
    prepare_ml_data,
    load_data,
    load_data_v2,
)
from xgboost import XGBRegressor

# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore")

# ganz oben im Skript
# STORAGE_URL = "sqlite:///./optuna_study_ML_custom_function.db"
# STUDY_NAME = f"XGBOOST_points{OCEAN_POINTS}_ml"
# STUDY_NAME = "test"

OCEAN_POINTS = 30 

HORIZON = 24  # 7 Tage

































# def load_data():
#     ocean_data_path = Path(f"../data/numerical_data/points{OCEAN_POINTS}")
#     weather_data_path = Path(f"../data/numerical_data/points{WEATHER_POINTS}")

#     print("\nLoading data from:")
#     print(ocean_data_path)
#     print(weather_data_path)

#     # save df_merged to ../data/tabular_data_FI/
#     # file_name = f'df_merged{OCEAN_POINTS}_FI.tsv'
#     # output_path = Path('../data/tabular_data_FI/')

#     print("\nLoading ocean, weather and insitu data...")
#     df_ocean = load_ocean_data(ocean_data_path, OCEAN_POINTS, verbose=False)
#     df_ocean = process_df(df_ocean, drop_cols=["depth"], verbose=False)

#     df_weather = load_weather_data(weather_data_path, WEATHER_POINTS, verbose=False)
#     df_weather = process_df(df_weather, verbose=False, drop_cols=['showers'])

#     df_insitu = load_insitu_data(verbose=False)
#     df_insitu = process_flensburg_data(df_insitu, 
#                                         start_time=df_ocean['time'].min(),
#                                         end_time=df_ocean['time'].max(),
#                                         verbose=False)

#     df_insitu = group_data_hourly(df_insitu)
#     df_insitu = process_df(df_insitu, drop_cols=["depth",'deph', 'latitude', 'longitude', 'time_qc', 'slev_qc'], verbose=False)

#     print(f"\nShapes of loaded DataFrames before feature engineering:\n"
#           f"Ocean DataFrame: {df_ocean.shape}\n"
#           f"Weather DataFrame: {df_weather.shape}\n"
#           f"Insitu DataFrame: {df_insitu.shape}")

#     print("\nFeature engineering...")
#     df_ocean, df_weather, df_insitu = feature_engineering(df_ocean, df_weather, df_insitu)

#     print(f"\nShapes of loaded DataFrames after feature engineering:\n"
#           f"Ocean DataFrame: {df_ocean.shape}\n"
#           f"Weather DataFrame: {df_weather.shape}\n"
#           f"Insitu DataFrame: {df_insitu.shape}")

#     # Convert DataFrames to xarray DataArrays
#     print("\nConverting DataFrames to xarray DataArrays...")
#     ds_ocean = convert_to_xarray(df_ocean)
#     ds_weather = convert_to_xarray(df_weather)

#     # Run EOF analysis on the xarray DataArrays
#     print("\nRunning EOF analysis on ocean and weather data...")
#     df_ocean_pc = run_eof_reduction(ds_ocean, thresh=0.9)
#     df_weather_pc = run_eof_reduction(ds_weather, thresh=0.9)

#     # Convert the DataFrames with PCs back to DataFrames
#     print("\nConverting DataArrays with PCs back to DataFrames...")
#     merged_pc = pd.merge(df_ocean_pc, df_weather_pc, on='time', how='outer').dropna()
    
#     # Merge the PCs with the insitu data
#     print("\nMerging PCs with insitu data...")
#     df_merged = pd.merge(merged_pc, df_insitu, on='time', how='outer').dropna()
#     print(f"\nShape of merged DataFrame with PCs: {df_merged.shape}")

#     # df_merged['is_surge'] = df_merged['slev'] > 1.0
#     # df_merged['is_surge'] = df_merged['is_surge'].astype(bool)
    

#     #################################### TABULAR DATA ###################################
#     # print("\nConverting DataFrames to tables...")
#     # df_ocean_table = convert_df_to_table(df_ocean)
#     # df_weather_table = convert_df_to_table(df_weather)

#     # print("\nMerging DataFrames...")
#     # df_merged = merge_dataframes([df_ocean_table, df_weather_table, df_insitu])
#     # print(f"Shape of merged DataFrame: {df_merged.shape}")

#     print("\nPreparing ML data...")
#     df_merged = prepare_ml_data(df_merged)

#     print("\nSum of NaN values in each column:")
#     print(df_merged.isna().sum())

#     df_merged = df_merged.interpolate(method="linear")
#     df_merged = df_merged.replace([np.inf, -np.inf], np.nan).dropna() # remove inf values
#     print(f"Shape of prepared ML DataFrame: {df_merged.shape}")

#     print("\nColumns in the prepared DataFrame:")
#     print(df_merged.columns.tolist())

#     print("\nPrepared DataFrame for ML:")
#     print(df_merged.head())

#     print("\nData Info:")
#     print(df_merged.info())

#     # split data into train data
#     df_train = df_merged[df_merged['ds'] < '2025-01-01']

#     return df_train




def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trainiere Forecast-Modell mit Optuna-Hyperparameter-Tuning."
    )
    parser.add_argument("--n_trials", type=int, default=3, help="Anzahl der Optuna-Trials")
    parser.add_argument("--timeout", type=int, default=60 * 10, help="Zeitlimit in Sekunden")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "svr", "lgbm", "lr", "sarima"],
        help="Modelltyp für das Forecasting"
    )
    return parser.parse_args()

def get_param(trial, name, space):
    t = space["type"]
    if t == "int":
        return trial.suggest_int(name, int(space["low"]), int(space["high"]))
    elif t == "float":
        if space.get("log", False):
            return trial.suggest_float(name, float(space["low"]), float(space["high"]), log=True)
        else:
            return trial.suggest_float(name, float(space["low"]), float(space["high"]))
    elif t == "categorical":
        return trial.suggest_categorical(name, space["options"])



def load_search_space(model_name: str, base_path: Path) -> dict:
    """Lade den YAML-Suchraum je nach Modelltyp."""
    filename = {
        "random_forest": "rf_searchspace.yaml",
        "svr": "svr_searchspace.yaml",
        "xgboost": "xgb_searchspace.yaml",
        "lgbm": "lgbm_searchspace.yaml",
        "lr": "lr_searchspace.yaml",
        "sarima": "sarima_searchspace.yaml",
    }[model_name]
    file_path = base_path / filename
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def setup_optuna_study(study_name: str, storage_url: str) -> optuna.study.Study:
    """Erstelle oder lade die Optuna-Study mit Storage (SQLite)."""
    return optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True
    )






























def run_optuna_optimization(
    df: pd.DataFrame,
    search_space: dict,
    model_type: str,
    n_windows: int,
    horizon: int,
    step_size: int,
    n_trials: int,
    study: optuna.study.Study,
    timeout: int | None = None,
    
) -> None:
    """
    Definiere das Objective und führe die Optuna-Optimierung durch.
    """
    def objective(trial: optuna.trial.Trial) -> float:
        # Parameterauswahl
        params = {
            name: get_param(trial, name, space) if isinstance(space, dict) else space
            for name, space in search_space.items()
        }
        # Lags vorschlagen
        lag_options = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], 
                       [1, 2, 4, 6], [1, 2, 4, 6, 8], [1, 2, 4, 6, 8, 10], 
                       [1, 2, 4, 8, 12, 16, 24], [1, 2, 4, 8, 12, 16, 24, 48]]
        
        lags = trial.suggest_categorical("lags", lag_options)

        # Erstelle das Modell basierend auf dem Typ

        if model_type == "xgboost":
            params_xgb = params.copy()
            model_choice = XGBRegressor(**params_xgb)
        elif model_type == "lgbm":
            params_lgbm = params.copy()
            model_choice = LGBMRegressor(**params_lgbm)
        elif model_type == "random_forest":
            params_rf = params.copy()
            model_choice = RandomForestRegressor(**params_rf)
        elif model_type == "svr":
            params_svr = params.copy()
            model_choice = SVR(**params_svr)
        elif model_type == "lr":
                params_poly = params.copy()
                degree = params_poly.pop("degree")
                interaction_only = params_poly.pop("interaction_only")
                include_bias = params_poly.pop("include_bias")
                poly = PolynomialFeatures(
                    degree=degree,
                    interaction_only=interaction_only,
                    include_bias=include_bias
                )
                model_choice = LinearRegression(n_jobs=-1)
        elif model_type == "sarima":
            model_choice = None
            lags = None
        else:
            raise ValueError(f"Unbekannter Modelltyp: {model_type}")
           

        # If PolynomialFeatures is used, add it to the pipeline
        if model_type in ["lr"]:
            pipe = Pipeline([
                             #("poly", poly), 
                             ("scaler", StandardScaler()), 
                             ("model", model_choice)])
            
        elif model_type in ["sarima"]:
            # SARIMA benötigt keine Pipeline, da es direkt auf den Zeitreihen arbeitet
            pipe = model_choice
        else:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model_choice)])

        

        if model_type == "sarima":
            # Use just the columns ['ds', 'unique_id', 'y'] for SARIMA
            df_sarima = df[["ds", "unique_id", "y"]]

            model = StatsForecast(
                    models=[ARIMA(
                            order=(params["p"], params["d"], params["q"]), 
                            seasonal_order=(params["P"], params["D"], params["Q"]),
                            season_length=params["s"])],
                    freq="h",
                    n_jobs=-1
                )
            
            try:
                results = model.cross_validation(
                    #model=model,
                    df=df_sarima,
                    n_windows=n_windows,
                    h=horizon,
                    step_size=step_size,
                    test_size=horizon
                    #static_features=[],
                    #keep_last_n=24 * 7
                )
            except Exception as e:
                print(f"Fehler bei der Cross-Validation mit lags={lags}: {e}")
                return np.inf
        else:
            # exogene Features transformieren und entferne y_lag-Spalten um data leakage zu vermeiden
            df_trans = transform_exog(df, lags=lags, num_threads=-1)
            cols_to_remove = [col for col in df_trans.columns if col.startswith('y_lag')]
            df_trans = df_trans.drop(columns=set(cols_to_remove))

            # Model initialisieren
            model = MLForecast(models=[pipe], freq="h", lags=lags)
            try:
                results = model.cross_validation(
                    df=df_trans,
                    n_windows=n_windows,
                    h=horizon,
                    step_size=step_size,
                    static_features=[],
                    keep_last_n=HORIZON
                )
            except Exception as e:
                print(f"Fehler bei der Cross-Validation mit lags={lags}: {e}")
                return np.inf
        #print(f"\nErgebnisse der Cross-Validation mit lags={lags}:")
        #print(results.head())

        # Manuelle Berechnung von Recall und MSE:
        pred_cols = [c for c in results.columns if c not in ["unique_id", "ds", "cutoff", "y"]]
        if len(pred_cols) != 1:
            raise ValueError(f"Erwartet genau eine Vorhersage-Spalte, gefunden: {pred_cols}")
        pred_col = pred_cols[0]

        y_true = results["y"].values
        y_hat = results[pred_col].values

        # Klassifikation durch Schwellwert (z.B. threshold=1.0) und MSE
        alpha = 0.7

        bins = [1, 1.25, 1.5, 2.00]


        # Ordne die Werte in Klassen ein
        y_true_class = np.digitize(y_true, bins)   # Werte ≤1.5→0, (1.5,2.5]→1, (2.5,3.5]→2, >3.5→3
        y_pred_class = np.digitize(y_hat, bins)

        if np.isnan(y_true).any() or np.isnan(y_hat).any():
            print(f"Trial {trial.number} abgebrochen: NaN in y_true oder y_hat.")
            return np.inf  # oder return 1e10, damit Optuna diesen Trial verwirft


        # y_true_class = (y_true > threshold).astype(int)
        # y_pred_class = (y_hat > threshold).astype(int)
        recall = recall_score(y_true_class, y_pred_class, average="macro")
        mse = mean_squared_error(y_true, y_hat)

        # df_classes = pd.DataFrame({
        #     "y_true": y_true,
        #     "y_hat": y_hat,
        #     "y_true_class": y_true_class,
        #     "y_pred_class": y_pred_class
        # })

        # df_classes = df_classes.where(df_classes['y_true'] > 1.0).dropna()
        #print(df_classes)

        combined_score = alpha * (1 - recall) + (1 - alpha) * mse # kombinierter Score

        # Optional: kurz ausgeben, damit du in der Konsole siehst, was gerade passiert
        print(f"\n{trial} mit lags={lags}: Recall={recall:.4f}, MSE={mse:.4f} → Score={combined_score:.4f}")

        return combined_score
        

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        n_jobs=1
    )























def main():
    args = parse_arguments()

    # Modelltyp ausgeben
    print(f"\nModell ausgewählt: {args.model}")

    # Daten laden (hier als Beispiel Testdaten)
    #df = load_test_data(periods=20160)
    df = load_data(split=True)
    # df = load_data_v2()

    # Parametriere die Cross-Validation-Window-Logik
    len_df = len(df)
    initial_train_window = 24 * 30 * 6   # 6 Monate
    step_size = HORIZON                   # 7 Tage
    horizon = HORIZON                     # 7 Tage
    #horizon = 24 * 90                     # 90 Tage zum testen
    n_windows = (len_df - initial_train_window) // horizon
    print(f"Anzahl der Trainingsfenster: {n_windows}")

    # Suche den Hyperparameterraum
    yaml_base = Path("/gxfs_home/geomar/smomw693/Documents/Masterarbeit_code/code/utils")
    search_space = load_search_space(args.model, yaml_base)
    print("\nGeladener Suchraum:")
    pprint(search_space)

    # Optuna‐Study initialisieren
    if args.model == "xgboost":
        STORAGE_URL = "sqlite:///./Versuch1.1_optuna_study_ML_xgboost.db"
        STUDY_NAME = f"XGBOOST_points{OCEAN_POINTS}_h{horizon}_ml_test"
    elif args.model == "lgbm":
        STORAGE_URL = "sqlite:///./Versuch1.1_optuna_study_ML_lgbm.db"
        STUDY_NAME = f"LGBM_points{OCEAN_POINTS}_h{horizon}_ml_test"
    elif args.model == "random_forest":
        STORAGE_URL = "sqlite:///./Versuch1.1_optuna_study_ML_rf.db"
        STUDY_NAME = f"RF_points{OCEAN_POINTS}_h{horizon}_ml_test"
    elif args.model == "svr":
        STORAGE_URL = "sqlite:///./Versuch1.1_optuna_study_ML_svr.db"
        STUDY_NAME = f"SVR_points{OCEAN_POINTS}_h{horizon}_ml_test"
    elif args.model == "lr":
        STORAGE_URL = "sqlite:///./Versuch1.1_optuna_study_ML_lr.db"
        STUDY_NAME = f"LR_points{OCEAN_POINTS}_h{horizon}_ml_test"
    elif args.model == "sarima":
        STORAGE_URL = "sqlite:///./Versuch1.1_optuna_study_ML_sarima.db"
        STUDY_NAME = f"SARIMA_points{OCEAN_POINTS}_h{horizon}_ml_test"

    study = setup_optuna_study(STUDY_NAME, STORAGE_URL)
    run_optuna_optimization(
        df=df,
        search_space=search_space,
        model_type=args.model,
        n_windows=n_windows,
        horizon=horizon,
        step_size=step_size,
        n_trials=args.n_trials,
        #timeout=args.timeout,
        study=study
    )

    # Ergebnisse ausgeben
    print("\nOptuna-Studie abgeschlossen.")
    print("\nBeste Parameter:")
    pprint(study.best_params)
    print("Bestes Score (kleiner ist besser):", study.best_value)


if __name__ == "__main__":
    main()
