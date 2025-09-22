#!/usr/bin/env python3

import argparse

# import all necessary libraries
import warnings
from pathlib import Path
from pprint import pprint
from sklearn.metrics import recall_score, mean_squared_error
import numpy as np
import yaml
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    convert_df_to_table,
    feature_engineering,
    merge_dataframes,
    eof_solver,
    run_eof_reduction
)
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse
from xgboost import XGBRegressor



# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

#OCEAN_POINTS = 5
#OCEAN_POINTS = 10
#OCEAN_POINTS = 20
#OCEAN_POINTS = 30


# Display all columns
#pd.options.display.max_columns = None
STORAGE_URL = "sqlite:///./optuna_study_ML_custom_function.db"
STUDY_NAME = f"XGBOOST_points{OCEAN_POINTS}_ml"
STUDY_NAME = "test"

warnings.filterwarnings("ignore", category=UserWarning)




def convert_to_xarray(df):
    """
    Convert a DataFrame to an xarray DataArray.
    """
    df['time'] = pd.to_datetime(df['time'])

    # Optional: Setze einen MultiIndex, falls du nach Zeit, Breite, Länge strukturieren willst
    df = df.set_index(['time', 'latitude', 'longitude'])

    # In ein xarray.Dataset umwandeln:
    ds = df.to_xarray()

    return ds

def prepare_ml_data(df):
    df = df.copy()
    df = df.rename(columns={'slev': 'y', 'time': 'ds'})
    df['unique_id'] = 'Flensburg'
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S')
    return df

def load_test_data(periods=20160):
    df = pd.DataFrame({
        'time': pd.date_range(start='2022-12-03', periods=periods, freq='h')
    })
    df['time'] = pd.to_datetime(df['time'])
    t = np.arange(periods)
    sin_day = np.sin(2 * np.pi * t / 24)
    cos_day = np.cos(2 * np.pi * t / 24)
    sin_week = np.sin(2 * np.pi * t / 168)
    cos_week = np.cos(2 * np.pi * t / 168)
    sin_month = np.sin(2 * np.pi * t / 720)

    df['slev'] = (
        5 + 3 * sin_day + 1.5 * cos_week + np.random.normal(scale=0.2, size=periods)
    )
    df['feature1'] = 50 + 20 * sin_day + 5 * cos_day
    df['feature2'] = 30 + 10 * sin_day + 10 * sin_week
    df['feature3'] = 10 + 5 * cos_week + 3 * sin_month + np.random.normal(scale=0.5, size=periods)
    df['feature4'] = 20 + 15 * sin_week + np.random.normal(scale=0.3, size=periods)
    df['feature5'] = 10 + 8 * sin_day + np.random.normal(scale=0.2, size=periods)
    df['feature6'] = 15 + 12 * sin_week + np.random.normal(scale=0.3, size=periods)


    df = prepare_ml_data(df)

    return df




def load_data():
    ocean_data_path = Path(f"../data/numerical_data/points{OCEAN_POINTS}")
    weather_data_path = Path(f"../data/numerical_data/points{WEATHER_POINTS}")

    print("\nLoading data from:")
    print(ocean_data_path)
    print(weather_data_path)

    # save df_merged to ../data/tabular_data_FI/
    # file_name = f'df_merged{OCEAN_POINTS}_FI.tsv'
    # output_path = Path('../data/tabular_data_FI/')

    print("\nLoading ocean, weather and insitu data...")
    df_ocean = load_ocean_data(ocean_data_path, OCEAN_POINTS, verbose=False)
    df_ocean = process_df(df_ocean, drop_cols=["depth"], verbose=False)

    df_weather = load_weather_data(weather_data_path, WEATHER_POINTS, verbose=False)
    df_weather = process_df(df_weather, verbose=False, drop_cols=['showers'])

    df_insitu = load_insitu_data(verbose=False)
    df_insitu = process_flensburg_data(df_insitu, 
                                        start_time=df_ocean['time'].min(),
                                        end_time=df_ocean['time'].max(),
                                        verbose=False)

    df_insitu = group_data_hourly(df_insitu)
    df_insitu = process_df(df_insitu, drop_cols=["depth",'deph', 'latitude', 'longitude', 'time_qc', 'slev_qc'], verbose=False)



    print("\nFeature engineering...")
    df_ocean, df_weather, df_insitu = feature_engineering(df_ocean, df_weather, df_insitu)

    print(f"\nShapes of loaded DataFrames:\n"
          f"Ocean DataFrame: {df_ocean.shape}\n"
          f"Weather DataFrame: {df_weather.shape}\n"
          f"Insitu DataFrame: {df_insitu.shape}")

    # Convert DataFrames to xarray DataArrays
    print("\nConverting DataFrames to xarray DataArrays...")
    ds_ocean = convert_to_xarray(df_ocean)
    ds_weather = convert_to_xarray(df_weather)

    # Run EOF analysis on the xarray DataArrays
    print("\nRunning EOF analysis on ocean and weather data...")
    df_ocean_pc = run_eof_reduction(ds_ocean, thresh=0.9)
    df_weather_pc = run_eof_reduction(ds_weather, thresh=0.9)

    # Convert the DataFrames with PCs back to DataFrames
    print("\nConverting DataArrays with PCs back to DataFrames...")
    merged_pc = pd.merge(df_ocean_pc, df_weather_pc, on='time', how='outer').dropna()
    
    # Merge the PCs with the insitu data
    print("\nMerging PCs with insitu data...")
    df_merged = pd.merge(merged_pc, df_insitu, on='time', how='outer').dropna()
    print(f"\nShape of merged DataFrame with PCs: {df_merged.shape}")

    # df_merged['is_surge'] = df_merged['slev'] > 1.0
    # df_merged['is_surge'] = df_merged['is_surge'].astype(bool)
    

    #################################### TABULAR DATA ###################################
    # print("\nConverting DataFrames to tables...")
    # df_ocean_table = convert_df_to_table(df_ocean)
    # df_weather_table = convert_df_to_table(df_weather)

    # print("\nMerging DataFrames...")
    # df_merged = merge_dataframes([df_ocean_table, df_weather_table, df_insitu])
    # print(f"Shape of merged DataFrame: {df_merged.shape}")

    print("\nPreparing ML data...")
    df_merged = prepare_ml_data(df_merged)
    print(f"Shape of prepared ML DataFrame: {df_merged.shape}")


    return df_merged





def add_pca(model, df, n_components=0.99):
    df_preprocessed = model.preprocess(df, static_features=[])
    features_excluded = ['unique_id', 'ds', 'y']
    df_pca = df_preprocessed.drop(columns=features_excluded)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    df_pca['unique_id'] = df_preprocessed['unique_id']
    df_pca['ds'] = df_preprocessed['ds']
    df_pca['y'] = df_preprocessed['y']
    df_pca = df_pca.dropna()
    return df_pca

def model_stats(results):
    model_cols = [col for col in results.columns if col not in ['unique_id', 'ds', 'cutoff', 'y']]
    stats = []
    for model in model_cols:
        rmse_val = np.sqrt(np.mean((results['y'] - results[model])**2))
        mse = np.mean((results['y'] - results[model])**2)
        mape = np.mean(np.abs((results['y'] - results[model]) / results['y'])) * 100
        r = np.corrcoef(results['y'], results[model])[0, 1]
        stats.append({
            'model': model,
            'rmse': rmse_val,
            'mse': mse,
            'mape': mape,
            'r': r
        })
    return pd.DataFrame(stats)



def custom_combined_score(
    models: dict,        # HIER die neue Signatur - annotate als dict
    y: np.ndarray,
    y_hat: np.ndarray,
    alpha: float = 0.7,
    threshold: float = 1.0
) -> float:
    """
    Berechnet einen kombinierten Score aus Recall (für y > threshold) und MSE.
    Sinnvoll, um Extremwerte (Sturmflut, etc.) extra zu gewichten.

    models: dict
        Hier erwartet evaluate üblicherweise etwas – wir ignorieren es intern einfach.
    y: Array_like
        Ground-Truth-Werte (Observationen).
    y_hat: Array_like
        Prognostizierte Werte (Predictions).
    alpha: float, default=0.7
        Gewichtung zwischen Recall (1-Recall) und MSE.
    threshold: float, default=1.0
        Ab welchem Pegel (z. B. Sturmflut) wir in die Klassifikations-Phase gehen.
    """
    # Da 'models' nicht wirklich benötigt wird, ignorieren wir es:
    _ = models

    # Klassifikation durch Schwellwert
    y_true_class = (y > threshold).astype(int)
    y_pred_class = (y_hat > threshold).astype(int)
    recall = recall_score(y_true_class, y_pred_class)
    mse = mean_squared_error(y, y_hat)
    score = alpha * (1 - recall) + (1 - alpha) * mse

    print(f"\nCustom score: {score} (Recall: {recall}, MSE: {mse})")
    return score



def get_param(trial, name, space):
    t = space["type"]
    if t == "int":
        return trial.suggest_int(name, space["low"], space["high"])
    elif t == "float":
        if space.get("log", False):
            return trial.suggest_float(name, space["low"], space["high"], log=True)
        else:
            return trial.suggest_float(name, space["low"], space["high"])
    elif t == "categorical":
        return trial.suggest_categorical(name, space["options"])




def main():
    # ==== 1) Argumente parsen ====
    parser = argparse.ArgumentParser(description="Trainiere Forecast-Modell mit Optuna-Hyperparameter-Tuning.")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=3,
        help="Anzahl der Optuna-Trials (Standard: 100)."
    )

    # Argument um Timeout zu setzen
    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 10,
        help="Zeitlimit für Optuna-Studie in Sekunden (Standard: 600)."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "svr"],
        help="Modelltyp für das Forecasting (Standard: 'random_forest')."
    )

    args = parser.parse_args()

    if args.model == "xgboost":
        print("\nXGBoost-Modell ausgewählt.")
    elif args.model == "random_forest":
        print("\nRandom Forest-Modell ausgewählt.")
    elif args.model == "svr":
        print("\nSupport Vector Regression-Modell ausgewählt.")



    print("\nStarte Forecast-Modell mit Optuna-Hyperparameter-Tuning...")
    # ==== 2) Daten laden und vorbereiten ====
    print("Lade Testdaten ")
    df = load_test_data(periods=20160)   # hier wird args.periods verwendet
    #df = load_data()



    len_df = len(df)
    initial_train_window = 24 * 30 * 6   # 6 Monate
    step_size = 24 * 7                  # 7 Tage
    horizon = 24 * 7                    # 7 Tage
    n_windows = (len_df - initial_train_window) // horizon

    print(f"\nAnzahl der Trainingsfenster: {n_windows}")
    # ==== 3) Optuna-Objective definieren ====




    # load search space from yaml file
    if args.model == "random_forest":
        print("\nLade Hyperparameter-Suchraum für Random Forest...")
        with open("/gxfs_home/geomar/smomw693/Documents/GEOMAR-DeepLearning/code/utils/rf_searchspace.yaml", "r") as f:
            searchspace = yaml.safe_load(f)
    elif args.model == "svr":
        print("\nLade Hyperparameter-Suchraum für SVR...")
        with open("/gxfs_home/geomar/smomw693/Documents/GEOMAR-DeepLearning/code/utils/svr_searchspace.yaml", "r") as f:
            searchspace = yaml.safe_load(f)
    elif args.model == "xgboost":
        print("\nLade Hyperparameter-Suchraum für XGBoost...")
        with open("/gxfs_home/geomar/smomw693/Documents/GEOMAR-DeepLearning/code/utils/xgb_searchspace.yaml", "r") as f:
            searchspace = yaml.safe_load(f)

    print("\nHyperparameter-Suchraum aus YAML-Datei:")
    print(searchspace)

    def objective(trial):

        params = {}
        for name, space in searchspace.items():
            if isinstance(space, dict):
                params[name] = get_param(trial, name, space)
            else:
                params[name] = space  # Konstante


        lag_options = [
            [1],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 6, 24],
        ]
        lags = trial.suggest_categorical("lags", lag_options)

        if args.model == "random_forest":
            pipe = Pipeline([
                ('model', RandomForestRegressor(**params))
            ])
        elif args.model == "svr":
            
            pipe = Pipeline([
                ('model', SVR(**params))
            ])
        elif args.model == "xgboost":
            pipe = Pipeline([
                ('model', XGBRegressor(**params))
            ])

        model = MLForecast(
            models=[pipe],
            freq="h",
            lags=lags
        )
        #print("\nAdd PCA to model...")
        #df_pca = add_pca(model, df)

        print(f"\nFitting model... with lags: {lags}:")
        results = model.cross_validation(
            df=df,
            n_windows=n_windows,
            h=horizon,
            step_size=step_size,
            static_features=[]
        )

        score = evaluate(
            results.drop(columns='cutoff'),
            metrics=[custom_combined_score],
            agg_fn='mean',
        ).drop('metric', axis=1).iloc[0, 0]

        return score





    # ==== 4) Optuna Study starten ====
    print("\nStarte Optuna-Studie...")

    #study_name = f"test"
    study = optuna.create_study(direction='minimize', study_name=STUDY_NAME, storage=STORAGE_URL, load_if_exists=True)
    study.optimize(
        objective,
        n_trials=args.n_trials,      # hier wird args.n_trials verwendet
        show_progress_bar=True,
        timeout=args.timeout,             # 2 Stunden Zeitlimit
        n_jobs=1
    )




    print("\nOptuna-Studie abgeschlossen.")

    # ==== 5) Ergebnisse ausgeben/speichern ====
    print("\nBeste Ergebnisse:")
    pprint(study.best_params)
    print("Bestes RMSE:", study.best_value)


if __name__ == "__main__":
    main()
