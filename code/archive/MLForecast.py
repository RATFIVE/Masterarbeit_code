# Standardbibliotheken
import os
import warnings
from pathlib import Path

# Drittanbieter-Bibliotheken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
from mlforecast import MLForecast

# Eigene Module
from utils.eda_helper_functions import (
    group_data_hourly,
    load_insitu_data,
    load_ocean_data,
    load_weather_data,
    process_df,
    process_flensburg_data,
)
from utils.config import (
    OCEAN_POINTS,
    WEATHER_POINTS,
)

# Einstellungen
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "figure.labelsize": 14,
    "savefig.dpi": 300,
    "figure.dpi": 100,
})


def feature_engineering(df_ocean, df_weather, df_insitu):
    """
    Berechnet u- und v-Komponenten des Windes und entfernt unnötige Spalten.
    """
    theta = np.deg2rad(df_weather['wind_direction_10m'])
    df_weather['wind_u'] = df_weather['wind_speed_10m'] * np.sin(theta)
    df_weather['wind_v'] = df_weather['wind_speed_10m'] * np.cos(theta)
    df_weather.drop(columns=['wind_speed_10m', 'wind_direction_10m'], inplace=True)
    return df_ocean, df_weather, df_insitu


def convert_df_to_table(df: pd.DataFrame) -> pd.DataFrame:
    df['position'] = df.apply(lambda row: (row['latitude'], row['longitude']), axis=1)
    coordinates = df['position'].unique()
    df_merged = pd.DataFrame({'time': df['time'].unique()})
    for i in tqdm(range(len(coordinates)), desc="Processing coordinates", unit="coord", total=len(coordinates)):
        df_sub_data = df[df['position'] == coordinates[i]].drop(columns=['latitude', 'longitude'])
        cols = df_sub_data.columns.tolist()
        cols.remove('position')
        cols.remove('time')
        for col in cols:
            df_sub_data.rename(columns={col: col + '#' + str(coordinates[i])}, inplace=True)
        df_sub_data = df_sub_data.drop(columns='position')
        df_merged = df_merged.merge(df_sub_data, on='time')
    return df_merged


def merge_dataframes(dfs: list) -> pd.DataFrame:
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='time', how='inner')
    merged_df['time'] = pd.to_datetime(merged_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    return merged_df


def prepare_ml_data(df):
    df = df.copy()
    df = df.rename(columns={'slev': 'y', 'time': 'ds'})
    df['unique_id'] = 'Flensburg'
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S')
    return df


def train_test_split(df, n_future):
    train = df[:-n_future]
    test = df[-n_future:]
    actual = df[-n_future:]
    test = test.drop(columns=['y'])
    return train, test, actual


def build_mlforecast_model(lags=None, date_features=None, model_params=None):
    if lags is None:
        lags = []
    if model_params is None:
        model_params = {"n_jobs": -1, "random_state": 42}
    model = MLForecast(
        models=[RandomForestRegressor(**model_params)],
        freq='h',
        lags=lags,
        date_features=date_features if date_features else None
    )
    return model


def plot_forecast(original_df, pred_df, actual, ts_id='Flensburg'):
    df_plot = original_df[original_df['unique_id'] == ts_id]
    pred_plot = pred_df[pred_df['unique_id'] == ts_id]
    plt.figure(figsize=(12, 5))
    plt.plot(df_plot['ds'].iloc[-24*7:], df_plot['y'].iloc[-24*7:], label='Historie')
    plt.plot(pred_plot['ds'], pred_plot['RandomForestRegressor'], label='Prognose', linestyle='--')
    plt.plot(actual['ds'], actual['y'], label='Tatsächliche Werte', linestyle='--', color='red')
    plt.legend()
    plt.title(f'Forecast für {ts_id}')
    plt.grid()
    plt.tight_layout()
    plt.show()


def optuna_objective(trial, df, tscv, lags=None):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": 42,
    }
    rmses = []
    for train_idx, val_idx in tscv.split(df):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        forecast = MLForecast(
            models=[RandomForestRegressor(**params)],
            freq='h',
            lags=lags if lags else [],
        )
        forecast.fit(df_train, static_features=[])
        horizon = len(df_val)
        preds = forecast.predict(horizon, X_df=df_val)
        y_true = df_val['y'].values
        y_pred = preds['RandomForestRegressor'].values[:horizon]
        rmse = mean_squared_error(y_true, y_pred)
        rmses.append(rmse)
    return np.mean(rmses)





def train_final_model(df, study, n_future, lags=None):
    best_model = RandomForestRegressor(**study.best_params)
    final_forecast = MLForecast(
        models=[best_model],
        freq='h',
        lags=lags if lags else [],
    )
    tscv = TimeSeriesSplit(n_splits=3, test_size=n_future)
    for train_idx, val_idx in tscv.split(df):
        pass
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[val_idx].copy()
    forecast_horizon = len(df_test)
    final_forecast.fit(df_train, static_features=[])
    final_preds = final_forecast.predict(forecast_horizon, X_df=df_test)
    return final_preds, df_test


def load_data():

    # file paths
    ocean_data_path = Path(f"../data/numerical_data/points{OCEAN_POINTS}")
    weather_data_path = Path(f"../data/numerical_data/points{WEATHER_POINTS}")

    # load and process data
    df_ocean = load_ocean_data(ocean_data_path, OCEAN_POINTS, verbose=False)
    df_ocean = process_df(df_ocean, drop_cols=["depth"], verbose=False)
    df_weather = load_weather_data(weather_data_path, WEATHER_POINTS, verbose=False)
    df_weather = process_df(df_weather, verbose=False, drop_cols=['showers'])
    df_insitu = load_insitu_data(verbose=False)
    df_insitu = process_flensburg_data(df_insitu, start_time=df_ocean['time'].min(), end_time=df_ocean['time'].max(), verbose=False)
    df_insitu = group_data_hourly(df_insitu)
    df_insitu = process_df(df_insitu, drop_cols=["depth",'deph', 'latitude', 'longitude', 'time_qc', 'slev_qc'], verbose=False)

    # Feature Engineering
    df_ocean, df_weather, df_insitu = feature_engineering(df_ocean, df_weather, df_insitu)

    # Convert DataFrames to tables and merge
    df_ocean_table = convert_df_to_table(df_ocean)
    df_weather_table = convert_df_to_table(df_weather)
    df_merged = merge_dataframes([df_ocean_table, df_weather_table, df_insitu])
    

    return df_merged


def run_optuna(df, n_future, n_trials=3, lags=None, storage_url="sqlite:///./optuna_study.db", study_name=None):
    tscv = TimeSeriesSplit(n_splits=3, test_size=n_future)
    study = optuna.create_study(
            study_name=study_name,     # optional
            storage=storage_url,             # hier die DB-Verbindung
            direction="minimize",  # Minimierung des RMSE
            load_if_exists=True              # falls bereits vorhanden, weiterführen
        )
    study.optimize(lambda trial: optuna_objective(trial, df, tscv, lags=lags), n_trials=n_trials)
    print("Beste Parameter:")
    print(study.best_params)
    return study

def load_test_data(periods=48):
    """"""

    df = pd.DataFrame({
        'time': pd.date_range(start='2023-01-01', periods=periods, freq='H'),
        'slev': np.random.rand(periods) * 10 + 5,  # Simulierte Pegelstände
        'feature1': np.random.rand(periods) * 100,  # Simulierte Wetterdaten
        'feature2': np.random.rand(periods) * 50,   # Simulierte Wetterdaten
        'feature3': np.random.rand(periods) * 20,   # Simulierte Wetterdaten
        'unique_id': 'Flensburg',
    })
    df['ds'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    return df


def main():

    # Daten laden
    #df_merged = load_data()
    df_merged = load_test_data(periods=1000)  # Beispielhafte Testdaten



    # Prepare ML data by renaming columns and setting unique_id 
    df = prepare_ml_data(df_merged)


    n_future = 24 # Define the number of future time steps to predict
    train, test, actual = train_test_split(df, n_future)
    print("Train set shape:", train.shape)
    print("Test set shape:", test.shape)

    
    # Modell building and training
    model = build_mlforecast_model(model_params={"n_jobs": -1, "random_state": 42})
    model.fit(train, static_features=[])

    # 
    predictions = model.predict(h=n_future, X_df=test)
    print(predictions.head())
    
    # Optuna
    # Storage-URL für SQLite
    storage_url = "sqlite:///./optuna_study.db"
    #study_name = f"water_level_rf_points{OCEAN_POINTS}_mlforecast"
    study_name = 'test'
    study = run_optuna(df, n_future, n_trials=3, lags=[1,2,3], storage_url=storage_url, study_name=study_name)


if __name__ == "__main__":
    main()



