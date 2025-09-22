
#!/usr/bin/env python3

# import all necessary libraries
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof
from tqdm import tqdm


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


# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore")

from utils.config import (
    INSITU_DICT,
    OCEAN_DICT,
    WEATHER_DICT,
)

try:
    from config import (
        INSITU_DICT,
        OCEAN_DICT,
        WEATHER_DICT,
    )
except ImportError:
    pass

# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)




def convert_df_to_table(df: pd.DataFrame) -> pd.DataFrame:


    df['position'] = df.apply(lambda row: (row['latitude'], row['longitude']), axis=1)
    coordinates = df['position'].unique()

    df_merged = pd.DataFrame({'time': df['time'].unique()})
    for i in tqdm(range(len(coordinates)), desc="Processing coordinates", unit="coord", total=len(coordinates)):

        df_sub_data = df[df['position'] == coordinates[i]]
        df_sub_data = df_sub_data.drop(columns=['latitude', 'longitude'])

        cols = df_sub_data.columns.tolist()
        cols.remove('position')
        cols.remove('time')


        for col in cols:
            df_sub_data.rename(columns={col: col + '#' + str(coordinates[i])}, inplace=True)

        df_sub_data = df_sub_data.drop(columns='position')


        df_merged = df_merged.merge(df_sub_data, on='time')
        
    return df_merged



def merge_dataframes(dfs: list) -> pd.DataFrame:
    """
    Merge multiple DataFrames on the 'time' column.
    """
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='time', how='inner')

    merged_df['time'] = pd.to_datetime(merged_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    return merged_df


def feature_engineering(df_ocean:pd.DataFrame | None, df_weather:pd.DataFrame, df_insitu:pd.DataFrame, eda=False, light_mode=False):
    """
    Funktion zur Berechnung der u- und v-Komponenten des Windes aus der Windgeschwindigkeit und -richtung.
    """
    
    # df_weather


    #Calculate Radial Wind Speed
    theta = np.deg2rad(df_weather['wind_direction_10m'])

    #Calculate u and v components
    #    Definition der u- und v-Komponenten:
    #    u = Windgeschwindigkeit * sin(Windrichtung)
    #    v = Windgeschwindigkeit * cos(Windrichtung)
    df_weather['wind_u'] = df_weather['wind_speed_10m'] * np.sin(theta)  # positiver Wert = Wind nach Osten
    df_weather['wind_v'] = df_weather['wind_speed_10m'] * np.cos(theta)  # positiver Wert = Wind nach Norden

    # Remove the original wind speed and direction columns
    df_weather.drop(columns=['wind_speed_10m', 'wind_direction_10m'], inplace=True)

    # Remove unnecessary columns from df_ocean, df_weather, and df_insitu
    drop_cols = ['snowfall', 'siconc', 'rain', 'sithick', 'precipitation', 'weather_code', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_high', 'cloud_cover_mid', 'vapour_pressure_deficit', 'et0_fao_evapotranspiration', 'relative_humidity_2m', 'temperature_2m', 'dew_point_2m', 'apparent_temperature']
    drop_cols.append('mlotst') # drop mlotst from df_ocean, as it is correlated with sla
    drop_for_test = ["bottomT", "so", "sob", "thetao", "uo", "vo", "wo", "temperature_2m", 
                     "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", 
                     "rain", "snowfall", "weather_code", "surface_pressure", "cloud_cover", "cloud_cover_low", 
                     "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_gusts_10m"]

    # append drop_for_test to drop_cols
    if light_mode:
        drop_cols.extend(drop_for_test)

    print(f"\nDropping columns: {drop_cols}")
    if not eda:
        for col in drop_cols:
            if col in df_weather.columns:
                df_weather.drop(columns=col, inplace=True)
            if df_ocean is not None:  # Check if df_ocean is provided
                if col in df_ocean.columns:
                    df_ocean.drop(columns=col, inplace=True)
            if col in df_insitu.columns:
                df_insitu.drop(columns=col, inplace=True)
        

    
    print(f"\nShapes of DataFrames after dropping columns:\n"
          f"Ocean DataFrame: {df_ocean.shape if df_ocean is not None else 'N/A'}\n"
          f"Weather DataFrame: {df_weather.shape}\n"
          f"Insitu DataFrame: {df_insitu.shape}")
    return df_ocean, df_weather, df_insitu



def eof_solver(ds, feature, center=False, n_modes=10):

    """
    Perform EOF analysis on a given feature of an xarray Dataset.
    """
    # Maske für gültige Ozeanpunkte
    mask = ~np.isnan(ds[feature].isel(time=0))
    ds_cleaned = ds[feature].where(mask)

    # Flächengewichtung nach Breite (broadcast auf (lat, lon))
    lat_deg = ds_cleaned["latitude"].values  # (n_lat,)
    weights_2d = np.sqrt(np.cos(np.deg2rad(lat_deg)))[:, np.newaxis]  # (n_lat, 1)

    # Eof-Objekt erzeugen
    solver = Eof(ds_cleaned, weights=weights_2d, center=center)
    return solver

def run_eof_reduction(ds, thresh=0.9):
    """ Perform EOF analysis on all data variables in the dataset and return a DataFrame with PCs."""

    # data vars to list
    data_vars = list(ds.data_vars)
    df_merged = pd.DataFrame({
        'time': ds['time'].values  # Zeitstempel als Basis für die Zusammenführung
    })  # Leeres DataFrame für die Zusammenführung
    for var in data_vars:

        # --- (A) Solver initialisieren ---
        # ds_cleaned: dein vorverarbeiteter DataArray oder Dataset (time, lat, lon), bereits anomaliesiert
        # weights: 1D-Array über latitude für Flächengewichtung (z.B. sqrt(cos(lat)))
        solver = eof_solver(ds, var, center=True)

        # --- (B) Varianzanteile anschauen, um k zu wählen ---
        navors = 30
        var_frac = solver.varianceFraction(neigs=navors)
        cumvar = var_frac.cumsum(dim="mode")

        thresh = 0.90  # Schwellenwert für kumulative Varianz
        # z.B. suche k, um ≥95 % abzudecken:
        k = int((cumvar >= thresh).argmax().item() + 1)
        #print(f"Modi für ≥95 % Varianz: {k}")


        # --- (C) PCs berechnen ---
        pcs  = solver.pcs(npcs=k, pcscaling=1)    # (time:20184, mode:3)

        # --- (E) PCs als reduzierte Features extrahieren ---
        # In pandas DataFrame umwandeln, so dass jede Mode eine Spalte wird:
        # PCs direkt in DataFrame umwandeln und Spalten sinnvoll benennen
        df_pc = pcs.to_pandas()
        df_pc.columns = [f'{var}_PC_{i+1}' for i in range(df_pc.shape[1])]
        df_pc = df_pc.reset_index()  # 'time' als Spalte



        # Merge PCS to one dataframe
        df_merged = pd.merge(df_merged, df_pc, on='time', how='outer')
        
    
    return df_merged






def prepare_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereite das Insitu-DataFrame für das MLForecast-Modell vor.
    - Rename 'slev' → 'y'
    - Rename 'time' → 'ds'
    - Füge Spalte 'unique_id' hinzu

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mit Spalten ['time', 'slev', ...]

    Returns
    -------
    pd.DataFrame
        DataFrame mit Spalten ['ds', 'y', 'unique_id', ...]
    """
    df = df.copy()
    df = df.rename(columns={'slev': 'y', 'time': 'ds'})
    df['unique_id'] = 'Flensburg'
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S')
    return df



def convert_to_xarray(df: pd.DataFrame) -> xr.Dataset:
    """
    Wandelt ein DataFrame mit Spalten ['time', 'latitude', 'longitude', ...] in ein xarray Dataset um.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, das mindestens die Spalten 'time', 'latitude', 'longitude' enthält.

    Returns
    -------
    xr.Dataset
        Das resultierende xarray-Dataset mit MultiIndex (time, latitude, longitude).
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index(['time', 'latitude', 'longitude'])
    ds = df.to_xarray()
    return ds



def load_data(split=False):
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

    print(f"\nShapes of loaded DataFrames before feature engineering:\n"
          f"Ocean DataFrame: {df_ocean.shape}\n"
          f"Weather DataFrame: {df_weather.shape}\n"
          f"Insitu DataFrame: {df_insitu.shape}")

    print("\nFeature engineering...")
    df_ocean, df_weather, df_insitu = feature_engineering(df_ocean, df_weather, df_insitu)

    print(f"\nShapes of loaded DataFrames after feature engineering:\n"
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

    print(f"\nShapes of DataFrames with PCs:\n"
          f"Ocean PCs DataFrame: {df_ocean_pc.shape}\n"
          f"Weather PCs DataFrame: {df_weather_pc.shape}")

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

    print("\nSum of NaN values in each column:")
    print(df_merged.isna().sum())

    df_merged = df_merged.interpolate(method="linear")
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan).dropna() # remove inf values
    print(f"Shape of prepared ML DataFrame: {df_merged.shape}")

    print("\nColumns in the prepared DataFrame:")
    print(df_merged.columns.tolist())

    print("\nPrepared DataFrame for ML:")
    print(df_merged.head())

    print("\nData Info:")
    print(df_merged.info())

    # split data into train data
    if split:
        df_merged = df_merged[df_merged['ds'] < '2025-01-01']

    return df_merged



def load_data_v2(split=False):
    
    weather_data_path = Path(f"../data/numerical_data/points{WEATHER_POINTS}")

    print("\nLoading data from:")
    #print(ocean_data_path)
    print(weather_data_path)

    

    print("\nLoading weather and insitu data...")
  
    df_weather = load_weather_data(weather_data_path, WEATHER_POINTS, verbose=False)
    df_weather = process_df(df_weather, verbose=False, drop_cols=['showers'])

    df_insitu = load_insitu_data(verbose=False)
    df_insitu = process_flensburg_data(df_insitu, 
                                        start_time=df_weather['time'].min(),
                                        end_time=df_weather['time'].max(),
                                        verbose=False)

    df_insitu = group_data_hourly(df_insitu)
    df_insitu = process_df(df_insitu, drop_cols=["depth",'deph', 'latitude', 'longitude', 'time_qc', 'slev_qc'], verbose=False)

    print(f"\nShapes of loaded DataFrames before feature engineering:\n"
          f"Weather DataFrame: {df_weather.shape}\n"
          f"Insitu DataFrame: {df_insitu.shape}")

    print("\nFeature engineering...")
    df_ocean = pd.DataFrame()  # Placeholder for ocean data, if needed in future
    df_ocean, df_weather, df_insitu = feature_engineering(df_ocean, df_weather, df_insitu)

    print(f"\nShapes of loaded DataFrames after feature engineering:\n"
          f"Weather DataFrame: {df_weather.shape}\n"
          f"Insitu DataFrame: {df_insitu.shape}")

    # Convert DataFrames to xarray DataArrays
    print("\nConverting DataFrames to xarray DataArrays...")
    ds_weather = convert_to_xarray(df_weather)

    # Run EOF analysis on the xarray DataArrays
    print("\nRunning EOF analysis on ocean and weather data...")
    df_weather_pc = run_eof_reduction(ds_weather, thresh=0.9)

    # # Convert the DataFrames with PCs back to DataFrames
    # print("\nConverting DataArrays with PCs back to DataFrames...")
    # merged_pc = pd.merge(df_ocean_pc, df_weather_pc, on='time', how='outer').dropna()

    # Versuch 2: MERGE JUST df_weather_pc with df_inistu
    print("\nMerging PCs with insitu data...")
    df_merged = pd.merge(df_weather_pc, df_insitu, on='time', how='outer').dropna()
    print(f"\nShape of merged DataFrame with PCs: {df_merged.shape}")
    
    # # Merge the PCs with the insitu data
    # print("\nMerging PCs with insitu data...")
    # df_merged = pd.merge(merged_pc, df_insitu, on='time', how='outer').dropna()
    # print(f"\nShape of merged DataFrame with PCs: {df_merged.shape}")

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

    print("\nSum of NaN values in each column:")
    print(df_merged.isna().sum())

    df_merged = df_merged.interpolate(method="linear")
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan).dropna() # remove inf values
    print(f"Shape of prepared ML DataFrame: {df_merged.shape}")

    print("\nColumns in the prepared DataFrame:")
    print(df_merged.columns.tolist())

    print("\nPrepared DataFrame for ML:")
    print(df_merged.head())

    print("\nData Info:")
    print(df_merged.info())

    # split data into train data
    if split:
        df_merged = df_merged[df_merged['ds'] < '2025-01-01']

    return df_merged