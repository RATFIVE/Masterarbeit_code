import warnings
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from joblib import Parallel, delayed
from scipy.interpolate import griddata
from shapely.geometry import Point
from shapely.prepared import prep
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
    convert_to_xarray,
    feature_engineering,
)

torch.manual_seed(42)  # For reproducibility
np.random.seed(42)  # For reproducibility
# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Display all columns
pd.options.display.max_columns = None

writer = SummaryWriter(log_dir="runs/ConvLSTM")


OCEAN_POINTS = 30
GRID_SIZE = 5
HORIZON = 168


def load_picture_lagged_data(ocean_points=OCEAN_POINTS, weather_points=WEATHER_POINTS, return_common_time=False, grid_size=GRID_SIZE, verbose=False, n_jobs=1, dtype=np.float32, pca=False, keep_ocean_data=True, eda=False, lags=24, land_values=9999, light_mode=False):


    ocean_data_path = Path(f"../data/numerical_data/points{ocean_points}")
    weather_data_path = Path(f"../data/numerical_data/points{weather_points}")

    if verbose:
        print("\nLoading data from:")
        print(ocean_data_path)
        print(weather_data_path)
    

    # save df_merged to ../data/tabular_data_FI/
    # file_name = f'df_merged{OCEAN_POINTS}_FI.tsv'
    # output_path = Path('../data/tabular_data_FI/')
    if verbose:
        print("\nLoading ocean, weather and insitu data...")
    df_ocean = load_ocean_data(ocean_data_path, ocean_points, verbose=False)

    df_ocean = process_df(df_ocean, drop_cols=["depth"], verbose=False)

    df_weather = load_weather_data(weather_data_path, weather_points, verbose=False)
    df_weather = process_df(df_weather, verbose=False, drop_cols=["showers"])

    df_insitu = load_insitu_data(verbose=False)

    df_insitu = process_flensburg_data(df_insitu,
        start_time=df_ocean["time"].min(),
        end_time=df_ocean["time"].max(),
        verbose=False)


    df_insitu = group_data_hourly(df_insitu)
    df_insitu = process_df(df_insitu,
                        drop_cols=["depth", "deph", "latitude", "longitude", "time_qc", "slev_qc"],
                        verbose=False)
    if verbose:
        print("\nShapes of loaded DataFrames before feature engineering:")
        print(f"Ocean DataFrame: {df_ocean.shape}")
        print(f"Weather DataFrame: {df_weather.shape}")
        print(f"Insitu DataFrame: {df_insitu.shape}")
        
    if verbose:
        print("\nFeature engineering...")
    df_ocean, df_weather, df_insitu = feature_engineering(df_ocean, df_weather, df_insitu, eda=eda, light_mode=light_mode)
    if verbose:
        print("\nShapes of loaded DataFrames after feature engineering:\n")

        print(f"Ocean DataFrame: {df_ocean.shape}\n")
        print(f"Weather DataFrame: {df_weather.shape}\n")
        print(f"Insitu DataFrame: {df_insitu.shape}\n")

    if verbose:
        # Convert DataFrames to xarray DataArrays
        print("\nConverting DataFrames to xarray DataArrays...")


    ds_ocean = convert_to_xarray(df_ocean)
    ds_weather = convert_to_xarray(df_weather)



    def interpolate_time_step(da, lon_mesh, lat_mesh):
        # 1. Werte extrahieren
        lon, lat = np.meshgrid(da.longitude, da.latitude)
        lon_flat = lon.flatten()
        lat_flat = lat.flatten()
        values_flat = da.values.flatten()

        # 2. Nur gültige Werte
        mask = ~np.isnan(values_flat)
        if np.count_nonzero(mask) == 0:
            return np.full(lon_mesh.shape, np.nan)  # kein gültiger Punkt

        points = np.column_stack((lon_flat[mask], lat_flat[mask]))
        values = values_flat[mask]

        # 3. Interpolation
        interp_grid = griddata(points, values, (lon_mesh, lat_mesh), method="linear")
        return interp_grid


    def make_grid_parallel(ds, n_jobs=n_jobs, fill_na=False):
        time_vals = ds.time.values

        # Zielgrid
        lon_target = np.linspace(ds.longitude.min(), ds.longitude.max(), grid_size)
        lat_target = np.linspace(ds.latitude.min(), ds.latitude.max(), grid_size)
        lon_mesh, lat_mesh = np.meshgrid(lon_target, lat_target)

        variables_interp = {}

        for var_name in ds.data_vars:
            
            #print(f"Interpolating variable: {var_name}")

            da_var = ds[var_name]

            # Parallelisierung über Zeit
            interpolated_values = Parallel(n_jobs=n_jobs)(
                delayed(interpolate_time_step)(da_var.isel(time=t), lon_mesh, lat_mesh)
                for t in tqdm(range(len(time_vals)), desc=f"Interpolating {var_name}")
            )

            # In xarray schreiben
            variables_interp[var_name] = xr.DataArray(
                data=np.array(interpolated_values),
                dims=["time", "latitude", "longitude"],
                coords={"time": time_vals, "latitude": lat_target, "longitude": lon_target},
            )

        ds_interp = xr.Dataset(variables_interp)

        if fill_na:
            # Fülle verbleibende NaNs mit land_values
            return ds_interp.fillna(land_values)
        return ds_interp


    ds_weather_interp = make_grid_parallel(ds_weather)


    def is_ocean(latitudes, longitudes):
        """Gibt eine bool-Maske zurück, wo True = Ozean, False = Land"""
        land = cfeature.NaturalEarthFeature("physical", "land", "10m")
        geometries = list(land.geometries())
        land_geom = prep(geometries[0])

        mask = np.ones(latitudes.shape, dtype=bool)
        for i in range(latitudes.shape[0]):
            for j in range(latitudes.shape[1]):
                point = Point(longitudes[i, j], latitudes[i, j])
                if land_geom.contains(point):
                    mask[i, j] = False  # Land
        return mask


    def interpolate_single_timestep(da, lon_mesh, lat_mesh, ocean_mask, t):
        """Interpoliert eine Variable an einem Zeitschritt"""
        frame = da.isel(time=t)
        lon, lat = np.meshgrid(frame.longitude, frame.latitude)
        lon_flat = lon.flatten()
        lat_flat = lat.flatten()
        values_flat = frame.values.flatten()

        mask = ~np.isnan(values_flat)
        if np.count_nonzero(mask) == 0:
            return np.full(lon_mesh.shape, np.nan)

        points = np.column_stack((lon_flat[mask], lat_flat[mask]))
        values = values_flat[mask]

        interp_grid = griddata(points, values, (lon_mesh, lat_mesh), method="linear")

        # Landmaske anwenden
        interp_grid[~ocean_mask] = np.nan
        return interp_grid


    def make_grid_ocean_parallel(ds, n_jobs=n_jobs, fill_na=False):
        time_vals = ds.time.values

        # Zielgrid
        lon_target = np.linspace(ds.longitude.min(), ds.longitude.max(), grid_size)
        lat_target = np.linspace(ds.latitude.min(), ds.latitude.max(), grid_size)
        lon_mesh, lat_mesh = np.meshgrid(lon_target, lat_target)

        # Landmaske
        ocean_mask = is_ocean(lat_mesh, lon_mesh)

        variables_interp = {}

        for var_name in ds.data_vars:
            
            #print(f"Interpolating variable: {var_name}")

            da = ds[var_name]

            # Parallelisiere über Zeitachse
            results = Parallel(n_jobs=n_jobs)(
                delayed(interpolate_single_timestep)(da, lon_mesh, lat_mesh, ocean_mask, t)
                for t in tqdm(range(len(time_vals)), desc=f"Interpolating {var_name}")
            )

            variables_interp[var_name] = xr.DataArray(
                data=np.array(results),
                dims=["time", "latitude", "longitude"],
                coords={"time": time_vals, "latitude": lat_target, "longitude": lon_target},
            )

        if fill_na:
            ds_interp = xr.Dataset(variables_interp).fillna(land_values)
        else:
            ds_interp = xr.Dataset(variables_interp)
        return ds_interp


    ds_ocean_interp = make_grid_ocean_parallel(ds_ocean)


    # def plot_xarray(ds, variable, time_idx=0, cmap="viridis", cbar_label=None):
    #     """
    #     Plots a variable from an xarray Dataset for a specific time index.

    #     Parameters:
    #     - ds: xarray Dataset
    #     - variable: str, name of the variable to plot
    #     - time_idx: int, index of the time step to plot
    #     - cmap: str, colormap to use for the plot
    #     - cbar_label: str, label for the colorbar
    #     """
    #     data = ds[variable].isel(time=time_idx)

    #     # 1. Setze land_values → np.nan, damit sie nicht geplottet werden
    #     data = data.where(data < 9000)

    #     # 2. Plotten
    #     plt.figure(figsize=(8, 5))
    #     data.plot(x="longitude", y="latitude", cmap=cmap, cbar_kwargs={"label": cbar_label})

    #     plt.title(f"{variable} at {str(ds.time.values[time_idx])[:16]}")
    #     plt.xlabel("Longitude")
    #     plt.ylabel("Latitude")
    #     plt.show()

    if verbose:

        print("\nOcean DataArray:")
        print(ds_ocean_interp)
        print("\nWeather DataArray:")
        print(ds_weather_interp)


    
    # Zielgitter vom Ozean-Dataset
    time_target = ds_ocean_interp.time
    lat_target = ds_ocean_interp.latitude
    lon_target = ds_ocean_interp.longitude

    # Wetterdaten auf das Ocean-Gitter interpolieren
    ds_weather_interp_resampled = ds_weather_interp.interp(time=time_target, latitude=lat_target, longitude=lon_target, method="linear")

    # Merge beider Datasets auf dem gemeinsamen Gitter

    if keep_ocean_data:
        ds_combined = xr.merge([ds_ocean_interp, ds_weather_interp_resampled])
    else:
        ds_combined = ds_weather_interp

    # Füllwerte angleichen
    ds_combined = ds_combined.where(ds_combined < 9000).fillna(land_values)


    # idx = 0
    # plot_xarray(ds_combined, "sla", time_idx=idx, cmap="viridis", cbar_label="SLA (m)")
    # plot_xarray(ds_combined, "pressure_msl", time_idx=idx, cmap="coolwarm", cbar_label="Pressure (Pa)")

    df_insitu.rename(columns={"slev": "y"}, inplace=True)


    # calculate the lagged values for the insitu data
    # calculate laggs
    def calculate_lags(df: pd.DataFrame, lags):
        for lag in lags:
            df[f"y_lag_{lag}"] = df["y"].shift(lag)
        return df


    y_df_lagged = calculate_lags(df_insitu, lags=[i for i in range(1, lags + 1, 1)])

    y_df_lagged.dropna(inplace=True)


    # convert df to xarray
    y_df_lagged = y_df_lagged.set_index("time", drop=True).to_xarray()



    # Gemeinsame Zeitstempel
    common_time = np.intersect1d(ds_combined.time.values, y_df_lagged.time.values)

    # Auswahl
    X_ds = ds_combined.sel(time=common_time)
    y = y_df_lagged.sel(time=common_time)["y"]
    y_lagged = y_df_lagged.drop_vars("y").sel(time=common_time)

    if eda:
        return X_ds, y_lagged, y, common_time

    # if pca if True the reduce the dimensionality of X_ds using EOF analysis
    if pca:
        from utils.ml_helper_functions import run_eof_reduction
        df_pca = run_eof_reduction(X_ds)

        # turn df_pca to np.array
        print(df_pca.head())

        X = df_pca.drop(columns=["time"]).to_numpy()
        y_lagged = y_lagged.to_array().transpose("time", "variable").values
        y = y.values 
        return X, y_lagged, y, common_time


    # In Arrays umwandeln
    X = (X_ds.to_array().transpose("time", "variable", "latitude", "longitude").values)  
    y_lagged = y_lagged.to_array().transpose("time", "variable").values
    y = y.values  


    if verbose:
        print("\nShapes of the final datasets:")
        print(f"X shape: {X.shape}")
        print(f"y_lagged shape: {y_lagged.shape}")
        print(f"y shape: {y.shape}")
        
        if return_common_time:
            print(f"common_time shape: {common_time.shape}")
    if return_common_time:
        return X, y_lagged, y, common_time
    
    # turn X, y_lagged, y into float32 numpy arrays
    X = X.astype(dtype)
    y_lagged = y_lagged.astype(dtype)
    y = y.astype(dtype)

    return X, y_lagged, y




def create_sequences_test(X_data: np.ndarray, y_lagged_data: np.ndarray, y_data: np.ndarray, seq_len=168, horizon=168, dtype=np.float32):
    """
    Erzeugt nur Sequenzen, die innerhalb eines zusammenhängenden Bereichs liegen.

    Rückgabewerte:
    - X: np.ndarray, Form (n_sequences, seq_len, ...)
    - y_lagged: np.ndarray, Form (n_sequences, seq_len, ...)
    - y: np.ndarray, Form (n_sequences, horizon, ...)
    """
    if len(X_data) < seq_len + horizon:
        raise ValueError(f"Die Eingabedaten sind zu kurz: len(X_data)={len(X_data)}, erforderlich={seq_len + horizon}")

    X, y, y_lagged = [], [], []
    for i in range(len(X_data) - seq_len - horizon + 1):
        window_x = X_data[i : i + seq_len + horizon]
        window_y = y_data[i : i + seq_len + horizon]
        window_y_lagged = y_lagged_data[i : i + seq_len + horizon]

        # Sicherheitscheck: Kein NaN und Index ist lückenlos
        if np.any(np.isnan(window_x)) or np.any(np.isnan(window_y)) or np.any(np.isnan(window_y_lagged)):
            continue  # überspringen

        X.append(window_x[:seq_len])
        y_lagged.append(window_y_lagged[:seq_len])
        y.append(window_y[seq_len : seq_len + horizon])  # z. B. nur Feature 0 als Ziel

    return np.array(X, dtype=dtype), np.array(y_lagged, dtype=dtype), np.array(y, dtype=dtype)


def create_sequences(X_data: np.ndarray, y_lagged_data: np.ndarray, y_data: np.ndarray, seq_len=168, horizon=168, dtype=np.float32):
    """
    Erzeugt nur Sequenzen, die innerhalb eines zusammenhängenden Bereichs liegen.
    """

    X, y, y_lagged = [], [], []
    for i in range(len(X_data) - seq_len - horizon + 1):
        window_x = X_data[i : i + seq_len]
        window_y_lagged = y_lagged_data[i : i + seq_len]
        window_y = y_data[i + seq_len : i + seq_len + horizon]

        # Sicherheitscheck: Kein NaN und Index ist lückenlos
        if np.any(np.isnan(window_x)) or np.any(np.isnan(window_y)):
            continue  # überspringen
        X.append(window_x)
        y_lagged.append(window_y_lagged)
        y.append(window_y)  # z. B. nur Feature 0 als Ziel

    return np.array(X, dtype=dtype), np.array(y_lagged, dtype=dtype), np.array(y, dtype=dtype)


# def create_nowcasting_sequences(X_data: np.ndarray, y_lagged_data: np.ndarray, y_data: np.ndarray, seq_len=168):
#     """
#     Erzeugt Sequenzen zur Nowcasting-Vorhersage von y_t0 anhand von X[t-seq_len:t0] und y_lagged[t-seq_len:t0].
#     y_t0 ist also das Ziel zum letzten Zeitpunkt in der Sequenz.
#     """
#     X_seq, y_target, y_lagged_seq = [], [], []
    
#     for i in range(seq_len - 1, len(X_data)):
#         # Index t0 = i, Sequenz geht von i-seq_len+1 bis i
#         x_window = X_data[i - seq_len + 1 : i + 1] # X[t-seq_len:t0]
#         y_lagged_window = y_lagged_data[i - seq_len + 1 : i + 1] # y_lagged[t-seq_len:t0]
#         y_t0 = y_data[i]  # Nowcasting-Ziel: der aktuelle Zeitpunkt t0

#         # Sicherheitscheck
#         if np.any(np.isnan(x_window)) or np.any(np.isnan(y_lagged_window)) or np.any(np.isnan(y_t0)):
#             continue
        
#         X_seq.append(x_window)
#         y_lagged_seq.append(y_lagged_window)
#         y_target.append(y_t0)

#     return np.array(X_seq), np.array(y_lagged_seq), np.array(y_target)


def create_nowcasting_sequences(X_data: np.ndarray, y_lagged_data: np.ndarray, y_data: np.ndarray, seq_len=168):
    """
    Erzeugt Sequenzen zur Nowcasting-Vorhersage von y_t0 anhand von X[t-seq_len:t0] und y_lagged[t-seq_len:t0].
    y_t0 ist also das Ziel zum letzten Zeitpunkt in der Sequenz.
    """
    X_seq, y_target, y_lagged_seq = [], [], []
    
    for i in range(len(X_data) - seq_len):
        # Index t0 = i, Sequenz geht von i-seq_len+1 bis i
        x_window = X_data[i: i + seq_len] 
        y_lagged_window = y_lagged_data[i: i+seq_len] 
        y_t1 = y_data[i + seq_len]  

        # Sicherheitscheck
        if np.any(np.isnan(x_window)) or np.any(np.isnan(y_lagged_window)) or np.any(np.isnan(y_t1)):
            continue
        
        X_seq.append(x_window)
        y_lagged_seq.append(y_lagged_window)
        y_target.append(y_t1)

    return np.array(X_seq), np.array(y_lagged_seq), np.array(y_target)





def create_train_val_test_split(X, y_lagged, y, train_percentage=0.5, val_percentage=0.2, test_percentage=0.3):

    if not (train_percentage + val_percentage + test_percentage == 1):
        raise ValueError("Die Summe der Prozentsätze muss 1 ergeben. Aktuell: "
                         f"Train: {train_percentage}, Val: {val_percentage}, Test: {test_percentage}")
    


    print("\nSplitting data into train, validation and test sets...")
    print(f"Train: {round(train_percentage * 100, 3)}%, Validation: {round(val_percentage * 100, 3)}%, Test: {round(test_percentage * 100, 3)}%")

    X_seq_train = X[: int(train_percentage * len(X))]  # 50% für Training
    X_seq_val = X[int(train_percentage * len(X)) : int((train_percentage + val_percentage) * len(X))]  # 20% für Validierung
    X_seq_test = X[int((train_percentage + val_percentage) * len(X)) :]  # 30% für Test

    y_lagged_seq_train = y_lagged[: int(train_percentage * len(y_lagged))]  # 50% für Training
    y_lagged_seq_val = y_lagged[int(train_percentage * len(y_lagged)) : int((train_percentage + val_percentage) * len(y_lagged))]  # 20% für Validierung
    y_lagged_seq_test = y_lagged[int((train_percentage + val_percentage) * len(y_lagged)) :]  # 30% für Test

    y_seq_train = y[: int(train_percentage * len(y))]  # 50% für Training
    y_seq_val = y[int(train_percentage * len(y)) : int((train_percentage + val_percentage) * len(y))]  # 20% für Validierung
    y_seq_test = y[int((train_percentage + val_percentage) * len(y)) :]  # 30% für Test


    return (X_seq_train, y_lagged_seq_train, y_seq_train,
            X_seq_val, y_lagged_seq_val, y_seq_val,
            X_seq_test, y_lagged_seq_test, y_seq_test)



import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def scale_array(arr, scalers=None, fit=False, desc="", verbose=False, dtype=np.float32):

    # shape of arr: (n_samples, seq, n_features, H, W)
    # if shape = (n_samples, seq, n_features, H, W) n_features is shape[2]
    # if shape = (n_samples, seq, n_features) n_features is shape[2]

    scaled = np.empty_like(arr, dtype=dtype)

    # Falls keine Scalern übergeben → neue Liste erstellen
    if arr.ndim == 5:
        n_features = arr.shape[2]
        if verbose:
            print(f"\nScaling array with shape {arr.shape}...")
            print(f"Number of features to scale: {n_features}")

        if scalers is None:
            scalers = [StandardScaler() for _ in range(n_features)]
        for i in tqdm(range(n_features), desc=desc, disable=not verbose):
            
            feature = arr[:, :, i, :, :].reshape(-1, 1)
            if fit:
                transformed = scalers[i].fit_transform(feature)
            else:
                transformed = scalers[i].transform(feature)
            scaled[:, :, i, :, :] = transformed.reshape(scaled[:, :, i, :, :].shape)

    elif arr.ndim == 3:
        scalers = StandardScaler() if scalers is None else scalers
        if fit:
            scaled = scalers.fit_transform(arr.reshape(-1, 1)).reshape(arr.shape)
        else:
            scaled = scalers.transform(arr.reshape(-1, 1)).reshape(arr.shape)


    return scaled, scalers

def scale_data(X_train=None, y_lagged_train=None, y_train=None,
               X_val=None, y_lagged_val=None, y_val=None,
               X_test=None, y_lagged_test=None, y_test=None, 
               dtype=np.float32, verbose=False):

    # Initialize scalers
    X_scalers = None
    y_lagged_scalers = None

    # Scale X_train
    if X_train is not None:
        X_train_scaled, X_scalers = scale_array(X_train, fit=True, desc="Scaling Training features", verbose=verbose, dtype=dtype)
    else:
        X_train_scaled = None

    # Scale y_lagged_train
    if y_lagged_train is not None:
        y_lagged_train_scaled, y_lagged_scalers = scale_array(y_lagged_train, fit=True, desc="Scaling Training lagged values", verbose=verbose, dtype=dtype)
    else:
        y_lagged_train_scaled = None

    # Scale X_val
    if X_val is not None:
        X_val_scaled, _ = scale_array(X_val, scalers=X_scalers, fit=False, desc="Scaling Validation features", verbose=verbose, dtype=dtype)
    else:
        X_val_scaled = None

    # Scale y_lagged_val
    if y_lagged_val is not None:
        y_lagged_val_scaled, _ = scale_array(y_lagged_val, scalers=y_lagged_scalers, fit=False, desc="Scaling Validation lagged values", verbose=verbose, dtype=dtype)
    else:
        y_lagged_val_scaled = None

    # Scale X_test
    if X_test is not None:
        X_test_scaled, _ = scale_array(X_test, scalers=X_scalers, fit=False, desc="Scaling Test features", verbose=verbose, dtype=dtype)
    else:
        X_test_scaled = None

    # Scale y_lagged_test
    if y_lagged_test is not None:
        y_lagged_test_scaled, _ = scale_array(y_lagged_test, scalers=y_lagged_scalers, fit=False, desc="Scaling Test lagged values", verbose=verbose, dtype=dtype)
    else:
        y_lagged_test_scaled = None

    return (X_train_scaled, y_lagged_train_scaled, y_train,
            X_val_scaled, y_lagged_val_scaled, y_val,
            X_test_scaled, y_lagged_test_scaled, y_test,
            X_scalers, y_lagged_scalers)







# Scaling in parallel
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
# This code was implemented with the help of modern AI methods
def _scale_feature(feature_array, scaler, shape):
    return scaler.fit_transform(feature_array.reshape(-1, 1)).reshape(shape)

def _transform_feature(feature_array, scaler, shape):
    return scaler.transform(feature_array.reshape(-1, 1)).reshape(shape)

def scale_data_parallel(X_scaler=None, y_lagged_scaler=None,
               X_train=None, y_lagged_train=None, y_train=None,
               X_val=None, y_lagged_val=None, y_val=None,
               X_test=None, y_lagged_test=None, y_test=None,
               dtype=np.float32, verbose=False, n_jobs=-1):

    def scale_set(data, scaler, fit=True, desc=""):
        if data is None:
            return None

        scaled = np.empty_like(data, dtype=dtype)
        features = range(data.shape[-1])

        # tqdm optional
        loop = tqdm(features, desc=desc, disable=not verbose)

        results = Parallel(n_jobs=n_jobs)(
            delayed(_scale_feature if fit else _transform_feature)(
                data[..., i], scaler if fit else scaler, scaled[..., i].shape
            )
            for i in loop
        )

        for i, res in zip(features, results):
            scaled[..., i] = res

        return scaled

    # Fit & transform für Training, nur transform für Val/Test
    X_train_scaled = scale_set(X_train, X_scaler, fit=True, desc="Scaling Training features")
    y_lagged_train_scaled = scale_set(y_lagged_train, y_lagged_scaler, fit=True, desc="Scaling Training lagged values")

    X_val_scaled = scale_set(X_val, X_scaler, fit=False, desc="Scaling Validation features")
    y_lagged_val_scaled = scale_set(y_lagged_val, y_lagged_scaler, fit=False, desc="Scaling Validation lagged values")

    X_test_scaled = scale_set(X_test, X_scaler, fit=False, desc="Scaling Test features")
    y_lagged_test_scaled = scale_set(y_lagged_test, y_lagged_scaler, fit=False, desc="Scaling Test lagged values")

    return (X_train_scaled, y_lagged_train_scaled, y_train,
            X_val_scaled, y_lagged_val_scaled, y_val,
            X_test_scaled, y_lagged_test_scaled, y_test)









































def convert_to_tensors(X_train, y_lagged_train, y_train,
                     X_val, y_lagged_val, y_val,
                     X_test, y_lagged_test, y_test, dtype=torch.float32):

    # Convert to tensor
    if X_train is not None:
        X_train_tensor = torch.tensor(X_train, dtype=dtype)
    if y_lagged_train is not None:
        y_lagged_train_tensor = torch.tensor(y_lagged_train, dtype=dtype)
    if y_train is not None:
        y_train_tensor = torch.tensor(y_train, dtype=dtype)
    else:
        X_train_tensor = None
        y_lagged_train_tensor = None
        y_train_tensor = None

    if X_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=dtype)
    if y_lagged_val is not None:
        y_lagged_val_tensor = torch.tensor(y_lagged_val, dtype=dtype)
    if y_val is not None:
        y_val_tensor = torch.tensor(y_val, dtype=dtype)
    else:
        X_val_tensor = None
        y_lagged_val_tensor = None
        y_val_tensor = None

    if X_test is not None:
        X_test_tensor = torch.tensor(X_test, dtype=dtype)
    if y_lagged_test is not None:
        y_lagged_test_tensor = torch.tensor(y_lagged_test, dtype=dtype)
    if y_test is not None:
        y_test_tensor = torch.tensor(y_test, dtype=dtype)
    else:
        X_test_tensor = None
        y_lagged_test_tensor = None
        y_test_tensor = None
   



    return (X_train_tensor, y_lagged_train_tensor, y_train_tensor, \
           X_val_tensor, y_lagged_val_tensor, y_val_tensor, \
           X_test_tensor, y_lagged_test_tensor, y_test_tensor)