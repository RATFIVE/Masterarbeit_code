# import all necessary libraries
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.dates as mdates

try:
    from config import (
        OCEAN_DICT,
        WEATHER_DICT,
        INSITU_DICT,
        )
except ImportError:
    from utils.config import (
        OCEAN_DICT,
        WEATHER_DICT,
        INSITU_DICT,
        )
sns.set_theme(style="white")

plt.rcParams.update({
    "font.size": 16,                # Grundschriftgröße (wirkt auf alles, sofern nicht überschrieben)
    "axes.titlesize": 16,           # Größe des Titels der Achse (z.B. 'Subplot Title')
    "axes.labelsize": 16,           # Achsenbeschriftung (x/y label)
    "xtick.labelsize": 14,          # X-Tick-Beschriftung
    "ytick.labelsize": 14,          # Y-Tick-Beschriftung
    "legend.fontsize": 14,          # Legendentext
    "figure.titlesize": 18,         # Gesamttitel der Abbildung (plt.suptitle)
    "figure.labelsize": 16,         # (optional, selten verwendet)
    "savefig.dpi": 300,             # DPI beim Speichern
    "figure.dpi": 100,              # DPI bei Anzeige
})

# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)




def load_ocean_data(
    ocean_data_path: Path, ocean_points: str, verbose=False) -> pd.DataFrame:
    """
    Load and concatenate ocean data from NetCDF (.nc) files in the given directory.

    Parameters:
        ocean_data_path (Path): Path to the directory containing .nc files.
        ocean_points (str): Identifier used to exclude a specific weather data file.

    Returns:
        pd.DataFrame: Combined DataFrame of ocean data.
    """
    # Collect all .nc files
    nc_files = sorted(
        file for file in os.listdir(ocean_data_path) if file.endswith(".nc")
    )

    # Remove the weather file if it exists
    weather_file = f"df_weather{ocean_points}.nc"
    if weather_file in nc_files:
        nc_files.remove(weather_file)
        print(f"{weather_file} removed")

    # Load and combine the data
    df_list = []
    for file in nc_files:
        file_path = ocean_data_path / file
        with xr.open_dataset(file_path) as ds:
            df = ds.to_dataframe().reset_index()
            df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)

    if verbose:
        print(df.head())
        print(df.info(verbose=True, show_counts=True, memory_usage="deep"))

    return df


def load_weather_data(weather_data_path, weather_points, verbose=False):
    """
    Load the weather data from the specified path and process it.
    """
    file_path = weather_data_path / f"df_weather{weather_points}.csv"
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])

    if verbose:
        print(df.head())
        print(df.info(verbose=True, show_counts=True, memory_usage="deep"))

    return df


def load_insitu_data(file_name="NO_TS_TG_FlensburgTG.nc", file_path:str=None, verbose=False) -> pd.DataFrame:
    """
    Load the insitu data from the specified NetCDF file and process it.
    """
    if file_path is None:
        file_path = Path("../data/observation") / file_name
    else:
        file_path = Path(file_path) / file_name
    ds = xr.open_dataset(file_path)
    df = ds.to_dataframe().reset_index()

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="any")

    df.columns = df.columns.str.lower()  # make all columns lowercase

    if "station" in df.columns:
        df = df.drop(columns=["station"], axis=1)

    if verbose:
        show_df(df)
    return df


def process_df(df: pd.DataFrame, drop_cols: list | None = None, verbose=False) -> pd.DataFrame:
    """
    Process the DataFrame by dropping specified columns, converting data types,
    and resetting the index."""
    if drop_cols is not None:
        # Drop specified columns
        df = df.drop(columns=drop_cols, axis=1)

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="any")

    df = df[["time"] + [col for col in df.columns if col != "time"]]
    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.round("h")

    df = df.reset_index(drop=True)

    if verbose:
        show_df(df)

    return df


def process_flensburg_data(df: pd.DataFrame, start_time: str | None = None, end_time: str | None = None, verbose: bool = False, order=3) -> pd.DataFrame:
    """
    Processes water level data for Flensburg by applying optional time filtering,
    removing outliers, and performing interpolation.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing columns 'time', 'slev', and 'slev_qc'.
    start_time : str | None, optional
        Optional start timestamp (ISO 8601 format). If provided, filters data from this point onward.
    end_time : str | None, optional
        Optional end timestamp (ISO 8601 format). If provided, filters data up to this point.
    verbose : bool, default=False
        If True, plots the 'slev' values before and after interpolation for visual inspection.

    Returns:
    -------
    pd.DataFrame
        The processed DataFrame with interpolated 'slev' values and filtered time range.
    """

    # Apply time filtering and quality control if a time range is specified
    if start_time or end_time:
        df = df[
            (df["time"] >= start_time) & (df["time"] <= end_time)
        ].reset_index(drop=True)
        df = df[df["slev_qc"] == 1].reset_index(drop=True)

    # Define the time range containing the known outlier
    outlier_range = ("2023-09-09 01:00", "2023-09-09 23:00")
    interpolation_range = ("2023-09-09 19:30", "2023-09-09 19:50")

    if verbose:
        df.loc[
            (df["time"] >= outlier_range[0]) & (df["time"] <= outlier_range[1])
        ].plot(x="time", y="slev", title="slev before interpolation", figsize=(12, 6))

    # Set the 'time' column as index for interpolation
    df.set_index("time", inplace=True)

    # Set known outlier range to NA
    df.loc[interpolation_range[0]:interpolation_range[1], "slev"] = pd.NA

    # Perform polynomial interpolation
    df["slev"] = df["slev"].interpolate(
        method="polynomial", order=order, limit_direction="both"
    )

    # Reset the index back to column
    df.reset_index(inplace=True)

    if verbose:
        df.loc[
            (df["time"] >= outlier_range[0]) & (df["time"] <= outlier_range[1])
        ].plot(x="time", y="slev", title="slev after interpolation", figsize=(12, 6))

        
    df = interpolate_missing_times(df) # Interpolate missing time points

    if verbose:
        show_df(df)

    return df



def check_missing_times(df: pd.DataFrame) -> None:
    """
    Check for missing time points in the DataFrame.
    This function calculates the time differences between consecutive rows
    and checks for any gaps larger than the expected interval.
    """
    # Zeitdifferenzen berechnen
    time_diff = df["time"].diff()

    # Falls der Zeitabstand konstant sein sollte (z. B. 1 Stunde), prüfe Abweichungen:
    expected_interval = pd.Timedelta(hours=1)  # Anpassen je nach erwartetem Intervall
    missing_times = df["time"][time_diff > expected_interval]

    # Fehlende Zeitpunkte ausgeben
    if missing_times.empty:
        print("Keine fehlenden Zeitpunkte!")
    else:
        print("Fehlende Zeitpunkte erkannt:")
        print(missing_times)


def show_df(df: pd.DataFrame) -> None:
    """
    Show the DataFrame information, including the first and last 5 rows,
    columns, unique coordinates, and NaN values.
    """
    # show the first 5 rows of the dataframe
    print("\nFirst 5 rows:")
    print(df.head())
    # show the last 5 rows of the dataframe
    print("\nLast 5 rows:")
    print(df.tail())

    # show columns
    print("\nColumns:", df.columns.tolist())

    unique_coordinates = df[["latitude", "longitude"]].drop_duplicates()
    n_unique_coordinates = len(unique_coordinates)
    print(f"\nNumber of unique coordinates: {n_unique_coordinates}")
    print("\nInfo:")
    print(df.info(show_counts=True, verbose=True, memory_usage="deep"))

    print("\nNaN values in each column:")
    print(df.isna().sum())
    print("\nData Statistics:")
    print(df.describe())

    print("\nChecking for missing times:")
    check_missing_times(df)

    # Show NaN Rows
    print("\nNaN Rows:")
    nan_rows = df[df.isna().any(axis=1)]
    print(nan_rows)
    print(f"Number of NaN rows: {len(nan_rows)}")


def interpolate_missing_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates missing time points in a DataFrame with a time index.

    Parameters:
        df (pd.DataFrame): DataFrame with a time index.

    Returns:
        pd.DataFrame: DataFrame with interpolated missing time points.
    """

    df = df.set_index("time")
    # Erzeuge vollständigen Zeitindex
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10min', name='time')

    # Reindexiere den DataFrame (fehlende Zeitpunkte bekommen NaN)
    df_full = df.reindex(full_index)

    # Interpolieren der fehlenden Werte (linear oder zeitlich)
    df_full_interpolated = df_full.interpolate(method='polynomial', order=3, limit_direction='both') 
    df_full_interpolated = df_full_interpolated.reset_index()
    return df_full_interpolated


def group_data_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the DataFrame by hour and calculates the mean for each hour.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'time' column.

    Returns:
        pd.DataFrame: DataFrame grouped by hour with mean values.
    """
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df_hourly = df.resample("h").mean().reset_index()
    return df_hourly


# ---------------------- Plots ---------------------- #


def plot_water_level_anomalies(df: pd.DataFrame, anomaly_threshold: float = 1.0, start_date:str | None=None, end_date:str | None=None) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots water level time series with anomalies and categorized flood levels.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'time' (datetime) and 'slev' (water level in meters).
    anomaly_threshold : float, default=1.0
        Threshold above which water levels are considered anomalies (potential storm surges).

    Returns:
    -------
    None
        Displays a matplotlib plot.
    """
    df_anomalies = df.copy()

    if start_date or end_date:
        # Convert to datetime if not already
        df_anomalies['time'] = pd.to_datetime(df_anomalies['time'])

        # Filter the DataFrame based on the date range
        if start_date:
            df_anomalies = df_anomalies[df_anomalies['time'] >= start_date]
        if end_date:
            df_anomalies = df_anomalies[df_anomalies['time'] <= end_date]

    df_anomalies['anomaly'] = (df_anomalies['slev'] > anomaly_threshold).astype(int)

    # Group anomalies by time
    df_anomalies_grouped = (
        df_anomalies[df_anomalies['anomaly'] == 1]
        .groupby('time')[['slev']]
        .mean()
        .reset_index()
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot water level line
    ax.plot(df_anomalies['time'], df_anomalies['slev'], label='water level', color='blue')

    # Plot anomalies as red dot
    ax.scatter(df_anomalies_grouped['time'], df_anomalies_grouped['slev'],
               color='red', label='anomaly', marker='o')

    # Highlight storm surge classes with colored bands
    flood_levels = [
        (1.0, 1.25, 'yellow', 'storm surge'),
        (1.25, 1.5, 'orange', 'medium storm surge'),
        (1.5, 2.0, 'red', 'heavy storm surge'),
        (2.0, 3.5, 'darkred', 'very heavy storm surge'),
    ]

    for y0, y1, color, label in flood_levels:
        ax.axhspan(y0, y1, facecolor=color, alpha=0.3, label=label)

    # Time range for title
    min_time = df_anomalies['time'].min().strftime('%Y-%m-%d')
    max_time = df_anomalies['time'].max().strftime('%Y-%m-%d')

    # Axis labeling and formatting
    ax.set_title(f"Water Level with Anomalies and Flood Levels from {min_time} to {max_time}")
    ax.set_xlabel("time")
    ax.set_ylabel("water level [m]")
    ax.legend(loc='upper right')
    ax.set_ylim(df_anomalies['slev'].min() - 0.5, df_anomalies['slev'].max() + 0.5)  # Set y-axis limits
    ax.yaxis.set_major_locator(plt.MaxNLocator(10)) # make y-axis labels readable
    ax.xaxis.set_major_locator(plt.MaxNLocator(25)) # make x-axis labels readable
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    #ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    fig.autofmt_xdate()

    plt.grid(True)
    plt.tight_layout()
    

    return fig, ax






def plot_coordinates(df_ocean: pd.DataFrame, df_weather: pd.DataFrame, df_insitu: pd.DataFrame, save_png: bool = False) -> None:
    """
    Plot ocean and weather coordinates on a map using Basemap.

    Parameters:
        df_ocean (pd.DataFrame): DataFrame containing ocean data with 'latitude' and 'longitude'.
        df_weather (pd.DataFrame): DataFrame containing weather data with 'latitude' and 'longitude'.
        save_png (bool): Whether to save the map as a PNG file. Default is False.
    """
    # Drop duplicate coordinates
    ocean_coords = df_ocean[["latitude", "longitude"]].drop_duplicates()
    weather_coords = df_weather[["latitude", "longitude"]].drop_duplicates()

    # Count for filename and label
    num_ocean_coords = len(ocean_coords)
    num_weather_coords = len(weather_coords)

    # Calculate mean coordinates for map centering
    mean_lat = ocean_coords["latitude"].mean()
    mean_lon = ocean_coords["longitude"].mean()

    # Create map
    plt.figure(figsize=(10, 8))
    m = Basemap(
        projection="lcc",
        resolution="i",
        lat_0=mean_lat,
        lon_0=mean_lon,
        width=1.2e6,
        height=1.2e6,
    )

    # Draw map features
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="0.8")
    m.drawstates()
    m.drawmapboundary(fill_color="white")
    m.fillcontinents(color="grey", lake_color="white", alpha=0.2)

    # get unique coordinates of df_insitu
    unique_coordinates = df_insitu[["latitude", "longitude"]].drop_duplicates()
    target_latitude = unique_coordinates["latitude"].values[0]
    target_longitude = unique_coordinates["longitude"].values[0]

    # Convert lat/lon to map projection coordinates
    x_ocean, y_ocean = m(ocean_coords["longitude"].values, ocean_coords["latitude"].values)
    x_weather, y_weather = m(weather_coords["longitude"].values, weather_coords["latitude"].values)
    x_target, y_target = m(target_longitude, target_latitude)
    # Plot points
    m.scatter(x_ocean, y_ocean, color="blue", label="Ocean Data", zorder=5)
    m.scatter(x_weather, y_weather, color="red", label="Weather Data", zorder=5)
    m.scatter(x_target, y_target, color="green", marker="*", label="Target Point", s=200)

    # Add labels and grid
    plt.title(f"Ocean Data ({num_ocean_coords} points) & Weather Data ({num_weather_coords} points)")
    plt.legend(loc="upper left")
    m.drawparallels(np.arange(-90, 91, 2), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180, 181, 2), labels=[0, 0, 0, 1])

    # Save plot if requested
    if save_png:
        filename = f"map_oceanpoints_{num_ocean_coords}_weatherpoints_{num_weather_coords}.png"
        plt.savefig(filename, dpi=300)
        print(f"Map saved as {filename}")

    plt.show()







def plot_histogram(
    df: pd.DataFrame, 
    column: str, 
    bins: int = 50, 
    color: str = 'steelblue', 
    kde: bool = True,
    ax: plt.Axes | None = None,
    fig: plt.Figure | None = None,
    show_stats: bool = True,  # <--- neue Option
    title: str = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a styled histogram of a specified column in the DataFrame with optional KDE overlay and statistical markers.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    column : str
        Column name to plot.
    bins : int, default=50
        Number of bins for the histogram.
    color : str, default='steelblue'
        Color of the histogram bars.
    kde : bool, default=True
        If True, overlays a kernel density estimate.
    show_stats : bool, default=True
        If True, plot vertical lines for min, max, mean, and median.
    """
    
    
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    sns.histplot(df[column], bins=bins, color=color, kde=kde, edgecolor='white', linewidth=0.5, ax=ax)

    if show_stats:
        min_val = df[column].min()
        max_val = df[column].max()
        mean_val = df[column].mean()
        median_val = df[column].median()
        std_val = df[column].std()

        # Plot lines
        if column == 'time':
            # Convert time to numeric for plotting
            min_val = pd.to_datetime(min_val)
            max_val = pd.to_datetime(max_val)
            mean_val = pd.to_datetime(mean_val)
            median_val = pd.to_datetime(median_val)
            
            ax.axvline(min_val, color='red', linestyle='--', linewidth=1.5, label=f'Min: {pd.to_datetime(min_val, unit="s").strftime("%Y-%m-%d %H:%M")}')
            ax.axvline(max_val, color='purple', linestyle='--', linewidth=1.5, label=f'Max: {pd.to_datetime(max_val, unit="s").strftime("%Y-%m-%d %H:%M")}')
            ax.axvline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean: {pd.to_datetime(mean_val, unit="s").strftime("%Y-%m-%d %H:%M")}')
            ax.axvline(median_val, color='orange', linestyle='-', linewidth=2, label=f'Median: {pd.to_datetime(median_val, unit="s").strftime("%Y-%m-%d %H:%M")}')
            #ax.axvspan(mean_val - std_val, mean_val + std_val, color='blue', alpha=0.1, label=f'±1σ: ({pd.to_datetime(std_val, unit="s").strftime("%Y-%m-%d %H:%M")})')
       
        # elif column in []:
        #     ax.axvline(min_val, color='red', linestyle='--', linewidth=1.5, label=f'Min: {min_val:.3f}')
        #     ax.axvline(max_val, color='purple', linestyle='--', linewidth=1.5, label=f'Max: {max_val:.3f}')
        #     ax.axvline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
        #     ax.axvline(median_val, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_val:.3f}')
        #     ax.axvspan(mean_val - std_val, mean_val + std_val, color='blue', alpha=0.1, label=f'±1σ: {std_val:.3f}')
        elif column in ['wo', 'showers', 'sithick', 'snowfall', 'rain', 'precipitation']:
            scale_factor = 1e6

            ax.axvline(min_val, color='red', linestyle='--', linewidth=1.5,
                    label=f'Min: {min_val * scale_factor:.2f}×10⁻⁶')
            ax.axvline(max_val, color='purple', linestyle='--', linewidth=1.5,
                    label=f'Max: {max_val * scale_factor:.2f}×10⁻⁶')
            ax.axvline(mean_val, color='green', linestyle='-', linewidth=2,
                    label=f'Mean: {mean_val * scale_factor:.2f}×10⁻⁶')
            ax.axvline(median_val, color='orange', linestyle='-', linewidth=2,
                    label=f'Median: {median_val * scale_factor:.2f}×10⁻⁶')
            ax.axvspan(mean_val - std_val, mean_val + std_val, color='blue', alpha=0.1,
                    label=f'±1σ: {std_val * scale_factor:.2f}×10⁻⁶')

        else:
            ax.axvline(min_val, color='red', linestyle='--', linewidth=1.5, label=f'Min: {min_val:.2f}')
            ax.axvline(max_val, color='purple', linestyle='--', linewidth=1.5, label=f'Max: {max_val:.2f}')
            ax.axvline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.axvspan(mean_val - std_val, mean_val + std_val, color='blue', alpha=0.1, label=f'±1σ: {std_val:.2f}')


        ax.legend()
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Histogram of {column}")
    

    if column == 'time':
        #rotate x-axis labels for time
        #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pd.to_datetime(x).strftime('%Y-%m-%d')))
        #ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xticks(rotation=30)
        
    if column in OCEAN_DICT:
        unit = OCEAN_DICT[column]["unit"]
        ax.set_xlabel(f"{column} [{unit}]")
    elif column in WEATHER_DICT:
        unit = WEATHER_DICT[column]["unit"]
        ax.set_xlabel(f"{column} [{unit}]")
    elif column in INSITU_DICT:
        unit = INSITU_DICT[column]["unit"]
        ax.set_xlabel(f"{column} [{unit}]")
    
    # make y-axis log
    ax.set_yscale('log')
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig, ax

