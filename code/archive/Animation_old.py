import warnings
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from tqdm import tqdm
from utils.eda_helper_functions import (
    group_data_hourly,
    load_insitu_data,
    load_ocean_data,
    load_weather_data,
    process_df,
    process_flensburg_data,
)

# Configs
OCEAN_POINTS = 30
WEATHER_POINTS = 10
LAT_FLENSBURG = 54.796001
LON_FLENSBURG = 9.436999
mpl.rcParams['animation.ffmpeg_path'] = '.venv/lib/python3.11/site-packages/ffmpeg'
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.options.display.max_columns = None

print("\nLoading data...")

# Load and process data
df_ocean = process_df(load_ocean_data(Path(f"data/numerical_data/points{OCEAN_POINTS}"), OCEAN_POINTS, False), drop_cols=["depth"])
df_weather = process_df(load_weather_data(Path(f"data/numerical_data/points{WEATHER_POINTS}"), WEATHER_POINTS, False))
df_insitu = group_data_hourly(process_df(process_flensburg_data(load_insitu_data(file_path="data/observation", verbose=False), start_time=df_ocean['time'].min(), end_time=df_ocean['time'].max())))

# Compute land mask once
def create_land_mask_once(lon_grid, lat_grid, timepoint):
    coords = [(lon, lat) for lat in lat_grid for lon in lon_grid]
    mask_flat = Parallel(n_jobs=-1)(delayed(lambda p: not is_on_land(*p))(p) for p in tqdm(coords, desc=f"Creating land mask for {timepoint}", total=len(coords)))
    return np.array(mask_flat).reshape(len(lat_grid), len(lon_grid))

def is_on_land(lon, lat):
    land = cfeature.NaturalEarthFeature("physical", "land", "10m")
    return any(geom.contains(shapely.geometry.Point(lon, lat)) for geom in land.geometries())

# Main animation function
def animation_plot(
    timepoint,
    fig=None,
    ax=None,
    grid_size_ocean=50,
    wind_grid_size=25,
    vmin=-1.0,
    vmax=1.5,
    plot_water_velocity_data=True,
    plot_wind_data=True,
):
    global df_ocean, df_weather  # notwendig, wenn du außerhalb des Funktionsscopes auf globale df zugreifst

    # if fig is None or ax is None:
    #     fig, ax = plt.subplots(figsize=(12, 10))

    # Filter für aktuellen Zeitpunkt
    df_weather_time = df_weather[df_weather["time"] == timepoint]
    df_ocean_time = df_ocean[df_ocean["time"] == timepoint]

    if df_ocean_time.empty or df_weather_time.empty:
        ax.set_title("Keine Daten für diesen Zeitpunkt verfügbar.")
        return fig, ax

    # Raster definieren
    lon_grid = np.linspace(
        df_ocean_time["longitude"].min(),
        df_ocean_time["longitude"].max(),
        grid_size_ocean,
    )
    lat_grid = np.linspace(
        df_ocean_time["latitude"].min(),
        df_ocean_time["latitude"].max(),
        grid_size_ocean,
    )
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # SLA interpolieren
    sla_grid = griddata(
        (df_ocean_time["longitude"], df_ocean_time["latitude"]),
        df_ocean_time["sla"],
        (lon_mesh, lat_mesh),
        method="linear",
    )

    m = Basemap(
        projection="cyl",
        resolution="i",
        llcrnrlon=lon_grid.min(),
        urcrnrlon=lon_grid.max(),
        llcrnrlat=lat_grid.min(),
        urcrnrlat=lat_grid.max(),
        ax=ax,
    )
    m.fillcontinents(color="grey", lake_color="white", alpha=0.5)
    m.drawcoastlines()
    m.drawcountries()

    # plot target point
    x_target, y_target = m(LON_FLENSBURG, LAT_FLENSBURG)  # Reihenfolge: (longitude, latitude)
    m.scatter(x_target, y_target, color="green", marker="*", label="Target Point", s=200) 


    mask = create_land_mask_once(lon_grid, lat_grid, timepoint)
    if sla_grid is not None:
        sla_grid[~mask] = np.nan

        x_mesh, y_mesh = m(lon_mesh, lat_mesh)
        heatmap = m.pcolormesh(
            x_mesh, y_mesh, sla_grid, cmap="magma", shading="auto", vmin=vmin, vmax=vmax
        )

    # Wasserströmungen
    if plot_water_velocity_data:
        for comp, var in zip(["uo", "vo"], ["water_uo", "water_vo"]):
            if comp not in df_ocean_time.columns:
                plot_water_velocity_data = False

        if plot_water_velocity_data:
            water_uo = griddata(
                (df_ocean_time["longitude"], df_ocean_time["latitude"]),
                df_ocean_time["uo"],
                (lon_mesh, lat_mesh),
                method="linear",
            )
            water_vo = griddata(
                (df_ocean_time["longitude"], df_ocean_time["latitude"]),
                df_ocean_time["vo"],
                (lon_mesh, lat_mesh),
                method="linear",
            )

            stride = max(1, int(grid_size_ocean / 50))
            water_uo[~mask] = np.nan
            water_vo[~mask] = np.nan

            x_current = x_mesh[::stride, ::stride]
            y_current = y_mesh[::stride, ::stride]
            u_current = water_uo[::stride, ::stride]
            v_current = water_vo[::stride, ::stride]

            m.quiver(
                x_current,
                y_current,
                u_current,
                v_current,
                scale=20,
                color="grey",
                width=0.002,
                alpha=0.99,
                label="Current",
            )

    # Winddaten
    if plot_wind_data:
        lon_grid_wind = np.linspace(
            df_weather_time["longitude"].min(),
            df_weather_time["longitude"].max(),
            wind_grid_size,
        )
        lat_grid_wind = np.linspace(
            df_weather_time["latitude"].min(),
            df_weather_time["latitude"].max(),
            wind_grid_size,
        )
        lon_mesh_wind, lat_mesh_wind = np.meshgrid(lon_grid_wind, lat_grid_wind)

        wind_speed_grid = griddata(
            (df_weather_time["longitude"], df_weather_time["latitude"]),
            df_weather_time["wind_speed_10m"],
            (lon_mesh_wind, lat_mesh_wind),
            method="linear",
        )
        wind_dir_grid = griddata(
            (df_weather_time["longitude"], df_weather_time["latitude"]),
            df_weather_time["wind_direction_10m"],
            (lon_mesh_wind, lat_mesh_wind),
            method="linear",
        )

        if wind_speed_grid is not None and wind_dir_grid is not None:
            u = wind_speed_grid * -np.cos(np.deg2rad(wind_dir_grid))
            v = wind_speed_grid * -np.sin(np.deg2rad(wind_dir_grid))
            x_wind, y_wind = m(lon_mesh_wind, lat_mesh_wind)
            m.quiver(x_wind, y_wind, u, v, scale=1500, color="black")

    m.drawparallels(np.arange(0, 360, 2), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(0, 350, 2), labels=[0, 0, 0, 1])
    return heatmap

def run_initial_frame(timepoints, grid_size=50):
    fig_in, ax_in = plt.subplots(figsize=(14, 12))
    cbar_ax_in = fig_in.add_axes([0.25, 0.1, 0.5, 0.03])
    heatmap = animation_plot(timepoints[0], fig_in, ax_in, grid_size_ocean=grid_size, plot_water_velocity_data=True, plot_wind_data=True)
    ax_in.set_title(f"Zeitpunkt: {timepoints[0]}", fontsize=16)
    if heatmap is not None:
        fig_in.colorbar(heatmap, cax=cbar_ax_in, orientation="horizontal").set_label("Water Level (m)")
        legend_elements = [Line2D([0], [0], color='green', marker='*', linestyle='', markersize=15, label='Flensburg')]
        legend_elements.append(Line2D([0], [0], color='black', lw=4, marker=r'$\rightarrow$', label='Wind Direction', linestyle=''))
        legend_elements.append(Line2D([0], [0], color='grey', lw=4, marker=r'$\rightarrow$', label='Ocean Current', linestyle=''))
        fig_in.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.87, 0.59), frameon=True, title="Legend")
    fig_in.savefig(f"initial_frame_{timepoints[0]}_to_{timepoints[-1]}_grid{grid_size}.png", dpi=300)

def run_animation(timepoints, grid_size=50, fps=6):
    # lon_grid = np.linspace(df_ocean["longitude"].min(), df_ocean["longitude"].max(), grid_size)
    # lat_grid = np.linspace(df_ocean["latitude"].min(), df_ocean["latitude"].max(), grid_size)
    # mask = create_land_mask_once(lon_grid, lat_grid)

    fig, ax = plt.subplots(figsize=(12, 10))
    cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    
    def update(idx):

        if timepoints[idx] == timepoints[0]:
            run_initial_frame(timepoints, grid_size=grid_size)

        ax.clear()
        heatmap = animation_plot(timepoints[idx], fig, ax, grid_size_ocean=grid_size, plot_water_velocity_data=True, plot_wind_data=True)
        ax.set_title(f"Zeitpunkt: {timepoints[idx]}", fontsize=16)
        if idx == 0 and heatmap is not None:
            fig.colorbar(heatmap, cax=cbar_ax, orientation="horizontal").set_label("Water Level (m)")
            legend_elements = [Line2D([0], [0], color='green', marker='*', linestyle='', markersize=15, label='Flensburg')]
            legend_elements.append(Line2D([0], [0], color='black', lw=4, marker=r'$\rightarrow$', label='Wind Direction', linestyle=''))
            legend_elements.append(Line2D([0], [0], color='grey', lw=4, marker=r'$\rightarrow$', label='Ocean Current', linestyle=''))
            fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.87, 0.59), frameon=True, title="Legend")


    ani = FuncAnimation(fig, update, frames=len(timepoints), interval=1000//fps)
    ani.save(f"animation_{timepoints[0]}_to_{timepoints[-1]}_grid{grid_size}_fps{fps}.gif", writer=PillowWriter(fps=fps))
    #ani.save(f"animation_{grid_size}_{fps}.mp4", writer="ffmpeg", fps=fps)

if __name__ == "__main__":
    timepoints = sorted(set(df_ocean["time"]) & set(df_weather["time"]))
    #timepoints = [t for t in timepoints if pd.Timestamp("2023-10-17 00:00:00") <= t <= pd.Timestamp("2023-10-23 23:00:00")]
    timepoints = [t for t in timepoints if pd.Timestamp("2023-10-21 00:00:00") <= t <= pd.Timestamp("2023-10-21 02:00:00")]
    print('Running animation...')
    run_animation(timepoints, grid_size=10, fps=6)
