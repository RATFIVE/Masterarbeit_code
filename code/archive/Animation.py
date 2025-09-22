

# import all necessary libraries
# import all necessary libraries
import warnings
from pathlib import Path

import cartopy.feature as cfeature
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
import matplotlib as mpl
from utils.eda_helper_functions import (
    group_data_hourly,
    load_insitu_data,
    load_ocean_data,
    load_weather_data,
    process_df,
    process_flensburg_data,
)

# .venv/lib/python3.11/site-packages/ffmpeg

mpl.rcParams['animation.ffmpeg_path'] = '.venv/lib/python3.11/site-packages/ffmpeg'

# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Display all columns
pd.options.display.max_columns = None


# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

OCEAN_POINTS = 30
WEATHER_POINTS = 10
LAT_FLENSBURG = 54.796001
LON_FLENSBURG = 9.436999



ocean_data_path = Path(f"data/numerical_data/points{OCEAN_POINTS}")
# print(ocean_data_path)
weather_data_path = Path(f"data/numerical_data/points{WEATHER_POINTS}")
# print(weather_data_path)


df_ocean = load_ocean_data(ocean_data_path, OCEAN_POINTS, verbose=False)
df_ocean = process_df(df_ocean, drop_cols=["depth"], verbose=False)

df_weather = load_weather_data(weather_data_path, WEATHER_POINTS, verbose=False)
df_weather = process_df(df_weather, verbose=False)

df_insitu = load_insitu_data(verbose=False, file_path="data/observation")
df_insitu = process_flensburg_data(df_insitu, 
                                      start_time=df_ocean['time'].min(),
                                      end_time=df_ocean['time'].max(),
                                      verbose=False)

df_insitu = group_data_hourly(df_insitu)
df_insitu = process_df(df_insitu, drop_cols=["deph"], verbose=False)





# Funktion zur Landprüfung mit Cartopy
def is_on_land(lon, lat):
    land = cfeature.NaturalEarthFeature("physical", "land", "10m")
    for geom in land.geometries():
        if geom.contains(shapely.geometry.Point(lon, lat)):
            return True
    return False


# Funktion zum Erstellen der Landmaske
def create_land_mask(lon_grid, lat_grid):
    coords_list = [(lon, lat) for lat in lat_grid for lon in lon_grid]
    mask_flat = Parallel(n_jobs=-1)(
        delayed(lambda p: not is_on_land(*p))(p) for p in tqdm(coords_list)
    )
    return np.array(mask_flat).reshape(len(lat_grid), len(lon_grid))




def animation_plot(
    timepoint,
    fig=None,
    ax=None,
    grid_size_ocean=50,
    wind_grid_size=20,
    vmin=-1.0,
    vmax=1.5,
    plot_water_velocity_data=True,
    plot_wind_data=True,
):
    global df_ocean, df_weather  # notwendig, wenn du außerhalb des Funktionsscopes auf globale df zugreifst

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

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


    mask = create_land_mask(lon_grid, lat_grid)
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
    return fig, ax, heatmap







    
def test_plot():
    # Erstelle die Animation
    fig, ax, heatmap = animation_plot(
        timepoints[0],
        fig=fig,
        ax=ax,
        grid_size_ocean=grid_size,
        wind_grid_size=25,
        vmin=-1.0,
        vmax=1.5,
        plot_water_velocity_data=True,
        plot_wind_data=True,
    )
    ax.set_title(f"Zeitpunkt: {timepoints[0]}", fontsize=16)
    cbar = fig.colorbar(heatmap, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_label("Water Level (m)", fontsize=14)

    legend_elements = []
    legend_elements.append(Line2D([0], [0], color='green', marker='*', markersize=15, linestyle='', label='Flensburg')) # Add target point to legend
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.85, 0.59), frameon=True, title="Legend", fontsize=12)

    # Save the initial frame
    fig.savefig("ocean_wind_animation_initial_frame.png", dpi=300)



fig, ax = plt.subplots()
cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])  # eigene Achse für Colorbar

def update(frame_idx):
    current_time = timepoints[frame_idx]
    print(f"Processing timepoint: {current_time}")
    
    ax.clear()
    
    _, _, heatmap = animation_plot(
        current_time,
        fig=fig,
        ax=ax,
        grid_size_ocean=grid_size,
        wind_grid_size=25,
        vmin=-1.0,
        vmax=1.5,
        plot_water_velocity_data=True,
        plot_wind_data=True,
    )

    ax.set_title(f"Zeitpunkt: {current_time}", fontsize=16)

    # Colorbar im ersten Frame initialisieren
    if frame_idx == 0:
        cbar = fig.colorbar(heatmap, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Water Level (m)", fontsize=14)

        legend_elements = [
            Line2D([0], [0], color='green', marker='*', markersize=15, linestyle='', label='Flensburg')
        ]
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.85, 0.59), frameon=True, title="Legend", fontsize=12)



def run_animation(timepoints, grid_size=50, fps=6):
    # Animation erzeugen
    ani = FuncAnimation(fig, update, frames=len(timepoints), interval=500)

    # Als GIF speichern
    ani.save(f"ocean_wind_animation{grid_size}_Framrate{fps}.gif", writer=PillowWriter(fps=fps))

    # Als mp4 speichern
    ani.save(f"ocean_wind_animation{grid_size}_Framrate{fps}.mp4", writer="ffmpeg", fps=fps)




if __name__ == "__main__":
    
    # Erstelle die Liste der Zeitpunkte (achte darauf, dass sie in beiden DataFrames vorkommen)
    timepoints = sorted(set(df_ocean["time"]).intersection(set(df_weather["time"])))

    # Konvertiere die Strings in Timestamps
    start_date = pd.Timestamp("2023-10-22")
    end_date = pd.Timestamp("2023-10-22")



    # Filtere die Zeitpunkte
    timepoints = [
        timepoint
        for timepoint in timepoints
        if timepoint >= start_date and timepoint <= end_date
    ]
    timepoints


    # Setup der Figur
    # Plot-Setup
    fig, ax = plt.subplots(figsize=(12, 10))

    grid_size = 50
    run_animation(timepoints, grid_size=grid_size, fps=6)

