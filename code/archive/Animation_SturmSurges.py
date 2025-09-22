import datetime
import warnings
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
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
from utils.config import (
    LAT_FLENSBURG,
    LON_FLENSBURG,
    SUB_BOX,
    OCEAN_DICT,
    WEATHER_DICT,
    INSITU_DICT,
    OCEAN_POINTS,
    WEATHER_POINTS,
    )

plt.rcParams.update({
    "font.size": 14,                # Grundschriftgröße (wirkt auf alles, sofern nicht überschrieben)
    "axes.titlesize": 16,           # Größe des Titels der Achse (z.B. 'Subplot Title')
    "axes.labelsize": 14,           # Achsenbeschriftung (x/y label)
    "xtick.labelsize": 12,          # X-Tick-Beschriftung
    "ytick.labelsize": 12,          # Y-Tick-Beschriftung
    "legend.fontsize": 12,          # Legendentext
    "figure.titlesize": 18,         # Gesamttitel der Abbildung (plt.suptitle)
    "figure.labelsize": 14,         # (optional, selten verwendet)
    "savefig.dpi": 300,             # DPI beim Speichern
    "figure.dpi": 100,              # DPI bei Anzeige
})


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
    grid_size_ocean=200,
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
    
    # Add contour lines for pressure_msl
    if "pressure_msl" in df_weather_time.columns:
        pressure_msl_grid = griddata(
            (df_weather_time["longitude"], df_weather_time["latitude"]),
            df_weather_time["pressure_msl"],
            (lon_mesh_wind, lat_mesh_wind),
            method="linear",
        )
        if pressure_msl_grid is not None:
            x_pressure, y_pressure = m(lon_mesh_wind, lat_mesh_wind)
            # Konturstufen definieren (alle 2 hPa z.B.)
            

            # 1. Konturstufen und Konturen zeichnen
            levels = np.arange(np.nanmin(pressure_msl_grid), np.nanmax(pressure_msl_grid), 2)
            cmap = plt.get_cmap('coolwarm')

            cs = m.contour(x_pressure, y_pressure, pressure_msl_grid, 
                        levels=levels, 
                        cmap=cmap, 
                        linewidths=1.0)

            plt.clabel(cs, inline=True, 
                       #fontsize=8, 
                       fmt="%.0f hPa")

            # 2. Farbskala (optional)
            #cbar = plt.colorbar(cs, orientation='horizontal', pad=0.05)
            #cbar.set_label('Luftdruck (hPa)')

            # 3. Hoch- und Tiefdruckzentren finden
            # Tiefdruck: Lokale Minima, Hochdruck: Lokale Maxima
            pressure_msl_masked = np.ma.masked_invalid(pressure_msl_grid)

            # Lokale Minima und Maxima suchen
            minima = (scipy.ndimage.minimum_filter(pressure_msl_masked, size=20, mode='nearest') == pressure_msl_masked)
            maxima = (scipy.ndimage.maximum_filter(pressure_msl_masked, size=20, mode='nearest') == pressure_msl_masked)

            # 4. Positionen der Minima und Maxima bestimmen
            min_locs = np.where(minima)
            max_locs = np.where(maxima)

            # 5. "L" für Tiefdruck, "H" für Hochdruck auf Karte plotten
            for y_idx, x_idx in zip(*min_locs):
                x, y = m(x_pressure[y_idx, x_idx], y_pressure[y_idx, x_idx])
                plt.text(x, y, 'L', 
                         #fontsize=15, 
                         fontweight='bold', ha='center', va='center', color='blue')

            for y_idx, x_idx in zip(*max_locs):
                x, y = m(x_pressure[y_idx, x_idx], y_pressure[y_idx, x_idx])
                plt.text(x, y, 'H', 
                         #fontsize=15, 
                         fontweight='bold', ha='center', va='center', color='red')

            # Optional: Nur starke Hochs und Tiefs markieren   
            # for y_idx, x_idx in zip(*min_locs):
            #     if pressure_msl_grid[y_idx, x_idx] < 1005:
            #         x, y = m(x_pressure[y_idx, x_idx], y_pressure[y_idx, x_idx])
            #         plt.text(x, y, 'L', fontsize=15, fontweight='bold', ha='center', va='center', color='blue')

            # for y_idx, x_idx in zip(*max_locs):
            #     if pressure_msl_grid[y_idx, x_idx] > 1015:
            #         x, y = m(x_pressure[y_idx, x_idx], y_pressure[y_idx, x_idx])
            #         plt.text(x, y, 'H', fontsize=15, fontweight='bold', ha='center', va='center', color='red')

    m.drawparallels(np.arange(0, 360, 2), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(0, 350, 2), labels=[0, 0, 0, 1])
    return heatmap

def run_initial_frame(timepoints):
    fig_in = plt.figure(figsize=(16, 14))
    gs_in = gridspec.GridSpec(2, 1, height_ratios=[1, 2], wspace=0.2, hspace=0.4)
    
    # --- Row 1, Column 1: Pegelplot mit vertikaler Linie ---
    timepoints_np = np.array(timepoints)
    min_timepoint = timepoints_np.min()
    max_timepoint = timepoints_np.max()
    ax1_in = fig_in.add_subplot(gs_in[0, 0])

    df_plot = df_insitu.loc[(df_insitu['time'] >= min_timepoint - datetime.timedelta(hours=1)) & (df_insitu['time'] <= max_timepoint + datetime.timedelta(hours=1))]

    ax1_in.plot(df_plot['time'], df_plot['slev'], color='blue', label='water level (slev)')
    ax1_in.axvline(timepoints[0], color='red', linestyle='--', label='time')
    #ax1_in.set_xlabel('time')
    ax1_in.set_ylabel('water level (m)')
    ax1_in.set_xlim(df_plot['time'].min(), df_plot['time'].max())
    ax1_in.set_ylim(df_insitu['slev'].min(), df_insitu['slev'].max())
    ax1_in.set_title('water level in Flensburg')
    ax1_in.legend(bbox_to_anchor=(0.95, 0.9)) # x, y
    ax1_in.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig_in.autofmt_xdate()  # Optional: Achsenbeschriftungen rotieren
    ax1_in.grid(True)


    # --- Row 2: Kartenvisualisierung (bestehende Funktion) ---
    ax2_in = fig_in.add_subplot(gs_in[1, :])  # Spannt sich über beide Columns
    heatmap = animation_plot(timepoints[0], fig=fig_in, ax=ax2_in)
    cbar_ax = fig_in.add_axes([0.25, 0.05, 0.5, 0.03]) # links, unten, breite, höhe
    fig_in.colorbar(heatmap, cax=cbar_ax, orientation="horizontal").set_label("water level (m)")
    legend_elements = [Line2D([0], [0], color='green', marker='*', linestyle='', markersize=15, label='Flensburg')]
    legend_elements.append(Line2D([0], [0], color='black', lw=4, marker=r'$\rightarrow$', label='wind direction', linestyle=''))
    legend_elements.append(Line2D([0], [0], color='grey', lw=4, marker=r'$\rightarrow$', label='ocean current', linestyle=''))
    fig_in.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.88, 0.59), frameon=True, title="Legend")

    # Save the initial frame
    min_timepoint_str = min_timepoint.strftime("%Y-%m-%d %H:%M:%S")
    max_timepoint_str = max_timepoint.strftime("%Y-%m-%d %H:%M:%S")
    fig_in.savefig(f"initial_frame_{min_timepoint_str}-{max_timepoint_str}", dpi=300)



def update_animation(timepoints, frame_idx, fig:plt.Figure, axs:plt.Axes, gs:mpl.gridspec.GridSpec):
    if timepoints[frame_idx] == timepoints[0]:
        run_initial_frame(timepoints)

    fig.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], figure=fig, wspace=0.1, hspace=0.4)
    
    # --- Row 1, Column 1: Pegelplot mit vertikaler Linie ---
    timepoints_np = np.array(timepoints)
    min_timepoint = timepoints_np.min()
    max_timepoint = timepoints_np.max()
    ax1 = fig.add_subplot(gs[0, 0])

    df_plot = df_insitu.loc[(df_insitu['time'] >= min_timepoint - datetime.timedelta(hours=1)) & (df_insitu['time'] <= max_timepoint + datetime.timedelta(hours=1))]
    df_plot['time'] = pd.to_datetime(df_plot['time'])
    ax1.plot(df_plot['time'], df_plot['slev'], color='blue', label='water level (slev)')

    # Highlight storm surge classes with colored bands
    flood_levels = [
        (1.0, 1.25, 'yellow', 'storm surge'),
        (1.25, 1.5, 'orange', 'medium storm surge'),
        (1.5, 2.0, 'red', 'heavy storm surge'),
        (2.0, 3.5, 'darkred', 'very heavy storm surge'),
    ]

    for y0, y1, color, label in flood_levels:
        ax1.axhspan(y0, y1, facecolor=color, alpha=0.3, label=label)

    ax1.axvline(timepoints[frame_idx], color='red', linestyle='--', label='time')
    #ax1.set_xlabel('time')
    ax1.set_ylabel('water level (m)')
    ax1.set_xlim(df_plot['time'].min(), df_plot['time'].max())
    
    ax1.set_title('water level in Flensburg')
    # ax1.legend(
    #         #title="Beaufort Scale",
    #         loc="upper left",
    #         #fontsize=12,
    #         #title_fontsize=10,
    #         framealpha=0.0,
    #         facecolor='white',
    #         #edgecolor='gray',
    #         ncol=1,
    #         fancybox=True,
    #         shadow=False,
    #         #borderaxespad=0.3,
    #         bbox_to_anchor=(1.02, 0.88),
    #     )
    ax1.legend(
        loc='lower left', 
        #bbox_to_anchor=(0.92, 0.88)
               ) # x, y    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()  # Optional: Achsenbeschriftungen rotieren
    ax1.grid(True)
    ax1.set_ylim(df_insitu['slev'].min() - 0.5, df_insitu['slev'].max() + 0.5)
    #plt.subplots_adjust(right=0.8)  # Platz rechts schaffen

    # --- Row 2: Kartenvisualisierung (bestehende Funktion) ---
    ax2 = fig.add_subplot(gs[1, :])  # Spannt sich über beide Columns
    heatmap = animation_plot(timepoints[frame_idx], fig=fig, ax=ax2)
    cbar_ax = fig.add_axes([0.25, 0.15, 0.5, 0.02])  # links, unten, breite, höhe
    fig.colorbar(heatmap, cax=cbar_ax, orientation="horizontal", pad=0.05).set_label("water level (m)")
    legend_elements = []
    # legend_elements.append(Line2D([0], [0], color='blue', lw=4, label='water level (m)'))
    # legend_elements.append(Line2D([0], [0], color='yellow', lw=4, label='storm surge', alpha=0.3))
    # legend_elements.append(Line2D([0], [0], color='orange', lw=4, label='medium storm surge', alpha=0.3))
    # legend_elements.append(Line2D([0], [0], color='red', lw=4, label='heavy storm surge', alpha=0.3))
    # legend_elements.append(Line2D([0], [0], color='darkred', lw=4, label='very heavy storm surge', alpha=0.3))
    legend_elements.append(Line2D([0], [0], color='green', marker='*', linestyle='', markersize=15, label='Flensburg'))
    legend_elements.append(Line2D([0], [0], color='black', lw=4, marker=r'$\rightarrow$', label='wind direction', linestyle=''))
    legend_elements.append(Line2D([0], [0], color='grey', lw=4, marker=r'$\rightarrow$', label='ocean current', linestyle=''))


    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.88, 0.55), frameon=True)



    

def start_full_animation(timepoints, fps=6, save_as_gif=False):
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], figure=fig)
    timepoints_np = np.array(timepoints)
    min_timepoint = timepoints_np.min().strftime("%Y-%m-%d %H:%M:%S")
    max_timepoint = timepoints_np.max().strftime("%Y-%m-%d %H:%M:%S")


        

    def update(frame_idx):
        current_time = timepoints[frame_idx]
        update_animation(timepoints,frame_idx, fig, None, gs)
        fig.suptitle(f"Zeitpunkt: {current_time}", 
                     #fontsize=18
                     )

    anim = FuncAnimation(fig, update, frames=len(timepoints), interval=1000/fps)

    if save_as_gif:
        writer = PillowWriter(fps=fps)
        anim.save(f'ocean_weather_animation_{min_timepoint}-{max_timepoint}.gif', writer=writer)
    else:
        plt.show()



if __name__ == "__main__":

    timepoints_all = sorted(set(df_ocean["time"]) & set(df_weather["time"]))

    sturm_surge_list = [datetime.datetime(2023, 2, 25, 17, 0),
                    datetime.datetime(2023, 4, 1, 12, 0),
                    datetime.datetime(2023, 10, 7, 20, 0),
                    datetime.datetime(2023, 10, 20, 0, 0),
                    datetime.datetime(2024, 1, 3, 9, 0),
                    datetime.datetime(2024, 2, 9, 18, 0),
                    datetime.datetime(2024, 12, 9, 16, 0),
                    ]
    
    for sturm_surge_time in sturm_surge_list:
        timepoints = [t for t in timepoints_all if t >= sturm_surge_time - datetime.timedelta(days=3) and t <= sturm_surge_time + datetime.timedelta(days=3)]
        print(f"\nRunning animation for storm surge at {sturm_surge_time}...")
        start_full_animation(timepoints, fps=6, save_as_gif=True)
    
    
    # timepoints = [t for t in timepoints_all if pd.Timestamp("2023-10-17 00:00:00") <= t <= pd.Timestamp("2023-10-17 04:00:00")]
    # #timepoints = [t for t in timepoints if pd.Timestamp("2023-10-21 00:00:00") <= t <= pd.Timestamp("2023-10-21 03:00:00")]
    # print('\nRunning animation...')
    # start_full_animation(timepoints, fps=6, save_as_gif=True)
