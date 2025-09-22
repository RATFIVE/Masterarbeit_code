from utils.Database import Database
from utils.Copernicus import AdvancedCopernicus
from utils.OpenMeteoWeather import OpenMeteoWeather
import pandas as pd
import numpy as np
import datetime
import json
from tqdm import tqdm
import os
import json
from dotenv import load_dotenv

# ------------ Initialize Global Variables ------------
ABSOLUTE_END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# .env-Datei laden
load_dotenv()

# Werte abrufen
ABSOLUTE_END_DATE = os.getenv("ABSOLUTE_END_DATE")  # als String
START_DATE = os.getenv("START_DATE")
END_DATE = os.getenv("END_DATE")

# JSON-String in ein Dictionary umwandeln
BBOX = json.loads(os.getenv("BBOX"))

OUTPUT_FILENAME = os.getenv("OUTPUT_FILENAME")
COORDINATE_ROUNDING = int(os.getenv("COORDINATE_ROUNDING"))

DB_CONFIG = {
    "url": os.getenv("DB_URL"),
    "name": os.getenv("DB_NAME"),
    "collection": os.getenv("DB_COLLECTION_OCEAN_WEATHER")
}
### Display Settings ###
print("\n\n")
print(ABSOLUTE_END_DATE, START_DATE, END_DATE)
print(json.dumps(BBOX, indent=4))
print(json.dumps(DB_CONFIG, indent=4))
print("\n\n")

#latitude
# ------------ Helper Functions ------------
def process_dataframe(df: pd.DataFrame, convert_time: bool = False) -> pd.DataFrame:
    """Converts float columns to float32 and rounds latitude/longitude for consistency."""
    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    df["latitude"] = df["latitude"].round(COORDINATE_ROUNDING)
    df["longitude"] = df["longitude"].round(COORDINATE_ROUNDING)

    if convert_time:
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.round("h")

    return df

# ------------ Fetch Data from AdvancedCopernicus ------------
print("\nFetching data from AdvancedCopernicus...\n")
copernicus = AdvancedCopernicus()
copernicus_data = copernicus.get_subset(
    dataset_id="cmems_mod_bal_phy_anfc_PT1H-i",
    dataset_version="202411",
    variables=["bottomT", "mlotst", "siconc", "sithick", "sla", "so", "sob", "thetao", "uo", "vo", "wo"],
    minimum_longitude=BBOX["min_lon"],
    maximum_longitude=BBOX["max_lon"],
    minimum_latitude=BBOX["min_lat"],
    maximum_latitude=BBOX["max_lat"],
    start_datetime=START_DATE,
    end_datetime=END_DATE,
    minimum_depth=0.5016462206840515,
    maximum_depth=0.5016462206840515,
    coordinates_selection_method="strict-inside",
    disable_progress_bar=False,
    output_filename=OUTPUT_FILENAME
)

df_copernicus = copernicus_data.to_dataframe().reset_index()
df_copernicus = df_copernicus[["time"] + [col for col in df_copernicus.columns if col != "time"]]

# Remove rows where all key variables are NaN
key_vars = ["bottomT", "mlotst", "siconc", "sithick", "sla", "so", "sob", "thetao", "uo", "vo", "wo"]
df_copernicus.dropna(subset=key_vars, how="all", inplace=True, axis=0)
df_copernicus = process_dataframe(df_copernicus, convert_time=True)

# ------------ Fetch Existing Data from Database ------------
db = Database(db_url=DB_CONFIG["url"], db_name=DB_CONFIG["name"], collection_name=DB_CONFIG["collection"])
db_data_all = db.get_all_data(key="time")
db.close_connection()

if db_data_all:
    df_db = pd.DataFrame(db_data_all).drop(columns=["_id"])[["time", "latitude", "longitude"]]
    df_db = process_dataframe(df_db, convert_time=True)
    len_before = len(df_copernicus)
    # Use a performant merge operation instead of looping
    df_copernicus = df_copernicus.merge(df_db, on=["time", "latitude", "longitude"], how="left", indicator=True)
    df_copernicus = df_copernicus[df_copernicus["_merge"] == "left_only"].drop(columns=["_merge"])
    len_after = len(df_copernicus)
    print(f"\nRemoved {len_before - len_after} existing records from the Copernicus data")
    print(f"Reduced data: {len(df_copernicus)} rows\n")

# ------------ Fetch OpenMeteoWeather Data and Upload ------------
unique_times = df_copernicus["time"].unique()

# Extract unique latitude-longitude pairs
lat_lon_list = sorted(set(zip(df_copernicus["latitude"], df_copernicus["longitude"])))
latitudes, longitudes = zip(*lat_lon_list)

print(f"\nUnique locations: {len(lat_lon_list)}, Unique times: {len(unique_times)}\n")

db = Database(db_url=DB_CONFIG["url"], db_name=DB_CONFIG["name"], collection_name=DB_CONFIG["collection"])

NUM_BATCHES = 30
for i in tqdm(range(0, len(unique_times), NUM_BATCHES), desc="\nUploading data to the database", total=len(unique_times) // NUM_BATCHES):
    time = unique_times[i]
    time_str = time.strftime("%Y-%m-%d")

    lat_subset = latitudes[:i + NUM_BATCHES]
    lon_subset = longitudes[:i + NUM_BATCHES]

    

    open_meteo_weather = OpenMeteoWeather(
        latitudes=lat_subset,
        longitudes=lon_subset,
        start_date=time_str,
        end_date=time_str
    )
    df_openweather = open_meteo_weather.get_weather_dataframe()
    df_openweather = df_openweather[["time"] + [col for col in df_openweather.columns if col != "time"]]
    df_openweather = process_dataframe(df_openweather, convert_time=True)

    df_merged = pd.merge(df_copernicus, df_openweather, on=["time", "latitude", "longitude"], how="inner")
    if not df_merged.empty:
        db.upload_many(df_merged.to_dict(orient="records"))
        print(f"Uploaded {len(df_merged)} records to the database\n")
    # if i >= 10:
    #     break
db.close_connection()
