
from utils.Database import Database
from utils.PlanetPositions import PlanetPositions
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import json

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
    "collection": os.getenv("DB_COLLECTION_PLANET")
}

### Display Settings ###
print("\n\n")
print(ABSOLUTE_END_DATE, START_DATE, END_DATE)
print(json.dumps(BBOX, indent=4))
print(json.dumps(DB_CONFIG, indent=4))
print("\n\n")




# ------------ Helper Functions ------------

def process_dataframe(df: pd.DataFrame, convert_time: bool = False) -> pd.DataFrame:
    """Converts float columns to float32 for consistency."""
    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    if convert_time:
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.round("h")

    return df



print("\nGetting data from PlanetPositions...\n")
pp = PlanetPositions(start_date=START_DATE, stop_date=END_DATE, step='1h')
pp.fetch_data()
pp.convert_time()
df_planet = pp.get_dataframe()





# convert colum datetime to YYYY-MM-DD HH:MM:SS
#df_planet['datetime'] = df_planet['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_planet = df_planet.drop(columns=['datetime_str', 'planet']).rename(columns={'datetime':'time', 'targetname':'planet'})
# put time column to the first position
df_planet = df_planet[['time'] + [col for col in df_planet.columns if col != 'time']]

df_planet = process_dataframe(df_planet, convert_time=True)



db = Database(
    db_url=DB_CONFIG["url"],
    db_name=DB_CONFIG["name"],
    collection_name=DB_CONFIG["collection"]
    )
    

db_data_all = db.get_all_data(key="time")
db.close_connection()

if db_data_all:
    df_db = pd.DataFrame(db_data_all).drop(columns=["_id"])[["time", "planet"]]
    df_db = process_dataframe(df_db, convert_time=True)
    len_before = len(df_planet)
    # Use a performant merge operation instead of looping
    df_planet = df_planet.merge(df_db, on=["time", "planet"], how="left", indicator=True)
    df_planet = df_planet[df_planet["_merge"] == "left_only"].drop(columns=["_merge"])
    len_after = len(df_planet)
    print(f"\nRemoved {len_before - len_after} existing records from the Copernicus data")
    print(f"Reduced data: {len(df_planet)} rows\n")



print("\nParsing data to upload to Database...\n")
db = Database(
    db_url=DB_CONFIG["url"],
    db_name=DB_CONFIG["name"],
    collection_name=DB_CONFIG["collection"]
    )

df_planet = process_dataframe(df_planet, convert_time=True)
#print(df_planet[['datetime_utc', 'time']].head())
if not df_planet.empty:
    db.upload_many(df_planet.to_dict(orient="records"))
    print(f"Uploaded {len(df_planet)} records to the database")
else:
    print("No data to upload to database")

db.close_connection()


print("Finished!\n")


