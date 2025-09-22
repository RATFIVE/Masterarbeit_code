# %%
import fastapi
import pandas as pd
import numpy as np
from utils.Database import Database
import os
import json
from dotenv import load_dotenv

# ------------ Initialize Global Variables ------------


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



db = Database(
    db_url=DB_CONFIG["url"], 
    db_name=DB_CONFIG["name"], 
    collection_name=DB_CONFIG["collection"]
    )

db_data_all = db.get_all_data(key="time")
db.close_connection()

df_db = pd.DataFrame(db_data_all).drop(columns=["_id"])



# %%
def process_dataframe(df: pd.DataFrame, convert_time: bool = False, drop_duplicates: bool = False, reorder: bool = False) -> pd.DataFrame:
    """Converts float columns to float32 and rounds latitude/longitude for consistency."""
    
    if drop_duplicates:
        df = df.drop_duplicates(keep='first')
        
    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    df["latitude"] = df["latitude"].astype(np.float32).round(COORDINATE_ROUNDING)
    df["longitude"] = df["longitude"].astype(np.float32).round(COORDINATE_ROUNDING)

    if convert_time and not np.issubdtype(df['time'].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.round("h")

    # put time, latitude, and longitude sla columns first
    if reorder:
        cols = ["time", "latitude", "longitude", "sla"]
        df = df[cols + [col for col in df.columns if col not in cols]]

    return df

# %%
print(df_db.shape)
df_cleaned = process_dataframe(df_db, convert_time=True, drop_duplicates=True, reorder=True)
df_cleaned = df_cleaned.dropna(axis=1, how='all')
display(df_cleaned.info())

# %%
df_cleaned

# %%

# fast api
app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ocean_data")
def read_data():
    return df_cleaned.to_dict(orient="records")


