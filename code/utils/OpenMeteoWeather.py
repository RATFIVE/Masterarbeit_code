import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import json
import os
from dotenv import load_dotenv


## .env-Datei laden
load_dotenv('.env')
api_key = str(os.getenv("API_KEY_OPENMETEO"))
print(f'\nApi key: {api_key}\n', type(api_key))

class OpenMeteoWeatherForecast:
    def __init__(self, latitude, longitude, start_date, end_date):
        self.latitude = latitude
        self.longitude = longitude
        self.start_date = start_date
        self.end_date = end_date
        
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        self.url = "https://api.open-meteo.com/v1/forecast"
        self.params = {
            "latitude": [latitude],
            "longitude": [longitude],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", 
                       "precipitation_probability", "precipitation", "rain", "showers", "snowfall", "snow_depth", 
                       "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", 
                       "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration", 
                       "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_80m", 
                       "wind_speed_120m", "wind_speed_180m", "wind_direction_10m", "wind_direction_80m", 
                       "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m", "temperature_120m", 
                       "temperature_80m", "temperature_180m", "soil_temperature_0cm", "soil_temperature_6cm", 
                       "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", 
                       "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", 
                       "soil_moisture_27_to_81cm"]
        }
        
        # Fetch data from the API
        self.responses = self.openmeteo.weather_api(self.url, params=self.params)
        self.response = self.responses[0]  # Process the first location

    def print_location_info(self):
        print(f"Coordinates {self.response.Latitude()}°N {self.response.Longitude()}°E")
        print(f"Elevation {self.response.Elevation()} m asl")
        print(f"Timezone {self.response.Timezone()}{self.response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {self.response.UtcOffsetSeconds()} s")

    def process_hourly_data(self):
        hourly = self.response.Hourly()

        # Extract hourly data and convert to numpy arrays
        variables = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", 
            "precipitation_probability", "precipitation", "rain", "showers", "snowfall", 
            "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", 
            "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration", 
            "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_80m", 
            "wind_speed_120m", "wind_speed_180m", "wind_direction_10m", "wind_direction_80m", 
            "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m", "temperature_120m", 
            "temperature_80m", "temperature_180m", "soil_temperature_0cm", "soil_temperature_6cm", 
            "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", 
            "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"
        ]
        
        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}

        # Loop through variables and add data to the dictionary
        for idx, var in enumerate(variables):
            hourly_data[var] = hourly.Variables(idx).ValuesAsNumpy()

        # Create a DataFrame from the data
        hourly_dataframe = pd.DataFrame(data=hourly_data)
        return hourly_dataframe













import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

class OpenMeteoWeatherHistory:
    def __init__(self, latitude, longitude, start_date, end_date):
        self.latitude = latitude
        self.longitude = longitude
        self.start_date = start_date
        self.end_date = end_date
        
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=1)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        self.url = "https://archive-api.open-meteo.com/v1/archive"
        self.params = {
            "latitude": [latitude],
            "longitude": [longitude],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", 
                       "precipitation_probability", "precipitation", "rain", "showers", "snowfall", "snow_depth", 
                       "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", 
                       "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration", 
                       "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_80m", 
                       "wind_speed_120m", "wind_speed_180m", "wind_direction_10m", "wind_direction_80m", 
                       "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m", "temperature_120m", 
                       "temperature_80m", "temperature_180m", "soil_temperature_0cm", "soil_temperature_6cm", 
                       "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", 
                       "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", 
                       "soil_moisture_27_to_81cm"],
            "apikey": api_key
        }
        
        # Fetch data from the API
        self.responses = self.openmeteo.weather_api(self.url, params=self.params)
        self.response = self.responses[1]  # Process the first location
       

    def print_location_info(self):
        for response in self.responses:
            print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
            print(f"Elevation {response.Elevation()} m asl")
            print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
            print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
            print()
    
    def concat_dataframes(self, dataframes):
        return pd.concat(dataframes, axis=0)


    def process_hourly_data(self):
        dataframes = []
        for response in self.responses:
            hourly = response.Hourly()

            # Extract hourly data and convert to numpy arrays
            variables = [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", 
                "precipitation_probability", "precipitation", "rain", "showers", "snowfall", 
                "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", 
                "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration", 
                "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_80m", 
                "wind_speed_120m", "wind_speed_180m", "wind_direction_10m", "wind_direction_80m", 
                "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m", "temperature_120m", 
                "temperature_80m", "temperature_180m", "soil_temperature_0cm", "soil_temperature_6cm", 
                "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", 
                "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"
            ]
            
            hourly_data = {"time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}

            # Loop through variables and add data to the dictionary
            for idx, var in enumerate(variables):
                hourly_data[var] = hourly.Variables(idx).ValuesAsNumpy()

            # add location information
            hourly_data["latitude"] = response.Latitude()
            hourly_data["longitude"] = response.Longitude()

            # Create a DataFrame from the data
            hourly_dataframe = pd.DataFrame(data=hourly_data)
            
            # Append the dataframe to the list
            dataframes.append(hourly_dataframe)
        
        # filter dataframes with no data
        dataframes = [df for df in dataframes if not df.empty]

        # bring all dataframes together
        hourly_dataframe = self.concat_dataframes(dataframes)

        # bring time, latitude and longitude to the front
        cols = ['time', 'latitude', 'longitude']
        hourly_dataframe = hourly_dataframe[cols + [col for col in hourly_dataframe.columns if col not in cols]]


        return hourly_dataframe
    



if __name__ == "__main__":
    # Example usage
    latitudes = [52.5, 52.6]
    longitudes = [13.4, 13.5]
    start_date = "2025-01-01"
    end_date = "2025-01-02"

    weather_data = OpenMeteoWeatherHistory(latitudes, longitudes, start_date, end_date)
    #weather_data.print_location_info()
    df = weather_data.process_hourly_data()
    print(df.head())
    print(df.info())
    print(df['latitude'].unique())
    print(df['longitude'].unique())


