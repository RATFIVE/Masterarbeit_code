# ================================
# Global Configuration Parameters
# ================================

# Grid resolution for ocean and weather data
OCEAN_POINTS = 30
WEATHER_POINTS = 10

# Reference location (Flensburg)
LAT_FLENSBURG = 54.796001
LON_FLENSBURG = 9.436999

# Geographic subregion of interest
SUB_BOX = {
    "lat_min": 54.0,
    "lat_max": 55.5,
    "lon_min": 9.2,
    "lon_max": 13.0
}


# =======================
# Ocean Variables Dictionary
# =======================
# Source: https://vocab.nerc.ac.uk/collection/P07/current/

OCEAN_DICT = {
    "bottomT": {
        "unit": "°C",
        "description": "Sea water potential temperature at sea floor",
        "explanation": "Potential temperature is the temperature a parcel of sea water would have if moved adiabatically to sea level pressure."
    },
    "mlotst": {
        "unit": "m",
        "description": "Ocean mixed layer thickness defined by sigma theta",
        "explanation": "Mixing layer with fairly uniform temperature and salinity."
    },
    "siconc": {
        "unit": "-",
        "description": "Sea ice area fraction",
        "explanation": "Fraction of a grid cell covered by sea ice."
    },
    "sithick": {
        "unit": "m",
        "description": "Sea ice thickness",
        "explanation": "Vertical extent of sea ice formed from freezing sea water."
    },
    "sla": {
        "unit": "m",
        "description": "Sea surface height above sea level",
        "explanation": ""
    },
    "so": {
        "unit": "$1 / 10^3$",
        "description": "Sea water salinity",
        "explanation": "Salt content of sea water, expressed in parts per thousand."
    },
    "sob": {
        "unit": "$1 / 10^3$",
        "description": "Sea water salinity at sea floor",
        "explanation": "Salinity adjacent to the ocean bottom."
    },
    "thetao": {
        "unit": "°C",
        "description": "Sea water potential temperature",
        "explanation": "Temperature a parcel would have at sea level pressure."
    },
    "uo": {
        "unit": "m/s",
        "description": "Eastward sea water velocity",
        "explanation": "Positive when directed eastward."
    },
    "vo": {
        "unit": "m/s",
        "description": "Northward sea water velocity",
        "explanation": "Positive when directed northward."
    },
    "wo": {
        "unit": "m/s",
        "description": "Upward sea water velocity",
        "explanation": "Positive when directed upward."
    }
}


# =========================
# Weather Variables Dictionary
# =========================

WEATHER_DICT = {
    "apparent_temperature": {
        "unit": "°C",
        "description": "Apparent Temperature",
        "explanation": "Perceived temperature considering wind and humidity."
    },
    "cloud_cover": {
        "unit": "%",
        "description": "Cloud cover Total",
        "explanation": "Total cloud coverage."
    },
    "cloud_cover_high": {
        "unit": "%",
        "description": "Cloud cover High",
        "explanation": "Cloud coverage by high-level clouds."
    },
    "cloud_cover_low": {
        "unit": "%",
        "description": "Cloud cover Low",
        "explanation": "Cloud coverage by low-level clouds."
    },
    "cloud_cover_mid": {
        "unit": "%",
        "description": "Cloud cover Mid",
        "explanation": "Cloud coverage by mid-level clouds."
    },
    "dew_point_2m": {
        "unit": "°C",
        "description": "Dewpoint (2 m)",
        "explanation": "Temperature at which air moisture condenses."
    },
    "et0_fao_evapotranspiration": {
        "unit": "mm",
        "description": "Reference Evapotranspiration (ET₀)",
        "explanation": "Standardized reference evapotranspiration according to FAO."
    },
    "evapotranspiration": {
        "unit": "mm",
        "description": "Evapotranspiration",
        "explanation": "Water loss through evaporation and transpiration."
    },
    "precipitation": {
        "unit": "mm",
        "description": "Precipitation (rain + showers + snow)",
        "explanation": "Total precipitation amount."
    },
    "precipitation_probability": {
        "unit": "%",
        "description": "Precipitation Probability",
        "explanation": "Probability of precipitation."
    },
    "pressure_msl": {
        "unit": "hPa",
        "description": "Sea level Pressure",
        "explanation": "Atmospheric pressure reduced to sea level."
    },
    "rain": {
        "unit": "mm",
        "description": "Rain",
        "explanation": "Precipitation amount due to rain."
    },
    "relative_humidity_2m": {
        "unit": "%",
        "description": "Relative Humidity (2 m)",
        "explanation": "Relative humidity at 2 meters above ground."
    },
    "showers": {
        "unit": "mm",
        "description": "Showers",
        "explanation": "Precipitation amount due to showers."
    },
    "snow_depth": {
        "unit": "cm",
        "description": "Snow Depth",
        "explanation": "Total snow depth on the ground."
    },
    "snowfall": {
        "unit": "cm",
        "description": "Snowfall",
        "explanation": "Precipitation amount due to snow."
    },
    "surface_pressure": {
        "unit": "hPa",
        "description": "Surface Pressure",
        "explanation": "Atmospheric pressure at the surface."
    },
    "temperature_120m": {
        "unit": "°C",
        "description": "Temperature (120 m)",
        "explanation": "Air temperature at 120 meters above ground."
    },
    "temperature_180m": {
        "unit": "°C",
        "description": "Temperature (180 m)",
        "explanation": "Air temperature at 180 meters above ground."
    },
    "temperature_2m": {
        "unit": "°C",
        "description": "Temperature (2 m)",
        "explanation": "Air temperature at 2 meters above ground."
    },
    "temperature_80m": {
        "unit": "°C",
        "description": "Temperature (80 m)",
        "explanation": "Air temperature at 80 meters above ground."
    },
    "vapour_pressure_deficit": {
        "unit": "hPa",
        "description": "Vapour Pressure Deficit",
        "explanation": "Difference between saturation and actual vapor pressure."
    },
    "visibility": {
        "unit": "m",
        "description": "Visibility",
        "explanation": "Visibility distance."
    },
    "weather_code": {
        "unit": "-",
        "description": "Weather Code",
        "explanation": "Classification of weather conditions by code."
    },
    "wind_direction_10m": {
        "unit": "°",
        "description": "Wind Direction (10 m)",
        "explanation": "Wind direction at 10 meters (0° = N, 90° = E, etc.)."
    },
    "wind_direction_120m": {
        "unit": "°",
        "description": "Wind Direction (120 m)",
        "explanation": "Wind direction at 120 meters."
    },
    "wind_direction_180m": {
        "unit": "°",
        "description": "Wind Direction (180 m)",
        "explanation": "Wind direction at 180 meters."
    },
    "wind_direction_80m": {
        "unit": "°",
        "description": "Wind Direction (80 m)",
        "explanation": "Wind direction at 80 meters."
    },
    "wind_gusts_10m": {
        "unit": "km/h",
        "description": "Wind Gusts (10 m)",
        "explanation": "Max gust wind speed at 10 meters."
    },
    "wind_speed_10m": {
        "unit": "km/h",
        "description": "Wind Speed (10 m)",
        "explanation": "Wind speed at 10 meters."
    },
    "wind_speed_120m": {
        "unit": "km/h",
        "description": "Wind Speed (120 m)",
        "explanation": "Wind speed at 120 meters."
    },
    "wind_speed_180m": {
        "unit": "km/h",
        "description": "Wind Speed (180 m)",
        "explanation": "Wind speed at 180 meters."
    },
    "wind_speed_80m": {
        "unit": "km/h",
        "description": "Wind Speed (80 m)",
        "explanation": "Wind speed at 80 meters."
    },
    "wind_u": {
        "unit": "m/s",
        "description": "Eastward Wind Component",
        "explanation": "Positive when directed eastward."
    },
    "wind_v": {
        "unit": "m/s",
        "description": "Northward Wind Component",
        "explanation": "Positive when directed northward."
    },
}


# ======================
# In-Situ Measurements
# ======================

INSITU_DICT = {
    "slev": {
        "unit": "m",
        "description": "Water Level",
        "explanation": ""
    }
}



# ======================
# Colors
# ======================

PRED_COLOR = '#dc6e58'  # Color for forecast lines
TRUE_COLOR = '#003238'      # Color for true values