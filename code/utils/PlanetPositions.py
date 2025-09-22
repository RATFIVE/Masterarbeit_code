from astroquery.jplhorizons import Horizons
import pandas as pd
from astropy.time import Time
import pytz

class PlanetPositions:
    def __init__(self, start_date: str, stop_date: str, step: str = '1h'):
        self.planets = {
            "Mercury": 199, "Venus": 299, "Earth": 399, "Mars": 499,
            "Jupiter": 599, "Saturn": 699, "Uranus": 799, "Neptune": 899, "Moon": 301
        }
        self.epochs = {'start': start_date, 'stop': stop_date, 'step': step}
        self.df_all = None

    def fetch_data(self):
        all_data = []
        
        for planet, planet_id in self.planets.items():
            obj = Horizons(id=planet_id, location='500@399', epochs=self.epochs)
            data = obj.vectors()
            df = data.to_pandas()
            df["planet"] = planet  # Planetenname hinzufügen
            all_data.append(df)
        
        self.df_all = pd.concat(all_data, ignore_index=True)
        
    def convert_time(self):
        if self.df_all is not None:
            # Umwandlung von Julianischem Datum in UTC-Zeit
            self.df_all["datetime_utc"] = self.df_all["datetime_jd"].apply(lambda jd: Time(jd, format='jd').to_datetime())
            
            # Umwandlung von UTC nach Berlin-Zeit
            berlin_tz = pytz.timezone("Europe/Berlin")
            self.df_all["datetime"] = self.df_all["datetime_utc"].apply(lambda dt: berlin_tz.fromutc(dt))
            
            # Entfernen der Zeitzone
            self.df_all["datetime"] = self.df_all["datetime"].dt.tz_localize(None)
            # Entfernen unnötiger Spalten
            self.df_all.drop(columns=["datetime_jd"], inplace=True)
        else:
            raise ValueError("Daten wurden noch nicht geladen. Rufe fetch_data() zuerst auf.")
    
    def save_to_csv(self, filename: str):
        if self.df_all is not None:
            self.df_all.to_csv(filename, index=False)
        else:
            raise ValueError("Keine Daten zum Speichern. Rufe fetch_data() zuerst auf.")
    
    def get_dataframe(self):
        if self.df_all is not None:
            return self.df_all
        else:
            raise ValueError("Keine Daten verfügbar. Rufe fetch_data() zuerst auf.")

# Beispielverwendung
if __name__ == "__main__":
    pp = PlanetPositions(start_date='2023-02-14', stop_date='2023-02-20', step='1h')
    pp.fetch_data()
    pp.convert_time()
    df = pp.get_dataframe()
    print(df.head())
