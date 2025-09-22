import warnings

warnings.filterwarnings('ignore')

# Third-Party
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

# Eigene Module
from utils.ml_helper_functions import load_data

# Konstante (z. B. für saisonale Periode)
SEASONAL_PERIOD = 24  # z. B. 24 Stunden
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Daten laden
df = load_data()

# cut the df in train and test set by splitting in 2/3 and 1/3
train_size = int(len(df) * 0.67)
df_train = df[:train_size]
df_test = df[train_size:]

ts = df_train[['ds', 'y']]
ts.set_index('ds', inplace=True)
ts.index = pd.to_datetime(ts.index)
y = ts['y']

# 🔍 Funktion: Stationarität prüfen
def check_stationarity(series, name=''):
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat, p_value, _, _, critical_values, _ = result
    print(f"[{name}] ADF = {adf_stat:.4f}, p-value = {p_value:.4f}")
    if p_value < 0.05 and adf_stat < critical_values['5%']:
        print("Stationär: d = 0")
        return True
    else:
        print("Nicht stationär: Differenzierung empfohlen")
        return False

# Differenzierung (d) prüfen
print("== Check Stationarity (d) ==")
check_stationarity(y, name="Original")

# Saisonale Differenzierung (D) prüfen
print("\n== Check Seasonality (D) ==")
seasonal_diff = y.diff(SEASONAL_PERIOD)
check_stationarity(seasonal_diff, name="Seasonal Differenced")

# Auto-ARIMA
print("\n== Auto ARIMA Suche startet ==")
model = auto_arima(
    y,
    seasonal=True,
    m=SEASONAL_PERIOD,
    start_p=0,
    start_q=0,
    max_p=3,
    max_q=3,
    start_P=0,
    start_Q=0,
    max_P=1,
    max_Q=1,
    d=None,            # automatisch ermitteln
    D=None,            # automatisch ermitteln
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=False,
    maxiter=50,
    n_jobs=-1,
    random_state=RANDOM_SEED
)

# Modell-Zusammenfassung
print("\n== Bestes Modell ==")
print(model.summary())

