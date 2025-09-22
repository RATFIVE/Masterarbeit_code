
# ## Import Libaries


import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from utils.dl_helper_functions import (
    create_sequences,
    load_picture_lagged_data,
    scale_data,
)
from xgboost import XGBRegressor

HORIZON = 48  # 2 days of forecast
INITIAL_TRAINING_SIZE = 24 * 183   # 6 months of data = 4392 h
SEQUENCE_LENGTH = 12  # 1 day of data
DTYPE_NUMPY = np.float32  # Use float32 for numpy arrays
n_jobs = -1  # Use all available CPU cores for parallel processing


# # Load Data
X, y_lagged, y, common_time = load_picture_lagged_data(return_common_time=True, verbose=True, grid_size=5, n_jobs=n_jobs, dtype=DTYPE_NUMPY, pca=True)


# convert X, y_lagged, y to numpy arrays of type float32
X = X.astype(DTYPE_NUMPY)
y_lagged = y_lagged.astype(DTYPE_NUMPY)
y = y.astype(DTYPE_NUMPY)

print(f"X shape: {X.shape}, y_lagged shape: {y_lagged.shape}, y shape: {y.shape}, common_time shape: {common_time.shape}")


# Modellwahl: eines von ['RandomForest', 'SVR', 'XGBoost', 'LGBM', 'Linear']
model_name = "Linear"


def get_model(name):
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=n_jobs)
    elif name == "SVR":
        return MultiOutputRegressor(SVR(), n_jobs=n_jobs)
    elif name == "XGBoost":
        return MultiOutputRegressor(XGBRegressor(n_estimators=500, random_state=42, n_jobs=n_jobs), n_jobs=n_jobs)
    elif name == "LGBM":
        return MultiOutputRegressor(LGBMRegressor(n_estimators=500, random_state=42, n_jobs=n_jobs), n_jobs=n_jobs)
    elif name == "Linear":
        return MultiOutputRegressor(LinearRegression(n_jobs=n_jobs), n_jobs=n_jobs)
    else:
        raise ValueError(f"Unbekanntes Modell: {name}")

folds = [
    pd.Timestamp("2023-02-25 16:00:00"),
    pd.Timestamp("2023-04-01 09:00:00"),
    pd.Timestamp("2023-10-20 21:00:00"),
    pd.Timestamp("2024-01-03 01:00:00"),
    pd.Timestamp("2024-02-09 15:00:00"),
    pd.Timestamp("2024-12-09 10:00:00"),
    pd.Timestamp("2025-01-05 14:00:00"),
]


# get idx in X of folds
folds_idx = [np.where(common_time == fold)[0][0] for fold in folds]

print(f"Folds: {folds}")
print(f"Folds idx: {folds_idx}")
delta = 168 * 4 # 4 weeks in hours
time_delta = pd.Timedelta(hours=delta)

# Scale the data
X_scaler = StandardScaler()
y_lagged_scaler = StandardScaler()

results = pd.DataFrame()

for fold in folds:
    start_cutoff = fold - time_delta
    end_cutoff = fold + time_delta
    print(f"Start Cutoff: {start_cutoff}, End Cutoff: {end_cutoff}")

    idx_start_cutoff = np.where(common_time == start_cutoff)[0][0]
    idx_end_cutoff = np.where(common_time == end_cutoff)[0][0]


    

    # Cut the data to the specified time range
    X_test = X[idx_start_cutoff:idx_end_cutoff]
    y_lagged_test = y_lagged[idx_start_cutoff:idx_end_cutoff]
    y_test = y[idx_start_cutoff:idx_end_cutoff]

    # make from idx_start_cutoff to idx_end_cutoff in X to NaN
    X_train = X.copy()
    X_train[idx_start_cutoff:idx_end_cutoff] = np.nan
    y_lagged_train = y_lagged.copy()
    y_lagged_train[idx_start_cutoff:idx_end_cutoff] = np.nan
    y_train = y.copy()
    y_train[idx_start_cutoff:idx_end_cutoff] = np.nan
    
     
    X_train, y_lagged_train, y_train = create_sequences(X_data=X_train, 
                                                        y_lagged_data=y_lagged_train, 
                                                        y_data=y_train, 
                                                        seq_len=SEQUENCE_LENGTH,
                                                        horizon=24,
                                                        dtype=DTYPE_NUMPY)
    
    X_test, y_lagged_test, y_test = create_sequences(X_data=X_test, 
                                                  y_lagged_data=y_lagged_test, 
                                                  y_data=y_test, 
                                                  seq_len=SEQUENCE_LENGTH,
                                                  horizon=24,
                                                  dtype=DTYPE_NUMPY)
    
    gap = 168  # 7 days in hours
    # delete the first and the last gap hours from X_test, y_lagged_test, y_test
    X_test = X_test[gap:-gap]
    y_lagged_test = y_lagged_test[gap:-gap]
    y_test = y_test[gap:-gap]



    

    data = scale_data(X_scaler=X_scaler,
                      y_lagged_scaler=y_lagged_scaler,
                      X_train=X_train,
                      y_lagged_train=y_lagged_train,
                      y_train=y_train,
                      X_val=None,
                      y_lagged_val=None,
                      y_val=None,
                      X_test=X_test,
                      y_lagged_test=y_lagged_test,
                      y_test=y_test,dtype=DTYPE_NUMPY)

    X_train, y_lagged_train, y_train, _, _, _, X_test, y_lagged_test, y_test = data

    print(f"X_train shape: {X_train.shape}, y_lagged_train shape: {y_lagged_train.shape}, y_train shape: {y_train.shape}")


    # Machine Learning Part
    X_train = np.hstack([X_train.reshape(X_train.shape[0], -1), y_lagged_train.reshape(X_train.shape[0], -1)])
    X_test = np.hstack([X_test.reshape(X_test.shape[0], -1), y_lagged_test.reshape(X_test.shape[0], -1)])
    
    # convert array to dtype
    X_train = X_train.astype(DTYPE_NUMPY)
    y_train = y_train.astype(DTYPE_NUMPY)
    X_test = X_test.astype(DTYPE_NUMPY)
    y_test = y_test.astype(DTYPE_NUMPY)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


    model = get_model(model_name)

    # calculate the time to train
    time_start = time.time()

    model.fit(X_train, y_train) 

    time_end = time.time()
    fit_time = time_end - time_start

    time_start = time.time()
    y_pred = model.predict(X_test)
    time_end = time.time()
    predict_time = time_end - time_start

    # calculate the mse
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Squared Error for fold {fold}: {mse}")

    # ...existing code...
    results = pd.concat([
        results,
        pd.DataFrame([{
            'fold': fold,
            'mse': mse,
            'model': model_name,
            'fit_time': fit_time,
            'predict_time': predict_time
        }])
    ], ignore_index=True)




    print(f"y_pred shape: {y_pred.shape}, y_test shape: {y_test.shape}")
    

    # Beispiel: Plot der ersten Sequenz (Index 0)

    seq_idx = len(y_test) // 2 - 24  # oder ein beliebiger Index von 0 bis 1296

    plt.figure(figsize=(10, 5))
    plt.plot(y_test[seq_idx], label="True")
    plt.plot(y_pred[seq_idx], label="Prediction")
    plt.xlabel("idx")
    plt.ylabel("Wert")
    plt.title(f"Vorhersage vs. Wahrheit für Sequenz {seq_idx}")
    plt.legend()
    plt.grid()
    plt.show()
    # save plot
    plt.savefig(f"{model_name}_fold_{fold}_plot.png", dpi=300)



# save results to csv
results.to_csv(f"{model_name}_results.csv", index=False)




