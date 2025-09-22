 
# # Import Libaries


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import joblib
# import mse 
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from utils.ml_helper_functions import load_data
import optuna
from sklearn.metrics import mean_squared_error, recall_score
from utils.Model_FFNN import FCSequencePredictor
from utils.Model_Training import training

torch.manual_seed(42)

np.random.seed(42)

WRITER = SummaryWriter(log_dir="runs/ConvLSTM")

SEQ_LEN = 168  # Anzahl der Stunden in der Vergangenheit
HORIZON = 168  # Anzahl der Stunden in der Zukunft, die vorhergesagt werden sollen
EPOCHS = 100000
PATIENCE = 100
BATCH_SIZE = 128




def create_sequences(df, seq_len=SEQ_LEN, horizon=HORIZON):
    """
    Erzeugt nur Sequenzen, die innerhalb eines zusammenhängenden Bereichs liegen.
    """

    X_data = df.drop(columns=["y"]).set_index("ds").to_numpy()
    y_data = df[["ds", "y"]].set_index("ds").to_numpy()



    X, y = [], []
    for i in range(len(X_data) - seq_len - horizon + 1):
        window_x = X_data[i:i+seq_len+horizon] 
        window_y = y_data[i:i+seq_len+horizon]

        # Sicherheitscheck: Kein NaN und Index ist lückenlos
        if np.any(np.isnan(window_x)) or np.any(np.isnan(window_y)):
            continue  # überspringen
        X.append(window_x[:seq_len])
        y.append(window_y[seq_len:seq_len+horizon, 0])  
    return np.array(X), np.array(y)
 
# ## Load Data

def process_data():
    df_loaded = load_data()


    df = df_loaded.copy()
    df.drop(columns=["unique_id"], inplace=True)

    # Calculating lagged features for all the columns which are numerical
    for col in df.select_dtypes(include=[np.number]).columns:
        for lag in range(1, 7):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            #df[f'{col}_future_{lag}'] = df[col].shift(-lag)

    df.dropna(inplace=True)

    return df

def split_data(df):
    # get the data from 21.10.2023 und zwei wochen vorher und danach
    time_delta = pd.Timedelta(days=48) # 2 weeks
    surge_date = pd.Timestamp("2023-10-21")
    start_date = surge_date - time_delta
    end_date = surge_date + time_delta

    df_surge = df.loc[df.ds > start_date][df.ds < end_date].copy()
    
    # make NaN values in range of start_date and end_date for all columns except 'ds'
    df.loc[(df.ds > start_date) & (df.ds < end_date), df.columns != 'ds'] = np.nan

    df_train = df.loc[df.ds < '2024-11-01'].copy()
    df_val = df.loc[df.ds >= '2024-11-01'].copy()

    X_train, y_train = create_sequences(df_train, seq_len=SEQ_LEN, horizon=HORIZON)
    X_val, y_val = create_sequences(df_val, seq_len=SEQ_LEN, horizon=HORIZON)
    X_test, y_test = create_sequences(df_surge, seq_len=SEQ_LEN, horizon=HORIZON)

    print("\nTrain shapes:")
    print(X_train.shape, y_train.shape, type(X_train), type(y_train))

    print("\nValidation shapes:")
    print(X_val.shape, y_val.shape, type(X_val), type(y_val))

    print("\nTest shapes:")
    print(X_test.shape, y_test.shape, type(X_test), type(y_test))

    return X_train, y_train, X_val, y_val, X_test, y_test

def scaling_data(scaler, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Skaliert die Daten mit StandardScaler.
    """

    # Scaling y data
    X_train_scaled = x_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print("Scaled shapes:")
    print(X_train_scaled.shape, y_train.shape, type(X_train_scaled), type(y_train))
    print(X_val_scaled.shape, y_val.shape, type(X_val_scaled), type(y_val))
    print(X_test_scaled.shape, y_test.shape, type(X_test_scaled), type(y_test))

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test

def convert_to_tensor(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Konvertiert die skalierten Daten in PyTorch Tensoren.
    """
    # Numpy → Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    print("\nTensor shapes:")
    print(X_train_tensor.shape, y_train_tensor.shape, type(X_train_tensor), type(y_train_tensor))
    print(X_val_tensor.shape, y_val_tensor.shape, type(X_val_tensor), type(y_val_tensor))
    print(X_test_tensor.shape, y_test_tensor.shape, type(X_test_tensor), type(y_test_tensor))

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor


def objective(trial):
    # Hyperparameter
    linear1 = trial.suggest_int("linear1_output", 128, 1024)
    linear2 = trial.suggest_int("linear2_output", 64, 1024)
    linear3 = trial.suggest_int("linear3_output", 32, 1024)
    dropout = trial.suggest_float("dropout", 0.1, 0.9)
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)

    # Modell
    model = FCSequencePredictor(
        input_dim=X_train_tensor.shape[2],
        seq_len=SEQ_LEN,
        linear1_output=linear1,
        linear2_output=linear2,
        linear3_output=linear3,
        output_horizon=HORIZON,
        n_classes=5,
        dropout=dropout,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    model = training(
        model=model,
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        X_val=X_val_tensor,
        y_val=y_val_tensor,
        optimizer=optimizer,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        writer=None,  # Kein TensorBoard hier
        verbose=False,
        log_tensorboard=False,
        log_dropout=False,
    )

    # Evaluation (auf Validation Set!)
    with torch.no_grad():
        X_eval = X_val_tensor.to(next(model.parameters()).device)
        y_pred, _ = model.predict(X_eval)

    y_true = y_val_tensor.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()

    true_classes = np.digitize(y_true, bins=[1.0, 1.25, 1.5, 2.0])
    pred_classes = np.digitize(y_pred, bins=[1.0, 1.25, 1.5, 2.0])

    recall = recall_score(true_classes, pred_classes, average="macro")
    mse = mean_squared_error(y_true, y_pred)

    alpha = 0.7
    score = alpha * (1 - recall) + (1 - alpha) * mse

    return score  # Optuna minimiert diese Funktion






if __name__ == "__main__":


    # Process data
    df = process_data()

    print("\nDataframe head:")
    print(df.head())
    print("\nDataframe tail:")
    print(df.tail())
    print("\nDataframe shape:", df.shape)
    
    # Split data into train, validation and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    
    # Scale the data
    x_scaler = StandardScaler()
    X_train, y_train, X_val, y_val, X_test, y_test = scaling_data(x_scaler, X_train, y_train, X_val, y_val, X_test, y_test)

    # Convert to tensors
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = convert_to_tensor(X_train, y_train, X_val, y_val, X_test, y_test)



    # Optuna starten
    study = optuna.create_study(direction="minimize", study_name="fc_sequence_optimization", storage=f"sqlite:///optuna_study_FFNN_horizon_{HORIZON}.db", load_if_exists=True)

    study.optimize(objective, n_trials=1000)

    print("\nBeste Parameter:")
    print(study.best_trial.params)
    print("Best Score (custom):", study.best_value)


    print("\n Retrain best model with best parameters...")
    # Speichern des besten Modells
    best_params = study.best_trial.params
    best_model = FCSequencePredictor(
        input_dim=X_train_tensor.shape[2],
        seq_len=SEQ_LEN,
        linear1_output=best_params["linear1_output"],
        linear2_output=best_params["linear2_output"],
        linear3_output=best_params["linear3_output"],
        output_horizon=HORIZON,
        n_classes=5,
        dropout=best_params["dropout"],
    )
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"])
    best_model = training(
        model=best_model,
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        X_val=X_val_tensor,
        y_val=y_val_tensor,
        optimizer=best_optimizer,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        writer=WRITER,  # Kein TensorBoard hier
        verbose=False,
        log_tensorboard=True,
        log_dropout=False,
    )

    # === Modell und Scaler speichern ===
    os.makedirs("models", exist_ok=True)
    torch.save(best_model.state_dict(), f"models/fcsequence_model_horizon_{HORIZON}.pth")
    # Speichern des Scalers mit Joblib (empfohlen für sklearn-Objekte)
    joblib.dump(x_scaler, f"models/x_scaler_horizon_{HORIZON}.pkl")
    print("Modell und Scaler wurden gespeichert.")


