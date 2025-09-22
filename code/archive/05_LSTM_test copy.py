 
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
from utils.Model_FFNN import FCSequencePredictor
from utils.Model_Training import training

torch.manual_seed(42)

np.random.seed(42)

WRITER = SummaryWriter(log_dir="runs/ConvLSTM")

SEQ_LEN = 168  # Anzahl der Stunden in der Vergangenheit
HORIZON = 168  # Anzahl der Stunden in der Zukunft, die vorhergesagt werden sollen



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


    lr = 0.001  # Learning rate

    #model = LSTMPredictor(input_dim=X_train.shape[2], hidden_dim=X_train.shape[2]//2, output_horizon=HORIZON, dropout=0.2)
    model = FCSequencePredictor(input_dim=X_train.shape[2], 
                                seq_len=SEQ_LEN, 
                                linear1_output=512,
                                linear2_output=256,
                                linear3_output=128,
                                output_horizon=HORIZON, 
                                n_classes=5, 
                                dropout=0.2
                                )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # Print Model Summary and Info
    print("\nModel Summary:")
    print(model)
    print("\nModel Info:")
    print(f"Input Dimension: {X_train.shape[2]}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Output Horizon: {HORIZON}")
    print(f"Number of Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Optimizer: {optimizer}")
    

    best_model = training(model=model,
                        X_train=X_train_tensor, 
                        y_train=y_train_tensor, 
                        X_val=X_val_tensor, 
                        y_val=y_val_tensor, 
                        #criterion=criterion, 
                        optimizer=optimizer, 
                        epochs=3,
                        writer=WRITER, 
                        batch_size=128, 
                        patience=50,  
                        log_tensorboard=True, 
                        log_dropout=True, 
                        verbose=True)
    

    # === Modell und Scaler speichern ===
    os.makedirs("models", exist_ok=True)
    torch.save(best_model.state_dict(), "models/fcsequence_model.pth")
    # Speichern des Scalers mit Joblib (empfohlen für sklearn-Objekte)
    joblib.dump(x_scaler, "models/x_scaler.pkl")
    print("Modell und Scaler wurden gespeichert.")

    


    # Evaluate the model on the test set
    device = next(best_model.parameters()).device  # holt das device vom Modell
    X_test_tensor = X_test_tensor.to(device)  # verschiebt den Tensor
    y_pred_test, class_pred_test = best_model.predict(X_test_tensor)  # Vorhersage auf dem Testset


    
    y_test = y_test_tensor.cpu().numpy()  # bring y_test_tensor to cpu
    y_pred_test = y_pred_test.cpu().numpy()  # bring y_pred_test to cpu
    class_pred_test = class_pred_test.cpu().numpy()  # bring class_pred_test to cpu

    # calculate MSE
    from sklearn.metrics import mean_squared_error, recall_score

    print("\nShape of class_pred_test:", class_pred_test.shape)
    print("Shape of y_pred_test:", y_pred_test.shape)
    print("Shape of y_test:", y_test.shape)



    # In Klassen umwandeln
    true_classes = np.digitize(y_test.flatten(), bins=[1.0, 1.25, 1.5, 2.0])
    pred_classes = np.digitize(y_pred_test.flatten(), bins=[1.0, 1.25, 1.5, 2.0])

    # Recall Score für alle Klassen (macro = ungewichtetes Mittel)
    recall = recall_score(true_classes, pred_classes, average="macro")

    # MSE
    mse_test = mean_squared_error(y_test.flatten(), y_pred_test.flatten())


    alpha = 0.7
    custom_score = alpha * ( 1- recall) + (1- alpha) * mse_test

    print("Custom Score (alpha = 0.7):", custom_score)
    print("MSE Test:", mse_test)
    print("Average Recall (macro):", recall)

