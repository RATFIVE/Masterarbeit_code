# import all necessary libraries
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from utils.Model_ConvLSTM import CNNLSTMWaterLevelModel
from utils.config import (
    OCEAN_POINTS,
)
from utils.dl_helper_functions import (load_picture_lagged_data, 
                                       create_sequences, 
                                       create_train_val_test_split,
                                       scale_data, 
                                       convert_to_tensors)

from torch.utils.tensorboard import SummaryWriter
from utils.Model_Training import training_ConvLSTM
torch.manual_seed(42)  # For reproducibility
np.random.seed(42)  # For reproducibility
# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Display all columns
pd.options.display.max_columns = None

writer = SummaryWriter()
plt.rcParams.update(
    {
        "font.size": 14,  # Grundschriftgröße (wirkt auf alles, sofern nicht überschrieben)
        "axes.titlesize": 16,  # Größe des Titels der Achse (z.B. 'Subplot Title')
        "axes.labelsize": 14,  # Achsenbeschriftung (x/y label)
        "xtick.labelsize": 12,  # X-Tick-Beschriftung
        "ytick.labelsize": 12,  # Y-Tick-Beschriftung
        "legend.fontsize": 12,  # Legendentext
        "figure.titlesize": 18,  # Gesamttitel der Abbildung (plt.suptitle)
        "figure.labelsize": 14,  # (optional, selten verwendet)
        "savefig.dpi": 300,  # DPI beim Speichern
        "figure.dpi": 100,  # DPI bei Anzeige
    }
)

OCEAN_POINTS = 30
GRID_SIZE = 10
HORIZON = 168




x_scaler = StandardScaler()
y_lagged_scaler = StandardScaler()

X, y_lagged, y = load_picture_lagged_data()

X_seq, y_lagged_seq, y_seq = create_sequences(X, y_lagged, y, seq_len=168, horizon=HORIZON)

data = create_train_val_test_split(X=X_seq, 
                                    y_lagged=y_lagged_seq, 
                                    y=y_seq, 
                                    train_percentage=0.5, 
                                    val_percentage=0.2, 
                                    test_percentage=0.3)
# unpack the data
X_train, y_lagged_train, y_train, X_val, y_lagged_val, y_val, X_test, y_lagged_test, y_test = data

data = scale_data(X_scaler=x_scaler,
                    y_lagged_scaler=y_lagged_scaler,

                    X_train=X_train, 
                    y_lagged_train=y_lagged_train, 
                    y_train=y_train, 

                    X_val=X_val, 
                    y_lagged_val=y_lagged_val, 
                    y_val=y_val, 

                    X_test=X_test, 
                    y_lagged_test=y_lagged_test, 
                    y_test=y_test)

# unpack the data
X_train_scaled, y_lagged_train_scaled, y_train_scaled, X_val_scaled, y_lagged_val_scaled, y_val_scaled, X_test_scaled, y_lagged_test_scaled, y_test_scaled = data

data = convert_to_tensors(X_train=X_train_scaled, 
                        y_lagged_train=y_lagged_train_scaled, 
                        y_train=y_train_scaled, 

                        X_val=X_val_scaled, 
                        y_lagged_val=y_lagged_val_scaled, 
                        y_val=y_val_scaled,

                        X_test=X_test_scaled, 
                        y_lagged_test=y_lagged_test_scaled, 
                        y_test=y_test_scaled)

# unpack the data
X_train_tensor, y_lagged_train_tensor, y_train_tensor, X_val_tensor, y_lagged_val_tensor, y_val_tensor, X_test_tensor, y_lagged_test_tensor, y_test_tensor = data

print("\nShapes of training tensors:")
print("X_train_tensor shape:", X_train_tensor.shape)
print("y_lagged_train_tensor shape:", y_lagged_train_tensor.shape)
print("y_train_tensor shape:", y_train_tensor.shape)
print("\nShapes of validation tensors:")
print("X_val_tensor shape:", X_val_tensor.shape)
print("y_lagged_val_tensor shape:", y_lagged_val_tensor.shape)
print("y_val_tensor shape:", y_val_tensor.shape)
print("\nShapes of test tensors:")
print("X_test_tensor shape:", X_test_tensor.shape)
print("y_lagged_test_tensor shape:", y_lagged_test_tensor.shape)
print("y_test_tensor shape:", y_test_tensor.shape)






for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
    print(f"Training model with learning rate: {lr}")
    writer.add_text("hyperparameters", f"Learning Rate: {lr}")

    model = CNNLSTMWaterLevelModel(in_channels=X_train_tensor.shape[2], forecast_horizon=HORIZON, lagged_input_dim=y_lagged_train_tensor.shape[2])



    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_model = training_ConvLSTM(
        model,
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        X_val=X_val_tensor,
        y_val=y_val_tensor,
        y_lagged_train=y_lagged_train_tensor,
        y_lagged_val=y_lagged_val_tensor,
        epochs=epochs,
        batch_size=128,
        optimizer=optimizer,
        writer=writer,
        verbose=True,
        log_tensorboard=True,
    )


    # validate on test set
    model.eval()
    with torch.no_grad():
        y_pred = model.predict(X_test_tensor, y_lagged_test_tensor)
        y_pred = y_pred.cpu().numpy()  # Konvertiere zu NumPy für weitere Verarbeitung


    y_true = y_test_tensor.cpu().numpy()  # Konvertiere zu NumPy für weitere Verarbeitung


    # get date from idx of y


    # Plotten der Vorhersagen gegen die wahren Werte
    now = time.time()
    # format now to string
    now = time.strftime("%Y-%m-%d_%H-%M-%S_", time.localtime(now))
    print(f"Saving plots with prefix: {now}")


    def plot_predictions(y_true, y_pred, n_samples=5):
        """
        Plottet die Vorhersagen gegen die wahren Werte für eine zufällige Auswahl von Samples.
        """
        indices = np.random.choice(len(y_true), n_samples, replace=False)

        plt.figure(figsize=(15, len(indices) * 3))
        for i, idx in enumerate(indices):
            plt.subplot(n_samples, 1, i + 1)
            plt.plot(y_true[idx], label="True", color="blue")
            plt.plot(y_pred[idx], label="Predicted", color="orange")
            plt.title(f"Sample {idx}")
            plt.xlabel("Time Step")
            plt.ylabel("Water Level")
            plt.legend()

        plt.tight_layout()
        # save fig
        #plt.savefig(f"{now}predictions_plot.png", dpi=300)
        plt.show()


    plot_predictions(y_true, y_pred, n_samples=5)
