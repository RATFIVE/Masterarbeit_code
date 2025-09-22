
# ## Import Libraries

# Standard Libraries
import warnings

# Scientific Libraries
import numpy as np
import optuna
import pandas as pd
import torch
import time
from sklearn.metrics import mean_squared_error, recall_score

# Sklearn
from sklearn.preprocessing import StandardScaler

# Torch Tools
from torch.utils.tensorboard import SummaryWriter

# Custom Utilities
from utils.dl_helper_functions import (
    convert_to_tensors,
    create_sequences,
    load_picture_lagged_data,
    scale_data,
    create_nowcasting_sequences
)
from utils.Model_ConvLSTM import CNNLSTM_Regression_Classification
from utils.Model_Training import training_ConvLSTM_Regression_Classification

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Display all columns in pandas
pd.options.display.max_columns = None

# TensorBoard writer
writer = SummaryWriter()

# Define global constants
DTYPE_NUMPY = np.float32              # Datentyp für numpy Arrays
n_jobs = -1                           # Anzahl CPUs für parallele Prozesse
HORIZON = 1                      # 1 Tage Vorhersagehorizont
SEQUENCE_LENGTH = 24              # 1 Tag als Input-Sequenz

storage = "sqlite:///Versuch1_ConvLSTM_CustomScore.db"  # Speicherort für Optuna Studien
study_name = f"{HORIZON}"

# Load data
X, y_lagged, y, common_time = load_picture_lagged_data(
    return_common_time=True,
    verbose=False,
    grid_size=25,
    n_jobs=n_jobs,
    dtype=DTYPE_NUMPY,
    pca=False,
    keep_ocean_data=True,
)

# Daten vorbereiten
X = X.astype(DTYPE_NUMPY)
y_lagged = y_lagged.astype(DTYPE_NUMPY)
y = y.astype(DTYPE_NUMPY)

# Cross-Validation Zeitpunkte
folds = {
    # "Surge1": pd.Timestamp("2023-02-25 16:00:00"),
    # "Surge2": pd.Timestamp("2023-04-01 09:00:00"),
    # "Surge3": pd.Timestamp("2023-10-07 20:00:00"),
    # "Surge4": pd.Timestamp("2023-10-20 21:00:00"),
    "Surge5": pd.Timestamp("2024-01-03 01:00:00"),
    # "Surge6": pd.Timestamp("2024-02-09 15:00:00"),
    # "Surge7": pd.Timestamp("2024-12-09 10:00:00"),
    # "normal1": pd.Timestamp("2023-07-01 14:00:00"),
    # "normal2": pd.Timestamp("2024-04-01 18:00:00"),
    # "normal3": pd.Timestamp("2025-01-01 12:00:00"),
}


def custom_score(y_true=None, y_pred=None, bins=[1, 2.00], alpha=0.7):

    # Falls y_true oder y_pred 1D sind, füge eine Dimension hinzu
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    
    # Initialisiere Recall- und MSE-Werte
    recalls = []
    for i in range(y_true.shape[1]):  # Iteriere über jede Spalte
        y_true_class = np.digitize(y_true[:, i], bins=bins)
        y_pred_class = np.digitize(y_pred[:, i], bins=bins)
        recalls.append(recall_score(y_true_class, y_pred_class, average="macro"))
    
    mean_recall = np.mean(recalls)  # Durchschnittlicher Recall
    mse = mean_squared_error(y_true, y_pred)
    return alpha * (1 - mean_recall) + (1 - alpha) * mse

def cross_validation_loop(model_name, folds, X, y_lagged, y, common_time, time_delta, trial_params, trial=None):
    fold_results = []

    for surge_name, fold in folds.items():
        start_cutoff = fold - time_delta
        end_cutoff = fold + time_delta
        idx_start_cutoff = np.where(common_time == start_cutoff)[0][0]
        idx_end_cutoff = np.where(common_time == end_cutoff)[0][0]

        X_test = X[idx_start_cutoff:idx_end_cutoff]
        y_lagged_test = y_lagged[idx_start_cutoff:idx_end_cutoff]
        y_test = y[idx_start_cutoff:idx_end_cutoff]

        X_train = X.copy()
        y_lagged_train = y_lagged.copy()
        y_train = y.copy()
        X_train[idx_start_cutoff:idx_end_cutoff] = np.nan
        y_lagged_train[idx_start_cutoff:idx_end_cutoff] = np.nan
        y_train[idx_start_cutoff:idx_end_cutoff] = np.nan

        X_train, y_lagged_train, y_train = create_nowcasting_sequences(X_train, y_lagged_train, y_train, SEQUENCE_LENGTH)
        X_test, y_lagged_test, y_test = create_nowcasting_sequences(X_test, y_lagged_test, y_test, SEQUENCE_LENGTH)

        gap = 168
        X_test = X_test[gap:-gap]
        y_lagged_test = y_lagged_test[gap:-gap]
        y_test = y_test[gap:-gap]

        assert not np.isnan(X_train).any(), "X_train enthält NaN-Werte."
        assert not np.isnan(y_train).any(), "y_train enthält NaN-Werte."


        X_train, y_lagged_train, y_train, _, _, _, X_test, y_lagged_test, y_test, X_scalers, y_lagged_scalers = scale_data(
            X_train=X_train, y_lagged_train=y_lagged_train, y_train=y_train,
            X_val=None, y_lagged_val=None, y_val=None,
            X_test=X_test, y_lagged_test=y_lagged_test, y_test=y_test,
            dtype=DTYPE_NUMPY, verbose=False
        )

        X_train_tensor, y_lagged_train_tensor, y_train_tensor, _, _, _, X_test_tensor, y_lagged_test_tensor, y_test_tensor = convert_to_tensors(
            X_train=X_train, y_lagged_train=y_lagged_train, y_train=y_train,
            X_val=None, y_lagged_val=None, y_val=None,
            X_test=X_test, y_lagged_test=y_lagged_test, y_test=y_test,
            dtype=torch.float32
        )

        model = CNNLSTM_Regression_Classification(
            in_channels=X_train_tensor.shape[2],
            forecast_horizon=HORIZON,
            lagged_input_dim=y_lagged_train_tensor.shape[2],
            H=X_train_tensor.shape[3],
            W=X_train_tensor.shape[4],
            cnn1_out_channels=trial_params["cnn1_out_channels"],
            cnn2_out_channels=trial_params["cnn2_out_channels"],
            cnn1_kernel_size=3,
            cnn1_padding=1,
            cnn2_kernel_size=3,
            cnn2_padding=1,
            cnn_linear_out_features=trial_params["cnn_linear_out_features"],
            lstm_hidden_dim=trial_params["lstm_hidden_dim"],
            lstm_layers=trial_params["lstm_layers"],
            lstm_input_size=trial_params["cnn_linear_out_features"],
            dropout=trial_params["dropout"]
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=trial_params["lr"])

        best_model = training_ConvLSTM_Regression_Classification(
            model,
            X_train=X_train_tensor,
            y_train=y_train_tensor,
            X_val=X_test_tensor,
            y_val=y_test_tensor,
            y_lagged_train=y_lagged_train_tensor,
            y_lagged_val=y_lagged_test_tensor,
            epochs=trial_params["epochs"],
            batch_size=16,
            optimizer=optimizer,
            writer=writer,
            verbose=True,
            log_tensorboard=True,
            patience=20,
            trial=trial,
            use_amp=False
        )

        model.eval()
        with torch.no_grad():
            y_pred, _ = model.predict(X_test_tensor, y_lagged_test_tensor)
            y_pred = y_pred.cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()
            
            
        print(f"Fold: {surge_name}, y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
        score = custom_score(y_true=y_true, y_pred=y_pred, bins=[1.0, 2.00], alpha=0.7)
        fold_results.append(score)

        # Report intermediate results to Optuna
        mean_score_so_far = np.mean(fold_results)
        trial.report(mean_score_so_far, step=len(fold_results))
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at fold {surge_name} with score {mean_score_so_far}.")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        torch.cuda.empty_cache()

    return fold_results


def objective(trial: optuna.Trial):
    trial_params = {
        "cnn1_out_channels": trial.suggest_int("cnn1_out_channels", 16, 64, step=8),
        "cnn2_out_channels": trial.suggest_int("cnn2_out_channels", 32, 128, step=8),
        "cnn_linear_out_features": trial.suggest_int("cnn_linear_out_features", 8, 128, step=8),
        "lstm_hidden_dim": trial.suggest_int("lstm_hidden_dim", 32, 128, step=8),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "lr": trial.suggest_float("lr", 0.0001, 0.1, log=True),
        "epochs": 1000,
    }

    scores = cross_validation_loop(
        model_name="ConvLSTM",
        folds=folds,
        X=X.astype(DTYPE_NUMPY),
        y_lagged=y_lagged.astype(DTYPE_NUMPY),
        y=y.astype(DTYPE_NUMPY),
        common_time=common_time,
        time_delta=pd.Timedelta(hours=168 * 4),
        trial_params=trial_params,
        trial=trial
    )

    return np.mean(scores)

pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, load_if_exists=True, pruner=pruner)
study.optimize(objective, n_trials=1000)  

print("Beste Parameter:", study.best_params)
print("Bester Score:", study.best_value)



