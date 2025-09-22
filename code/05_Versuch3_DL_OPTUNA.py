
# ## Import Libraries

# Standard Libraries
import warnings

# Scientific Libraries
import numpy as np

import matplotlib.pyplot as plt
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
)
from utils.Model_ConvLSTM import CNNLSTM_Regression_Classification, CNNLSTM_Regression
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
HORIZON = 24                      # 1 Tage Vorhersagehorizont
SEQUENCE_LENGTH = 24              # 1 Tag als Input-Sequenz

classification = False
light_mode = False

if classification:
    print("Classification Mode: Using custom score with bins [1, 2.00]")
else:
    print("Regression Mode: Using mean squared error as loss function")

if light_mode:
    print("Light Mode: Using reduced dataset for faster training")
else:
    print("Full Mode: Using full dataset for training")

if classification:
    storage = f"sqlite:///Versuch3_ConvLSTM_CustomScore_cv_light_mode={light_mode}.db"  # Speicherort für Optuna Studien
else:
    storage = f"sqlite:///Versuch3_ConvLSTM_Regression_cv_light_mode={light_mode}.db"

study_name = f"{HORIZON}"

# print which device is used
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
    gpu_name = torch.cuda.get_device_name(0)
else:
    gpu_name = "CPU"
    print("Using CPU")



# Load data
X, y_lagged, y, common_time = load_picture_lagged_data(
    return_common_time=True,
    verbose=False,
    grid_size=25,
    n_jobs=n_jobs,
    dtype=DTYPE_NUMPY,
    pca=False,
    keep_ocean_data=True,
    land_values=0.0,
    light_mode=light_mode
)



# Daten vorbereiten
X = X.astype(DTYPE_NUMPY)
y_lagged = y_lagged.astype(DTYPE_NUMPY)
y = y.astype(DTYPE_NUMPY)

# Cross-Validation Zeitpunkte
folds = {
    "Surge1": pd.Timestamp("2023-02-25 16:00:00"),
    "Surge2": pd.Timestamp("2023-04-01 09:00:00"),
    "Surge3": pd.Timestamp("2023-10-07 20:00:00"),
    "Surge4": pd.Timestamp("2023-10-20 21:00:00"),
    "Surge5": pd.Timestamp("2024-01-03 01:00:00"),
    "Surge6": pd.Timestamp("2024-02-09 15:00:00"),
    "Surge7": pd.Timestamp("2024-12-09 10:00:00"),
    # "normal1": pd.Timestamp("2023-07-01 14:00:00"),
    # "normal2": pd.Timestamp("2024-04-01 18:00:00"),
    # "normal3": pd.Timestamp("2025-01-01 12:00:00"),
}





def custom_score(y_true=None, y_pred=None, bins=[1, 2.00], alpha=0.7):
    
    # Initialisiere Recall- und MSE-Werte
    recalls = []
    for i in range(y_true.shape[1]):  # Iteriere über jede Spalte
        y_true_class = np.digitize(y_true[:, i], bins=bins)
        y_pred_class = np.digitize(y_pred[:, i], bins=bins)
        recalls.append(recall_score(y_true_class, y_pred_class, average="macro"))
    
    mean_recall = np.mean(recalls)  # Durchschnittlicher Recall
    mse = mean_squared_error(y_true, y_pred)
    return alpha * (1 - mean_recall) + (1 - alpha) * mse

def cross_validation_loop(model_name, folds, X, y_lagged, y, common_time, time_delta, trial_params, trial:optuna.Trial=None, batch_size=32):
    
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

        X_train, y_lagged_train, y_train = create_sequences(X_train, y_lagged_train, y_train, SEQUENCE_LENGTH, HORIZON)
        X_test, y_lagged_test, y_test = create_sequences(X_test, y_lagged_test, y_test, SEQUENCE_LENGTH, HORIZON)

        gap = 168
        X_test = X_test[gap:-gap]
        y_lagged_test = y_lagged_test[gap:-gap]
        y_test = y_test[gap:-gap]

        # check if X_test, y_lagged_test, y_test have NaN values
        if np.isnan(X_test).any() or np.isnan(y_lagged_test).any() or np.isnan(y_test).any():
            print(f"\nNaN values found in test data for {surge_name}.")
            print(f"X_test shape: {X_test.shape}, y_lagged_test shape: {y_lagged_test.shape}, y_test shape: {y_test.shape}")


        X_train, y_lagged_train, y_train, _, _, _, X_test, y_lagged_test, y_test, X_scalers, y_lagged_scalers = scale_data(
            X_train=X_train, y_lagged_train=y_lagged_train, y_train=y_train,
            X_val=None, y_lagged_val=None, y_val=None,
            X_test=X_test, y_lagged_test=y_lagged_test, y_test=y_test,
            dtype=DTYPE_NUMPY, verbose=True
        )

        X_train_tensor, y_lagged_train_tensor, y_train_tensor, _, _, _, X_test_tensor, y_lagged_test_tensor, y_test_tensor = convert_to_tensors(
            X_train=X_train, y_lagged_train=y_lagged_train, y_train=y_train,
            X_val=None, y_lagged_val=None, y_val=None,
            X_test=X_test, y_lagged_test=y_lagged_test, y_test=y_test,
            dtype=torch.float32
        )
        
        

        if classification:
            model = CNNLSTM_Regression_Classification(
                in_channels=X_train_tensor.shape[2],
                forecast_horizon=HORIZON,
                lagged_input_dim=y_lagged_train_tensor.shape[2],
                H=X_train_tensor.shape[3],
                W=X_train_tensor.shape[4],
                cnn1_out_channels=trial_params["cnn1_out_channels_cat"],
                cnn2_out_channels=trial_params["cnn2_out_channels_cat"],
                cnn1_kernel_size=trial_params["cnn1_kernel_size_cat"],
                cnn1_padding=trial_params["cnn1_padding_cat"],
                cnn2_kernel_size=trial_params["cnn2_kernel_size_cat"],
                cnn2_padding=trial_params["cnn2_padding_cat"],
                cnn_linear_out_features=trial_params["cnn_linear_out_features_cat"],
                lstm_hidden_dim=trial_params["lstm_hidden_dim_cat"],
                lstm_layers=trial_params["lstm_layers"],
                lstm_input_size=trial_params["lstm_input_size_cat"],
                dropout=trial_params["dropout"]
            )
        else:
            model = CNNLSTM_Regression(
                in_channels=X_train_tensor.shape[2],
                forecast_horizon=HORIZON,
                lagged_input_dim=y_lagged_train_tensor.shape[2],
                H=X_train_tensor.shape[3],
                W=X_train_tensor.shape[4],
                cnn1_out_channels=trial_params["cnn1_out_channels_cat"],
                cnn2_out_channels=trial_params["cnn2_out_channels_cat"],
                cnn1_kernel_size=trial_params["cnn1_kernel_size_cat"],
                cnn1_padding=trial_params["cnn1_padding_cat"],
                cnn2_kernel_size=trial_params["cnn2_kernel_size_cat"],
                cnn2_padding=trial_params["cnn2_padding_cat"],
                cnn_linear_out_features=trial_params["cnn_linear_out_features_cat"],
                lstm_hidden_dim=trial_params["lstm_hidden_dim_cat"],
                lstm_layers=trial_params["lstm_layers"],
                lstm_input_size=trial_params["lstm_input_size_cat"],
                dropout=trial_params["dropout"]
            )

        lr = trial_params["lr"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if gpu_name == "NVIDIA H100 PCIe" and torch.cuda.is_available():
            print("Using H100 GPU")
            
        else:
            print("Using non-H100 GPU")
            

        
        print(f"Training {model_name} on fold {surge_name} with learning rate {lr}...")
        print("Shapes:")
        print(f"X_train_tensor: {X_train_tensor.shape}")
        print(f"y_train_tensor: {y_train_tensor.shape}")
        print(f"X_test_tensor: {X_test_tensor.shape}")
        print(f"y_test_tensor: {y_test_tensor.shape}")

        best_model = training_ConvLSTM_Regression_Classification(
            model,
            X_train=X_train_tensor,
            y_train=y_train_tensor,
            X_val=X_test_tensor,
            y_val=y_test_tensor,
            y_lagged_train=y_lagged_train_tensor,
            y_lagged_val=y_lagged_test_tensor,
            epochs=trial_params["epochs"],
            batch_size=batch_size,
            optimizer=optimizer,
            writer=writer,
            verbose=True,
            log_tensorboard=True,
            patience=trial_params["patience"],
            trial=trial,
            use_amp=False,
            classification_loss=classification,
            reduce_lr_patience=trial_params["reduce_lr_patience"],
        )

        model.eval()
        with torch.no_grad():
            if classification:
                y_pred_test, class_out = model.predict(X_test_tensor, y_lagged_test_tensor)
            else:
                y_pred_test = model.predict(X_test_tensor, y_lagged_test_tensor)


            if torch.isnan(y_pred_test).any() or (classification and class_out.isnan().any()):
                print(f"⚠ NaN values found in predictions for {surge_name}.")
                print(f"y_pred shape: {y_pred_test.shape}, y_pred NaNs: {torch.isnan(y_pred_test).sum()}")
                if classification:
                    print(f"class_out shape: {class_out.shape}, class_out NaNs: {torch.isnan(class_out).sum()}")

                return np.inf

            y_pred = y_pred_test.cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()
            
            
        print(f"Fold: {surge_name}, y_pred shape: {y_pred_test.shape}, y_true shape: {y_true.shape}")
        score = custom_score(y_true=y_true, y_pred=y_pred, bins=[1, 1.25, 1.5, 2.0], alpha=0.7)
        fold_results.append(score)



        # Report intermediate results to Optuna
        mean_score_so_far = np.mean(fold_results)
        trial.report(mean_score_so_far, step=len(fold_results))
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at fold {surge_name} with score {mean_score_so_far}.")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        else:
            # Plot
            # Annahme: reg_out und y_test haben shape (961, 24)
            plt.figure(figsize=(15, 6))

            num_blocks = y_pred_test.shape[0]

            for i in range(0, num_blocks, HORIZON):
                y_pred_block = y_pred_test[i].cpu().numpy()
                y_true_block = y_test_tensor[i].cpu().numpy()
                
                plt.plot(range(i, i + HORIZON), y_pred_block, label=f'Pred Block {i}', color='blue', alpha=1.0)
                plt.plot(range(i, i + HORIZON), y_true_block, label=f'True Block {i}', color='orange', alpha=1.0)

            plt.xlabel("Time Steps")
            plt.ylabel("Value")
            plt.title(f"Predicted vs. True Values (every 24h block) with lr:{lr}")
            plt.legend(['Predicted', 'True'])
            plt.savefig(f"predicted_vs_true_{surge_name}_score_{score}_classification_{classification}_HORIZON_{HORIZON}.png")

        torch.cuda.empty_cache()

    return fold_results


def objective(trial: optuna.Trial):
    trial_params = {
        "cnn1_out_channels_cat": trial.suggest_categorical("cnn1_out_channels_cat", [64, 128, 256, 512]),
        "cnn2_out_channels_cat": trial.suggest_categorical("cnn2_out_channels_cat", [64, 128, 256, 512]),
        "cnn_linear_out_features_cat": trial.suggest_categorical("cnn_linear_out_features_cat", [64, 128, 256, 512]),
        "cnn1_kernel_size_cat": trial.suggest_categorical("cnn1_kernel_size_cat", [3]),
        "cnn2_kernel_size_cat": trial.suggest_categorical("cnn2_kernel_size_cat", [3]),
        "cnn1_padding_cat": trial.suggest_categorical("cnn1_padding_cat", [1]),
        "cnn2_padding_cat": trial.suggest_categorical("cnn2_padding_cat", [1]),
        "lstm_input_size_cat": trial.suggest_categorical("lstm_input_size_cat", [64, 128, 256, 512]),
        "lstm_hidden_dim_cat": trial.suggest_categorical("lstm_hidden_dim_cat", [64, 128, 256, 512]),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.7),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "epochs": 1000,
        "patience": trial.suggest_int("patience", 30, 50, step=5),
        "reduce_lr_patience": trial.suggest_int("reduce_lr_patience", 3, 30, step=3),
    }
    
    try:
        scores = cross_validation_loop(
            model_name="ConvLSTM",
            folds=folds,
            X=X.astype(DTYPE_NUMPY),
            y_lagged=y_lagged.astype(DTYPE_NUMPY),
            y=y.astype(DTYPE_NUMPY),
            common_time=common_time,
            time_delta=pd.Timedelta(hours=168 * 4),
            trial_params=trial_params,
            trial=trial,
            batch_size= trial_params["batch_size"],
        )
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("⚠ Out of memory! Trial skipped.")

            torch.cuda.empty_cache()
            print("Reducing Batch Size and retrying...")
    
            return np.inf
        else:
            raise e

    return np.mean(scores)

pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, load_if_exists=True, pruner=pruner)
study.optimize(objective, n_trials=1000, show_progress_bar=True)  

print("Beste Parameter:", study.best_params)
print("Bester Score:", study.best_value)



