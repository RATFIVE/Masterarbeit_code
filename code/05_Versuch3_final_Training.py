import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, recall_score, precision_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from utils.dl_helper_functions import (
    convert_to_tensors,
    create_sequences,
    load_picture_lagged_data,
    scale_data,
)
from utils.Model_ConvLSTM import CNNLSTM_Regression_Classification, CNNLSTM_Regression

from utils.Model_Training import training_ConvLSTM_Regression_Classification
# TensorBoard writer
writer = SummaryWriter()
torch.manual_seed(42)
np.random.seed(42)

# Define global constants
DTYPE_NUMPY = np.float32              # Datentyp für numpy Arrays
n_jobs = -1                           # Anzahl CPUs für parallele Prozesse
HORIZON = 24                      # 1 Tage Vorhersagehorizont
SEQUENCE_LENGTH = 24              # 1 Tag als Input-Sequenz

classification = True
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
    land_values=0.0
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
    "normal1": pd.Timestamp("2023-07-01 14:00:00"),
    "normal2": pd.Timestamp("2024-04-01 18:00:00"),
    "normal3": pd.Timestamp("2025-01-01 12:00:00"),
}

# === Load best parameters ===
study_name = f"{HORIZON}"
study = optuna.load_study(study_name=study_name, storage=storage)
best_params = study.best_params
best_params["epochs"] = 1000  # Optional: Set epochs explicitly

print("Verwende beste Parameter:", best_params)

# === Custom score function ===
def custom_score(y_true, y_pred, bins=[1, 2.00], alpha=0.7):
    recalls = []
    for i in range(y_true.shape[1]):
        y_true_class = np.digitize(y_true[:, i], bins=bins)
        y_pred_class = np.digitize(y_pred[:, i], bins=bins)
        #print(f"y_true_class: {y_true_class}, \ny_pred_class: {y_pred_class}")
        recalls.append(recall_score(y_true_class, y_pred_class, average="macro"))
    mean_recall = np.mean(recalls)
    mse = mean_squared_error(y_true, y_pred)
    return alpha * (1 - mean_recall) + (1 - alpha) * mse

def calculate_class_scores(y_true, y_pred, bins=[1, 2.00]):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_true_class = np.digitize(y_true, bins=bins)
    y_pred_class = np.digitize(y_pred, bins=bins)
    recall = recall_score(y_true_class, y_pred_class, average="macro")
    precision = precision_score(y_true_class, y_pred_class, average="macro")
    f1 = f1_score(y_true_class, y_pred_class, average="macro")
    accuracy = accuracy_score(y_true_class, y_pred_class)

    return recall, precision, f1, accuracy

# === Run cross-validation with best parameters ===
results_df = pd.DataFrame()
fold_results = []
for surge_name, fold in folds.items():
    start_cutoff = fold - pd.Timedelta(hours=168 * 4)
    end_cutoff = fold + pd.Timedelta(hours=168 * 4)
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
    
    use_cat = False
    for key, value in best_params.items():
        if "_cat" in key:
            use_cat = True

    if classification:
        model = CNNLSTM_Regression_Classification(
            in_channels=X_train_tensor.shape[2],
            forecast_horizon=HORIZON,
            lagged_input_dim=y_lagged_train_tensor.shape[2],
            H=X_train_tensor.shape[3],
            W=X_train_tensor.shape[4],
            cnn1_out_channels=best_params["cnn1_out_channels_cat"] if use_cat else best_params["cnn1_out_channels"],
            cnn2_out_channels=best_params["cnn2_out_channels_cat"] if use_cat else best_params["cnn2_out_channels"],
            cnn1_kernel_size=best_params["cnn1_kernel_size_cat"] if use_cat else best_params["cnn1_kernel_size"],
            cnn1_padding=best_params["cnn1_padding_cat"] if use_cat else best_params["cnn1_padding"],
            cnn2_kernel_size=best_params["cnn2_kernel_size_cat"] if use_cat else best_params["cnn2_kernel_size"],
            cnn2_padding=best_params["cnn2_padding_cat"] if use_cat else best_params["cnn2_padding"],
            cnn_linear_out_features=best_params["cnn_linear_out_features_cat"] if use_cat else best_params["cnn_linear_out_features"],
            lstm_hidden_dim=best_params["lstm_hidden_dim_cat"] if use_cat else best_params["lstm_hidden_dim"],
            lstm_layers=best_params["lstm_layers"],
            lstm_input_size=best_params["lstm_input_size_cat"] if use_cat else best_params["lstm_input_size"],
            dropout=best_params["dropout"]
        )
    else:
        model = CNNLSTM_Regression(
            in_channels=X_train_tensor.shape[2],
            forecast_horizon=HORIZON,
            lagged_input_dim=y_lagged_train_tensor.shape[2],
            H=X_train_tensor.shape[3],
            W=X_train_tensor.shape[4],
            cnn1_out_channels=best_params["cnn1_out_channels_cat"] if use_cat else best_params["cnn1_out_channels"],
            cnn2_out_channels=best_params["cnn2_out_channels_cat"] if use_cat else best_params["cnn2_out_channels"],
            cnn1_kernel_size=best_params["cnn1_kernel_size_cat"] if use_cat else best_params["cnn1_kernel_size"],
            cnn1_padding=best_params["cnn1_padding_cat"] if use_cat else best_params["cnn1_padding"],
            cnn2_kernel_size=best_params["cnn2_kernel_size_cat"] if use_cat else best_params["cnn2_kernel_size"],
            cnn2_padding=best_params["cnn2_padding_cat"] if use_cat else best_params["cnn2_padding"],
            cnn_linear_out_features=best_params["cnn_linear_out_features_cat"] if use_cat else best_params["cnn_linear_out_features"],
            lstm_hidden_dim=best_params["lstm_hidden_dim_cat"] if use_cat else best_params["lstm_hidden_dim"],
            lstm_layers=best_params["lstm_layers"],
            lstm_input_size=best_params["lstm_input_size_cat"] if use_cat else best_params["lstm_input_size"],
            dropout=best_params["dropout"]
        )

    lr = best_params["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if gpu_name == "NVIDIA H100 PCIe" and torch.cuda.is_available():
        print("Using H100 GPU")
        
    else:
        print("Using non-H100 GPU")
        

    
    
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
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        optimizer=optimizer,
        writer=writer,
        verbose=True,
        log_tensorboard=True,
        patience=best_params["patience"] if "patience" in best_params else 40,  # Default to 40 if not specified
        trial=None,
        use_amp=False,
        classification_loss=classification,
        reduce_lr_patience=best_params["reduce_lr_patience"] if "reduce_lr_patience" in best_params else 10,  # Default to 10 if not specified
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

            

        y_pred = y_pred_test.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        
        
    print(f"Fold: {surge_name}, y_pred shape: {y_pred_test.shape}, y_true shape: {y_true.shape}")
    
    # calculate custom score
    score = custom_score(y_true=y_true, y_pred=y_pred, bins=[1, 1.25, 1.5, 2.0], alpha=0.7)
    
    # calculate mean squared error
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())

    # calculate class scores
    recall, precision, f1, accuracy = calculate_class_scores(y_true, y_pred, bins=[1, 1.25, 1.5, 2.0])
    results_df = pd.concat([
        results_df,
        pd.DataFrame({
            "fold": [surge_name],
            "score": [score],
            "mse": [mse],
            "recall": [recall],
            "precision": [precision],
            "f1": [f1],
            "accuracy": [accuracy]
        })
    ], ignore_index=True)
    fold_results.append(score)

    # Save the model
    score = round(score, 3)
    torch.save(best_model.state_dict(), f"models/Versuch3_final_conv_lstm_model_cv_{surge_name}_classification={classification}_lightmode={light_mode}.pt")
    print(f"Modell gespeichert als 'Versuch3_final_conv_lstm_model_cv_{surge_name}_classification={classification}_lightmode={light_mode}.pt'")

    # === Save scalers ===
    for i, X_scaler in enumerate(X_scalers):
        joblib.dump(X_scaler, f"models/Versuch3_X_scaler_cv_{i}_{surge_name}_classification={classification}_lightmode={light_mode}.pkl")

    joblib.dump(y_lagged_scalers, f"models/Versuch3_y_lagged_scaler_cv_{surge_name}_classification={classification}_lightmode={light_mode}.pkl")
