import torch
import torch.nn as nn
import numpy as np
from utils.Model_ConvLSTM import CNNLSTM_Regression_Classification
from utils.Model_FFNN import Linear_Regression_Classification
from utils.Model_Training import training_ConvLSTM_Regression_Classification
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, recall_score
from sklearn.preprocessing import StandardScaler
from utils.dl_helper_functions import (
    convert_to_tensors,
    create_sequences,
    load_picture_lagged_data,
    scale_data,
)
from utils.Model_ConvLSTM import CNNLSTM_Regression_Classification
from utils.Model_Training import training_ConvLSTM_Regression_Classification

# === SETTINGS ===
DTYPE_NUMPY = np.float32
HORIZON = 24
SEQUENCE_LENGTH = 24
n_jobs = -1

# === Load data ===
X, y_lagged, y, common_time = load_picture_lagged_data(
    return_common_time=True,
    verbose=False,
    grid_size=25,
    n_jobs=n_jobs,
    dtype=DTYPE_NUMPY,
    pca=False,
    keep_ocean_data=True,
)

# X = X.astype(DTYPE_NUMPY)
# y_lagged = y_lagged.astype(DTYPE_NUMPY)
# y = y.astype(DTYPE_NUMPY)

# folds = {
#     "Surge1": pd.Timestamp("2023-02-25 16:00:00"),
#     "Surge2": pd.Timestamp("2023-04-01 09:00:00"),
#     "Surge3": pd.Timestamp("2023-10-07 20:00:00"),
#     "Surge4": pd.Timestamp("2023-10-20 21:00:00"),
#     "Surge5": pd.Timestamp("2024-01-03 01:00:00"),
#     "Surge6": pd.Timestamp("2024-02-09 15:00:00"),
#     "Surge7": pd.Timestamp("2024-12-09 10:00:00"),
#     "normal1": pd.Timestamp("2023-07-01 14:00:00"),
#     "normal2": pd.Timestamp("2024-04-01 18:00:00"),
#     "normal3": pd.Timestamp("2025-01-01 12:00:00"),
# }

# # === Load best parameters ===


# storage = "sqlite:///Versuch3_ConvLSTM_CustomScore.db"
# study_name = f"{HORIZON}"
# study = optuna.load_study(study_name=study_name, storage=storage)
# best_params = study.best_params
# best_params["epochs"] = 1000  # Optional: Set epochs explicitly

# print("Verwende beste Parameter:", best_params)

# # === Custom score function ===


# def custom_score(y_true, y_pred, bins=[1, 2.00], alpha=0.7):
#     recalls = []
#     for i in range(y_true.shape[1]):
#         y_true_class = np.digitize(y_true[:, i], bins=bins)
#         y_pred_class = np.digitize(y_pred[:, i], bins=bins)
#         recalls.append(recall_score(y_true_class, y_pred_class, average="macro"))
#     mean_recall = np.mean(recalls)
#     mse = mean_squared_error(y_true, y_pred)
#     return alpha * (1 - mean_recall) + (1 - alpha) * mse

# # === Run cross-validation with best parameters ===
# results = []
# for surge_name, fold in folds.items():
#     print(f"\n=== Fold: {surge_name} ===")

#     start_cutoff = fold - pd.Timedelta(hours=168 * 4)
#     end_cutoff = fold + pd.Timedelta(hours=168 * 4)
#     idx_start_cutoff = np.where(common_time == start_cutoff)[0][0]
#     idx_end_cutoff = np.where(common_time == end_cutoff)[0][0]

#     X_test = X[idx_start_cutoff:idx_end_cutoff]
#     y_lagged_test = y_lagged[idx_start_cutoff:idx_end_cutoff]
#     y_test = y[idx_start_cutoff:idx_end_cutoff]

#     X_train = X.copy()
#     y_lagged_train = y_lagged.copy()
#     y_train = y.copy()
#     X_train[idx_start_cutoff:idx_end_cutoff] = np.nan
#     y_lagged_train[idx_start_cutoff:idx_end_cutoff] = np.nan
#     y_train[idx_start_cutoff:idx_end_cutoff] = np.nan

#     X_train, y_lagged_train, y_train = create_sequences(X_train, y_lagged_train, y_train, SEQUENCE_LENGTH, HORIZON)
#     X_test, y_lagged_test, y_test = create_sequences(X_test, y_lagged_test, y_test, SEQUENCE_LENGTH, HORIZON)

#     gap = 168
#     X_test = X_test[gap:-gap]
#     y_lagged_test = y_lagged_test[gap:-gap]
#     y_test = y_test[gap:-gap]

#     scaler_X = StandardScaler()
#     scaler_y = StandardScaler()
#     X_train, y_lagged_train, y_train, _, _, _, X_test, y_lagged_test, y_test = scale_data(
#         X_scaler=scaler_X, y_lagged_scaler=scaler_y,
#         X_train=X_train, y_lagged_train=y_lagged_train, y_train=y_train,
#         X_val=None, y_lagged_val=None, y_val=None,
#         X_test=X_test, y_lagged_test=y_lagged_test, y_test=y_test,
#         dtype=DTYPE_NUMPY, verbose=False
#     )

#     X_train_tensor, y_lagged_train_tensor, y_train_tensor, _, _, _, X_test_tensor, y_lagged_test_tensor, y_test_tensor = convert_to_tensors(
#         X_train=X_train, y_lagged_train=y_lagged_train, y_train=y_train,
#         X_val=None, y_lagged_val=None, y_val=None,
#         X_test=X_test, y_lagged_test=y_lagged_test, y_test=y_test,
#         dtype=torch.float32
#     )

#     model = CNNLSTM_Regression_Classification(
#         in_channels=X_train_tensor.shape[2],
#         forecast_horizon=HORIZON,
#         lagged_input_dim=y_lagged_train_tensor.shape[2],
#         H=X_train_tensor.shape[3],
#         W=X_train_tensor.shape[4],
#         cnn1_out_channels=best_params["cnn1_out_channels"],
#         cnn2_out_channels=best_params["cnn2_out_channels"],
#         cnn1_kernel_size=3,
#         cnn1_padding=1,
#         cnn2_kernel_size=3,
#         cnn2_padding=1,
#         cnn_linear_out_features=best_params["cnn_linear_out_features"],
#         lstm_hidden_dim=best_params["lstm_hidden_dim"],
#         lstm_layers=best_params["lstm_layers"],
#         lstm_input_size=best_params["cnn_linear_out_features"],
#         dropout=best_params["dropout"]
#     )

#     optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

#     best_model = training_ConvLSTM_Regression_Classification(
#         model,
#         X_train=X_train_tensor,
#         y_train=y_train_tensor,
#         X_val=X_test_tensor,
#         y_val=y_test_tensor,
#         y_lagged_train=y_lagged_train_tensor,
#         y_lagged_val=y_lagged_test_tensor,
#         epochs=best_params["epochs"],
#         batch_size=128,
#         optimizer=optimizer,
#         writer=None,
#         verbose=True,
#         log_tensorboard=False,
#         patience=20
#     )

#     model.eval()
#     with torch.no_grad():
#         y_pred, _ = model.predict(X_test_tensor, y_lagged_test_tensor)
#         y_pred = y_pred.cpu().numpy()
#         y_true = y_test_tensor.cpu().numpy()

#     score = custom_score(y_true, y_pred, bins=[1.0, 2.00], alpha=0.7)
#     print(f"{surge_name}: Score = {score:.4f}")
#     results.append({"fold": surge_name, "score": score})

# # === Summary ===
# results_df = pd.DataFrame(results)
# print("\n=== Cross-Validation Results with Best Parameters ===")
# print(results_df)
# print("\nMean Score:", results_df["score"].mean())




# # Balkendiagramm der Cross-Validation-Scores
# plt.figure(figsize=(10, 6))
# plt.bar(results_df['fold'], results_df['score'])
# plt.xlabel('Fold')
# plt.ylabel('Custom Score')
# plt.title('Cross-Validation Scores with Best Parameters')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# # save fig 
# plt.savefig('cross_validation_scores.png', dpi=300)  # dpi=300 für hohe Auflösung
# plt.show()











# Train Final Model on All Data but leave out Surge4


X = X.astype(DTYPE_NUMPY)
y_lagged = y_lagged.astype(DTYPE_NUMPY)
y = y.astype(DTYPE_NUMPY)

folds = {
    "Surge4": pd.Timestamp("2023-10-20 21:00:00"),
}

# === Load best parameters ===




# === Custom score function ===


def custom_score(y_true, y_pred, bins=[1, 2.00], alpha=0.7):
    recalls = []
    for i in range(y_true.shape[1]):
        y_true_class = np.digitize(y_true[:, i], bins=bins)
        y_pred_class = np.digitize(y_pred[:, i], bins=bins)
        recalls.append(recall_score(y_true_class, y_pred_class, average="macro"))
    mean_recall = np.mean(recalls)
    mse = mean_squared_error(y_true, y_pred)
    return alpha * (1 - mean_recall) + (1 - alpha) * mse

# === Run cross-validation with best parameters ===
results = []
for surge_name, fold in folds.items():
    print(f"\n=== Fold: {surge_name} ===")

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

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train, y_lagged_train, y_train, _, _, _, X_test, y_lagged_test, y_test = scale_data(
        X_scaler=scaler_X, y_lagged_scaler=scaler_y,
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
        cnn1_out_channels=128,
        cnn2_out_channels=256,
        cnn1_kernel_size=3,
        cnn1_padding=1,
        cnn2_kernel_size=3,
        cnn2_padding=1,
        cnn_linear_out_features=256,
        lstm_hidden_dim=128,
        lstm_layers=1,
        lstm_input_size=256,
        dropout=0.1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_model = training_ConvLSTM_Regression_Classification(
        model,
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        X_val=X_test_tensor,
        y_val=y_test_tensor,
        y_lagged_train=y_lagged_train_tensor,
        y_lagged_val=y_lagged_test_tensor,
        epochs=1000,
        batch_size=16,
        optimizer=optimizer,
        writer=None,
        verbose=True,
        log_tensorboard=False,
        patience=20
    )

    model.eval()
    with torch.no_grad():
        y_pred, _ = model.predict(X_test_tensor, y_lagged_test_tensor)
        y_pred = y_pred.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

    score = custom_score(y_true, y_pred, bins=[1.0, 2.00], alpha=0.7)
    print(f"{surge_name}: Score = {score:.4f}")

    # torch.save(best_model.state_dict(), f"Versuch3_final_conv_lstm_model_{surge_name}_score={score}.pt")
    # print(f"Modell gespeichert als 'Versuch3_final_conv_lstm_model_{surge_name}_score={score}.pt'")

    # # === Save scalers ===
    # joblib.dump(scaler_X, f"Versuch3_scaler_X_{surge_name}_score={score}.pkl")
    # joblib.dump(scaler_y, f"Versuch3_scaler_y_{surge_name}_score={score}.pkl")
    # print(f"Scaler gespeichert als 'Versuch3_scaler_X_{surge_name}_score={score}.pkl' und 'Versuch3_scaler_y_{surge_name}_score={score}.pkl'")



