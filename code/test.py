import torch
import torch.nn as nn
import numpy as np
from utils.Model_ConvLSTM import CNNLSTM_Regression_Classification
from utils.Model_Training import training_ConvLSTM_Regression_Classification

# --- Deine Klasse importieren oder hier reinkopieren ---
# from dein_modul import CNNLSTM_Regression_Classification

# ----------------------------- Test-Script -----------------------------

def test_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    batch_size = 4
    time_steps = 10
    in_channels = 13
    height = 25
    width = 25
    lagged_dim = 7
    n_classes = 3
    forecast_horizon = 168

    # Dummy input data
    weather_seq = torch.randn(batch_size, time_steps, in_channels, height, width)
    y_lagged = torch.randn(batch_size, time_steps, lagged_dim)

    # Initialize model
    model = CNNLSTM_Regression_Classification(
        in_channels=in_channels,
        forecast_horizon=forecast_horizon,
        lagged_input_dim=lagged_dim,
        n_classes=n_classes,
        H=height,
        W=width
    )

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    weather_seq = weather_seq.to(device)
    y_lagged = y_lagged.to(device)

    # Run forward
    reg_out, class_out = model(weather_seq, y_lagged)

    print("✅ Forward pass successful!")
    print(f"reg_out shape: {reg_out.shape}")         # Expected: (batch_size, forecast_horizon)
    print(f"class_out shape: {class_out.shape}")     # Expected: (batch_size, forecast_horizon, n_classes) 

    # Run predict
    reg_pred, class_pred = model.predict(weather_seq, y_lagged)

    print("✅ Predict pass successful!")
    print(f"reg_pred shape: {reg_pred.shape}")       # Expected: (batch_size, forecast_horizon)
    print(f"class_pred shape: {class_pred.shape}")   # Expected: (batch_size, forecast_horizon)

    # Check for NaNs
    assert not torch.isnan(reg_out).any(), "NaNs detected in reg_out"
    assert not torch.isnan(class_out).any(), "NaNs detected in class_out"
    print("✅ No NaNs detected!")

if __name__ == "__main__":
    test_model()
