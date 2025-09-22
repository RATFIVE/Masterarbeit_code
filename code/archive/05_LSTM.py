
# # Import Libaries




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from utils.ml_helper_functions import load_data


torch.manual_seed(42)
np.random.seed(42)
# ## Load Data
df_loaded = load_data()
df = df_loaded.copy()
df.drop(columns=["unique_id"], inplace=True)



# get the data from 21.10.2023 und zwei wochen vorher und danach
time_delta = pd.Timedelta(days=48) # 2 weeks
surge_date = pd.Timestamp("2023-10-21")
start_date = surge_date - time_delta
end_date = surge_date + time_delta

# filter surge data from df
df_surge = df.loc[df.ds > start_date][df.ds < end_date].copy()


# make NaN values in range of start_date and end_date for all columns except 'ds'
df.loc[(df.ds > start_date) & (df.ds < end_date), df.columns != 'ds'] = np.nan






def create_sequences(df, seq_len=168, horizon=168):
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
        y.append(window_y[seq_len:seq_len+horizon, 0])  # z. B. nur Feature 0 als Ziel
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(df, seq_len=168, horizon=168)
X_test, y_test = create_sequences(df_surge, seq_len=168, horizon=168)

print("\nTrain shapes:")
print(X_train.shape, y_train.shape, type(X_train), type(y_train))

print("\nTest shapes:")
print(X_test.shape, y_test.shape, type(X_test), type(y_test))


plt.figure(figsize=(10, 2))
plt.plot(y_test[900], label='Surge Data')
plt.show()


# Scaling x data
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_scaled = x_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

print("\nScaled shapes:")
print(X_train_scaled.shape, y_train.shape, type(X_train_scaled), type(y_train))
print(X_test_scaled.shape, y_test.shape, type(X_test_scaled), type(y_test))


# # Scaling y_train and y_test
# y_scaler = StandardScaler()
# y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
# y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
# print("Scaled y shapes:")
# print(y_train_scaled.shape, type(y_train_scaled))
# print(y_test_scaled.shape, type(y_test_scaled))




# Numpy to Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print("\nTensor shapes:")
print(X_train_tensor.shape, y_train_tensor.shape, type(X_train_tensor), type(y_train_tensor))
print(X_test_tensor.shape, y_test_tensor.shape, type(X_test_tensor), type(y_test_tensor))





# set torch random seed for reproducibility

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_horizon=168):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_horizon)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Nur letzter Zeitschritt
        out = self.fc(last_hidden)        # (batch_size, output_horizon)
        return out






# Dataloader
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMPredictor(input_dim=X_train.shape[2], output_horizon=168).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 1000
model.train()
for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch_train, y_batch_train = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred_train = model(X_batch_train)
        loss = criterion(y_pred_train, y_batch_train)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")



model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    
    y_pred_test = model(X_test_tensor)  # (batch, 168)
    test_loss = criterion(y_pred_test, y_test_tensor)
    
    print(f"\nTest MSE on Sturmflutdaten: {test_loss.item():.4f}")







# Beispiel: erstes Testsample
idx = 920
y_true_plot = y_test_tensor[idx].cpu().numpy()
y_pred_plot = y_pred_test[idx].cpu().numpy()


# inverse scaling
# y_true_plot = y_scaler.inverse_transform(y_true_plot.reshape(-1, 1)).reshape(-1)
# y_pred_plot = y_scaler.inverse_transform(y_pred_plot.reshape(-1, 1)).reshape(-1)

plt.figure(figsize=(12, 5))
plt.plot(y_true_plot, label="True")
plt.plot(y_pred_plot, label="Predicted")
plt.title("Sturmflut-Vorhersage – 168 Stunden")
plt.xlabel("Stunden in der Zukunft")
plt.ylabel("z.B. Wasserstand")
plt.legend()
plt.grid()
# Save the plot
plt.savefig("sturmflut_vorhersage_plot.png")
