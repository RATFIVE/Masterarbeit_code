
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_horizon=168, n_classes=5, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.output_horizon = output_horizon
        self.n_classes = n_classes

        self.lstm = nn.LSTM(input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout)
                
        self.fc_hidden1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.reg_out = nn.Linear(hidden_dim // 2, output_horizon)

        # # Klassifikation with sigmoid activation
        # nn.Sequential(
        #     nn.Linear(hidden_dim // 2, output_horizon * n_classes),
        #     nn.Sigmoid()
        # )
        self.class_out = nn.Linear(hidden_dim // 2, output_horizon * n_classes)



    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)

        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Nur letzter Zeitschritt
        x = self.fc_hidden1(x)        # (batch_size, hidden_dim // 2)
        reg_out = self.reg_out(x)            # (batch_size, output_horizon)
        class_out = self.class_out(x)                         # (batch_size, output_horizon * n_classes)
        class_out = class_out.view(-1, self.output_horizon, self.n_classes)  # (batch_size, output_horizon, n_classes)

        return reg_out, class_out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                # Einzelne Sequenz: (seq_len, input_dim)
                x = x.unsqueeze(0)  # → (1, seq_len, input_dim)
                output = self.forward(x)
                return output.squeeze(0)  # → (output_horizon,)
            elif x.dim() == 3:
                # Batch: (batch_size, seq_len, input_dim)
                return self.forward(x)  # → (batch_size, output_horizon)
            else:
                raise ValueError(f"Unexpected input dimension: {x.shape}")

