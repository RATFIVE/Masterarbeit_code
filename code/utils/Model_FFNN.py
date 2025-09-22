import torch
import torch.nn as nn


class FCSequencePredictor(nn.Module):
    def __init__(self, 
                 input_dim, 
                 seq_len, 

                 linear1_output=512,
                 linear2_output=256,
                 linear3_output=128,

                 output_horizon=168, 
                 n_classes=5, 
                 dropout=0.3
                 ):
        
        super().__init__()
        self.output_horizon = output_horizon
        self.n_classes = n_classes
        self.flatten_dim = input_dim * seq_len

        self.model = nn.Sequential(
            nn.Linear(self.flatten_dim, linear1_output),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(linear1_output, linear2_output),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(linear2_output, linear3_output),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.reg_out = nn.Linear(linear3_output, output_horizon)
        self.class_out = nn.Linear(linear3_output, output_horizon * n_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, seq_len * input_dim)
        x = self.model(x)

        reg_out = self.reg_out(x)  # (batch_size, output_horizon)
        class_out = self.class_out(x).view(-1, self.output_horizon, self.n_classes)
        return reg_out, class_out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)
            return self.forward(x)







import torch
import torch.nn as nn

class Linear_Regression_Classification(nn.Module):
    def __init__(self,
                 in_channels=13,
                 forecast_horizon=168,
                 lagged_input_dim=7,
                 H=25,
                 W=25,
                 hidden_dim1=512,
                 hidden_dim2=256,
                 dropout=0.1,
                 n_classes=3):
        super().__init__()

        input_dim = in_channels * H * W  # Flattened input per timestep
        combined_input_dim = input_dim + lagged_input_dim

        self.forecast_horizon = forecast_horizon

        self.linear_stack = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim1),
            #nn.LeakyReLU(),
            #nn.BatchNorm1d(hidden_dim1),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(dropout),
        )

        self.reg_out = nn.Linear(hidden_dim2, forecast_horizon)
        self.class_out = nn.Linear(hidden_dim2, forecast_horizon * n_classes)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, weather_seq, y_lagged):
        """
        weather_seq: (B, T, C, H, W)
        y_lagged:    (B, T, lagged_dim)
        """
        B, T, C, H, W = weather_seq.shape

        # Flatten weather_seq per timestep: (B, T, C*H*W)
        weather_seq_flat = weather_seq.view(B, T, -1)

        # Concatenate lagged values: (B, T, C*H*W + lagged_dim)
        x = torch.cat([weather_seq_flat, y_lagged], dim=-1)

        # Average over time dimension: (B, C*H*W + lagged_dim)
        x = x.mean(dim=1)

        # Pass through linear stack
        x = self.linear_stack(x)

        # Outputs
        reg_out = self.reg_out(x)  # (B, forecast_horizon)
        class_out = self.class_out(x).view(B, self.forecast_horizon, -1)  # (B, forecast_horizon, n_classes)

        return reg_out, class_out

    def predict(self, weather_seq, y_lagged):
        self.eval()
        device = next(self.parameters()).device
        weather_seq, y_lagged = weather_seq.to(device), y_lagged.to(device)
        with torch.inference_mode():
            reg_out, class_out = self.forward(weather_seq, y_lagged)
            if torch.isnan(reg_out).any():
                print("⚠ NaNs in reg_out inside predict!")
            reg_out = reg_out.clamp(-1e6, 1e6)
            class_out = torch.argmax(class_out, dim=-1)
            return reg_out, class_out
