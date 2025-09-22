import torch
import torch.nn as nn


class CNNLSTMWaterLevelModel(nn.Module):
    def __init__(self, 
                 in_channels=13,
                 lstm_hidden_dim=64, 
                 lstm_layers=2, 
                 forecast_horizon=168,
                 lagged_input_dim=7,
                 H=25,
                 W=25,
                 cnn1_out_channels=32,
                 cnn1_kernel_size=3,
                 cnn1_padding=1,
                 cnn2_out_channels=128,
                 cnn2_kernel_size=3,
                 cnn2_padding=1,
                 cnn_linear_out_features=128,
                 lstm_input_size=128,
                 dropout=0.5
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.forecast_horizon = forecast_horizon

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=cnn1_out_channels, kernel_size=cnn1_kernel_size, padding=cnn1_padding),  
            nn.LeakyReLU(),
            nn.BatchNorm2d(cnn1_out_channels),
            nn.Conv2d(cnn1_out_channels, cnn2_out_channels, kernel_size=cnn2_kernel_size, padding=cnn2_padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cnn2_out_channels),
            nn.Flatten(),
            nn.Linear(cnn2_out_channels * H * W, cnn_linear_out_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.lstm_input_linear = nn.Linear(cnn_linear_out_features + lagged_input_dim, lstm_input_size)
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=self.lstm_hidden_dim,
                            num_layers=self.lstm_layers,
                            batch_first=True)
        self.lstm_bn = nn.BatchNorm1d(self.lstm_hidden_dim)
        self.lstm_act = nn.LeakyReLU()
        self.lstm_dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(self.lstm_hidden_dim, self.forecast_horizon)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, weather_seq, y_lagged):
        """
        weather_seq: (B, T, C, H, W)
        y_lagged:    (B, T, lagged_dim)
        """
        B, T, C, H, W = weather_seq.shape

        cnn_features = []
        for t in range(T):
            x_t = weather_seq[:, t]  # (B, C, H, W)
            feat_t = self.cnn(x_t)   # (B, cnn_feat)
            cnn_features.append(feat_t)
        cnn_seq = torch.stack(cnn_features, dim=1)  # (B, T, cnn_feat)

        lstm_input = torch.cat([cnn_seq, y_lagged], dim=-1)  # (B, T, D)
        lstm_input = self.lstm_input_linear(lstm_input)  # (B, T, lstm_input_size)

        lstm_out, _ = self.lstm(lstm_input)  # (B, T, lstm_hidden_dim)
        final_hidden = lstm_out[:, -1, :]  # (B, lstm_hidden_dim)

        final_hidden = self.lstm_bn(final_hidden)
        final_hidden = self.lstm_act(final_hidden)
        final_hidden = self.lstm_dropout(final_hidden)

        return self.fc_out(final_hidden)  # (B, forecast_horizon)

    def predict(self, weather_seq, y_lagged):
        weather_seq = weather_seq.to(self.device)
        y_lagged = y_lagged.to(self.device)
        self.eval()
        with torch.no_grad():
            return self.forward(weather_seq, y_lagged)





class CNNLSTM_Regression_Classification(nn.Module):

    """
    CNN-LSTM hybrid model for combined regression and classification tasks on spatio-temporal data.

    This model integrates a 2D CNN backbone for per-timestep feature extraction from spatial input 
    (e.g., weather or image data) and an LSTM module to capture temporal dependencies across the sequence.
    The architecture outputs:
    - A regression forecast over a specified forecast horizon.
    - A classification prediction (multi-class) over the same forecast horizon.

    Args:
        in_channels (int): Number of input channels per spatial input (e.g., number of weather variables).
        lstm_hidden_dim (int): Number of hidden units in the LSTM.
        lstm_layers (int): Number of stacked LSTM layers.
        forecast_horizon (int): Number of future time steps to predict.
        lagged_input_dim (int): Number of lagged input features per time step.
        H (int): Height of the input images/grids.
        W (int): Width of the input images/grids.
        cnn1_out_channels (int): Number of output channels of the first CNN layer.
        cnn1_kernel_size (int): Kernel size of the first CNN layer.
        cnn1_padding (int): Padding for the first CNN layer.
        cnn2_out_channels (int): Number of output channels of the second CNN layer.
        cnn2_kernel_size (int): Kernel size of the second CNN layer.
        cnn2_padding (int): Padding for the second CNN layer.
        cnn_linear_out_features (int): Number of features after the CNN flattening and dense layer.
        lstm_input_size (int): Input size fed into the LSTM (after CNN and lagged feature merging).
        dropout (float): Dropout rate used after the LSTM.
        n_classes (int): Number of output classes per forecasted time step.

    Inputs:
        weather_seq (torch.Tensor): Input sequence of shape (B, T, C, H, W), where 
                                    B=batch size, T=sequence length, C=channels, H=height, W=width.
        y_lagged (torch.Tensor): Lagged target input of shape (B, T, lagged_input_dim).

    Outputs:
        reg_out (torch.Tensor): Regression output of shape (B, forecast_horizon).
        class_out (torch.Tensor): Raw classification logits of shape (B, forecast_horizon, n_classes).

    Note:
        The predict() method:
        - Runs in eval mode without gradient computation.
        - Returns reg_out (clamped to [-1e6, 1e6]) and class_out as argmax class indices.

    """
    
    def __init__(self, 
                 in_channels=13,
                 lstm_hidden_dim=64, 
                 lstm_layers=2, 
                 forecast_horizon=168,
                 lagged_input_dim=7,
                 H=25,
                 W=25,
                 cnn1_out_channels=32,
                 cnn1_kernel_size=3,
                 cnn1_padding=1,
                 cnn2_out_channels=128,
                 cnn2_kernel_size=3,
                 cnn2_padding=1,

                 cnn_linear_out_features=128,
                 lstm_input_size=128,
                 dropout=0.5,
                 n_classes=3
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.forecast_horizon = forecast_horizon

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn1_out_channels, cnn1_kernel_size, padding=cnn1_padding),
            nn.ReLU(),
            nn.BatchNorm2d(cnn1_out_channels),

            nn.Conv2d(cnn1_out_channels, cnn2_out_channels, cnn2_kernel_size, padding=cnn2_padding),
            nn.ReLU(),
            nn.BatchNorm2d(cnn2_out_channels),

            nn.Flatten(),
            nn.Linear(cnn2_out_channels * H * W, cnn_linear_out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )


        self.lstm_input_linear = nn.Linear(cnn_linear_out_features + lagged_input_dim, lstm_input_size)
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=self.lstm_hidden_dim,
                            num_layers=self.lstm_layers,
                            batch_first=True)
        self.lstm_bn = nn.BatchNorm1d(self.lstm_hidden_dim)
        self.lstm_act = nn.LeakyReLU()
        self.lstm_dropout = nn.Dropout(dropout)
        #self.linear_hidden = nn.Linear(self.lstm_hidden_dim, self.forecast_horizon + self.forecast_horizon * n_classes)

        self.reg_out = nn.Linear(self.lstm_hidden_dim, self.forecast_horizon) # Regression output
        self.class_out = nn.Linear(self.lstm_hidden_dim, self.forecast_horizon * n_classes) # Classification output

        self.apply(self.init_weights)
        

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        

    def forward(self, weather_seq, y_lagged):
        """
        weather_seq: (B, T, C, H, W)
        y_lagged:    (B, T, lagged_dim)
        """
        B, T, C, H, W = weather_seq.shape

        x = []
        for t in range(T):
            feat_t = self.cnn(weather_seq[:, t])  # (B, C, H, W) → (B, cnn_feat)
            x.append(feat_t)
        x = torch.stack(x, dim=1)  # (B, T, cnn_feat)

        x = torch.cat([x, y_lagged], dim=-1)  # (B, T, D)
        
        x = self.lstm_input_linear(x)  # (B, T, lstm_input_size)

        x, _ = self.lstm(x)  # (B, T, lstm_hidden_dim)
        x = x[:, -1, :]  # (B, lstm_hidden_dim)

        x = self.lstm_bn(x)
        x = self.lstm_act(x)
        x = self.lstm_dropout(x)

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





    # def forward(self, weather_seq, y_lagged):
    #     """
    #     weather_seq: (B, T, C, H, W)
    #     y_lagged:    (B, T, lagged_dim)
    #     """
    #     B, T, C, H, W = weather_seq.shape

    #     cnn_features = []
    #     for t in range(T):
    #         x_t = weather_seq[:, t]  # (B, C, H, W)
    #         feat_t = self.cnn(x_t)   # (B, cnn_feat)
    #         cnn_features.append(feat_t)
    #     cnn_seq = torch.stack(cnn_features, dim=1)  # (B, T, cnn_feat)

    #     lstm_input = torch.cat([cnn_seq, y_lagged], dim=-1)  # (B, T, D)
    #     lstm_input = self.lstm_input_linear(lstm_input)  # (B, T, lstm_input_size)

    #     lstm_out, _ = self.lstm(lstm_input)  # (B, T, lstm_hidden_dim)
    #     final_hidden = lstm_out[:, -1, :]  # (B, lstm_hidden_dim)

    #     final_hidden = self.lstm_bn(final_hidden)
    #     final_hidden = self.lstm_act(final_hidden)
    #     final_hidden = self.lstm_dropout(final_hidden)

    #     reg_out = self.reg_out(final_hidden)  # (B, forecast_horizon)
    #     class_out = self.class_out(final_hidden).view(B, self.forecast_horizon, -1)  # (B, forecast_horizon, n_classes)

    #     return reg_out, class_out















import torch
import torch.nn as nn

class CNNLSTM_Regression(nn.Module):
    """
    CNN-LSTM hybrid model for multivariate time series regression.

    Combines:
    - CNN backbone for spatial feature extraction per time step (e.g., weather maps).
    - LSTM for temporal sequence modeling.
    - Fully connected layer for final forecast over specified horizon.

    Args:
        in_channels (int): Input channels per weather map.
        lstm_hidden_dim (int): Hidden units in LSTM.
        lstm_layers (int): Number of stacked LSTM layers.
        forecast_horizon (int): Steps to forecast.
        lagged_input_dim (int): Dimensionality of lagged target inputs.
        H, W (int): Height and width of input maps.
        cnn*_out_channels, cnn*_kernel_size, cnn*_padding (int): CNN layer configs.
        cnn_linear_out_features (int): Output features after CNN + linear.
        lstm_input_size (int): Input size for LSTM after concatenation.
        dropout (float): Dropout probability.
    """

    def __init__(self, 
                 in_channels=13,
                 lstm_hidden_dim=64, 
                 lstm_layers=2, 
                 forecast_horizon=168,
                 lagged_input_dim=7,
                 H=25,
                 W=25,
                 cnn1_out_channels=32,
                 cnn1_kernel_size=3,
                 cnn1_padding=1,
                 cnn2_out_channels=128,
                 cnn2_kernel_size=3,
                 cnn2_padding=1,

                 cnn_linear_out_features=128,
                 lstm_input_size=128,
                 dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.forecast_horizon = forecast_horizon
        self.lagged_input_dim = lagged_input_dim

        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn1_out_channels, cnn1_kernel_size, padding=cnn1_padding),
            nn.ReLU(),
            nn.BatchNorm2d(cnn1_out_channels),

            nn.Conv2d(cnn1_out_channels, cnn2_out_channels, cnn2_kernel_size, padding=cnn2_padding),
            nn.ReLU(),
            nn.BatchNorm2d(cnn2_out_channels),

            nn.Flatten(),
            nn.Linear(cnn2_out_channels * H * W, cnn_linear_out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM part
        self.lstm_input_linear = nn.Linear(cnn_linear_out_features + lagged_input_dim, lstm_input_size)
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True)

        # Post-LSTM layers
        self.lstm_bn = nn.BatchNorm1d(lstm_hidden_dim)
        self.lstm_act = nn.LeakyReLU()
        self.lstm_dropout = nn.Dropout(dropout)

        # Final regression head
        self.reg_out = nn.Linear(lstm_hidden_dim, forecast_horizon)

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, weather_seq, y_lagged):
        """
        Forward pass.

        Args:
            weather_seq (Tensor): (B, T, C, H, W)
            y_lagged (Tensor):    (B, T, lagged_input_dim)

        Returns:
            reg_out (Tensor): (B, forecast_horizon)
        """
        B, T, C, H, W = weather_seq.shape

        # Extract CNN features per time step
        x_cnn = [self.cnn(weather_seq[:, t]) for t in range(T)]  # list of (B, cnn_feat)
        x_cnn = torch.stack(x_cnn, dim=1)  # (B, T, cnn_feat)

        # Concatenate lagged inputs
        x = torch.cat([x_cnn, y_lagged], dim=-1)  # (B, T, cnn_feat + lagged_dim)

        # Project to LSTM input size
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(x.device)  # Initial cell state

        x = self.lstm_input_linear(x)  # (B, T, lstm_input_size)

        # Pass through LSTM
        x, _ = self.lstm(x, (h_0, c_0))  # (B, T, lstm_hidden_dim)
        x = x[:, -1, :]      # Take last time step → (B, lstm_hidden_dim)

        # Post-LSTM processing
        x = self.lstm_bn(x)
        x = self.lstm_act(x)
        x = self.lstm_dropout(x)

        # Final regression output
        reg_out = self.reg_out(x)  # (B, forecast_horizon)

        return reg_out

    def predict(self, weather_seq, y_lagged):
        """
        Predict without gradients.

        Args:
            weather_seq (Tensor): (B, T, C, H, W)
            y_lagged (Tensor):    (B, T, lagged_input_dim)

        Returns:
            reg_out (Tensor): (B, forecast_horizon)
        """
        self.eval()
        device = next(self.parameters()).device
        weather_seq, y_lagged = weather_seq.to(device), y_lagged.to(device)

        with torch.inference_mode():
            reg_out = self.forward(weather_seq, y_lagged)

            if torch.isnan(reg_out).any():
                print("⚠ Warning: NaNs detected in reg_out!")

            return reg_out





import torch
import torch.nn as nn

class CNNLSTM_Regression_V2(nn.Module):
    """
    CNN-LSTM hybrid model for multivariate time series regression.

    Combines:
    - CNN backbone for spatial feature extraction per time step (e.g., weather maps).
    - LSTM for temporal sequence modeling.
    - Fully connected layer for final forecast over specified horizon.

    Args:
        in_channels (int): Input channels per weather map.
        lstm_hidden_dim (int): Hidden units in LSTM.
        lstm_layers (int): Number of stacked LSTM layers.
        forecast_horizon (int): Steps to forecast.
        lagged_input_dim (int): Dimensionality of lagged target inputs.
        H, W (int): Height and width of input maps.
        cnn*_out_channels, cnn*_kernel_size, cnn*_padding (int): CNN layer configs.
        cnn_linear_out_features (int): Output features after CNN + linear.
        lstm_input_size (int): Input size for LSTM after concatenation.
        dropout (float): Dropout probability.
    """

    def __init__(self, 
                 in_channels=13,
                 lstm_hidden_dim=64, 
                 lstm_layers=2, 
                 forecast_horizon=168,
                 lagged_input_dim=7,
                 H=25,
                 W=25,
                 cnn1_out_channels=32,
                 cnn1_kernel_size=3,
                 cnn1_padding=1,
                 cnn2_out_channels=128,
                 cnn2_kernel_size=3,
                 cnn2_padding=1,

                 cnn_linear_out_features=128,
                 lstm_input_size=128,
                 dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.forecast_horizon = forecast_horizon
        self.lagged_input_dim = lagged_input_dim

        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn1_out_channels, cnn1_kernel_size, padding=cnn1_padding),
            nn.BatchNorm2d(cnn1_out_channels),
            nn.ReLU(),
            

            nn.Conv2d(cnn1_out_channels, cnn2_out_channels, cnn2_kernel_size, padding=cnn2_padding),
            nn.BatchNorm2d(cnn2_out_channels),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1),   # (B, C, 1, 1)
            nn.Flatten(),
            nn.Linear(cnn2_out_channels, cnn_linear_out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM part
        self.lstm_input_linear = nn.Linear(cnn_linear_out_features + lagged_input_dim, lstm_input_size)
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)

        # Post-LSTM layers
        self.lstm_bn = nn.BatchNorm1d(lstm_hidden_dim)
        self.lstm_act = nn.LeakyReLU()
        self.lstm_dropout = nn.Dropout(dropout)

        # Final regression head
        self.reg_out = nn.Linear(lstm_hidden_dim, forecast_horizon)

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, weather_seq, y_lagged):
        """
        Forward pass.

        Args:
            weather_seq (Tensor): (B, T, C, H, W)
            y_lagged (Tensor):    (B, T, lagged_input_dim)

        Returns:
            reg_out (Tensor): (B, forecast_horizon)
        """
        B, T, C, H, W = weather_seq.shape

        # Extract CNN features per time step
        x_cnn = [self.cnn(weather_seq[:, t]) for t in range(T)]  # list of (B, cnn_feat)
        x_cnn = torch.stack(x_cnn, dim=1)  # (B, T, cnn_feat)

        # Concatenate lagged inputs
        x = torch.cat([x_cnn, y_lagged], dim=-1)  # (B, T, cnn_feat + lagged_dim)

        # Project to LSTM input size
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(x.device)  # Initial cell state

        x = self.lstm_input_linear(x)  # (B, T, lstm_input_size)

        # Pass through LSTM
        x, _ = self.lstm(x, (h_0, c_0))  # (B, T, lstm_hidden_dim)
        x = x[:, -1, :]      # Take last time step → (B, lstm_hidden_dim)

        # Post-LSTM processing
        x = self.lstm_bn(x)
        x = self.lstm_act(x)
        x = self.lstm_dropout(x)

        # Final regression output
        reg_out = self.reg_out(x)  # (B, forecast_horizon)

        return reg_out

    def predict(self, weather_seq, y_lagged):
        """
        Predict without gradients.

        Args:
            weather_seq (Tensor): (B, T, C, H, W)
            y_lagged (Tensor):    (B, T, lagged_input_dim)

        Returns:
            reg_out (Tensor): (B, forecast_horizon)
        """
        self.eval()
        device = next(self.parameters()).device
        weather_seq, y_lagged = weather_seq.to(device), y_lagged.to(device)

        with torch.inference_mode():
            reg_out = self.forward(weather_seq, y_lagged)

            if torch.isnan(reg_out).any():
                print("⚠ Warning: NaNs detected in reg_out!")

            return reg_out