
import copy
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, recall_score
from utils.recall import RecallLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


# y_reg_batch = tensor([
#     [0.9, 1.1, 1.3],   # → Klassen [0, 1, 2]
#     [1.4, 1.6, 2.1],   # → Klassen [2, 3, 4]
#     [0.8, 1.0, 1.2]    # → Klassen [0, 1, 1]
# ])
# 
# 
# bins = [1.0, 1.25, 1.5, 2.0]
# → ergibt Klassen:
# < 1.00       → Klasse 0
# 1.00–1.24    → Klasse 1
# 1.25–1.49    → Klasse 2
# 1.50–1.99    → Klasse 3
# ≥ 2.00       → Klasse 4
# 


def training(model, X_train, y_train, X_val, y_val, optimizer, epochs, writer, batch_size=64, patience=50, log_tensorboard=True, log_dropout=True, verbose=True):

    # Set device to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Move data and model to device
    X_train, y_train, X_val, y_val = X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device)
    model.to(device)

    # Initialize best metrics and model
    best_val_metric = float('inf') # Initialize the best_eval_metric
    best_val_loss = float('inf') # Initialize the best_val_loss
    best_model = copy.deepcopy(model.state_dict()) # Initialize the best_model
    patience_counter_loss = 0  # Early stopping counter
    patience_counter_metric = 0  # Early stopping counter

    # DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


    # Scheduler for learning rate adjustment (if val_loss does not improve for 'patience' epochs)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)


    mse_loss = nn.MSELoss()
    weights = torch.tensor([1.0, 20.0, 30.0, 40.0, 50.0], device=device)  # Example weights for classes
    ce_loss = nn.CrossEntropyLoss(weight=weights)  # Example weights for classes

    bins = torch.tensor([1.0, 1.25, 1.5, 2.0], device=device)
    n_classes = len(bins) + 1  # Number of classes based on bins
    alpha = 0.7  # Weight for classification loss



    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop
        for X_batch_train, y_batch_train in train_loader:
            X_batch_train, y_batch_train = X_batch_train.to(device), y_batch_train.to(device)

            optimizer.zero_grad()  # Clear gradients
            reg_out_train, class_out_train = model(X_batch_train)  # Forward pass
            y_class_batch = torch.bucketize(y_batch_train, bins)  # (batch_size, output_horizon)


            # Output: class_out_train: (batch_size, output_horizon, n_classes)
            # Target: y_class_batch:  (batch_size, output_horizon)
            # print("class_out_train shape:", class_out_train.shape)
            # print("y_class_batch shape:", y_class_batch.shape)
            # print("class_out_train.view(-1, n_classes):", class_out_train.view(-1, n_classes).shape)
            # print("y_class_batch.view(-1):", y_class_batch.view(-1).shape)
            class_loss_train = ce_loss(
                class_out_train.view(-1, n_classes),         # (batch_size * horizon, n_classes)
                y_class_batch.view(-1)                       # (batch_size * horizon)
            )

            
            reg_loss_train = mse_loss(reg_out_train, y_batch_train)

            loss = alpha * class_loss_train + (1 - alpha) * reg_loss_train


            #loss = criterion(y_pred_train, y_batch_train)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            epoch_loss += loss.item()  # Accumulate batch loss

        # Validation
        model.eval()
        with torch.no_grad():
            reg_out_val, class_out_val = model(X_val) # Forward pass
            y_class_val = torch.bucketize(y_val, bins)  # (batch_size, output_horizon)

            class_loss_val = ce_loss(
                class_out_val.view(-1, n_classes),         # (batch_size * horizon, n_classes)
                y_class_val.view(-1)                       # (batch_size * horizon)
            )

            reg_loss_val = mse_loss(reg_out_val, y_val)

            loss_val = alpha * class_loss_val + (1 - alpha) * reg_loss_val
            scheduler.step(loss_val)  # Adjust learning rate based on loss
            
            # calculate the training accuracy
            predictions_train, _ = model.predict(X_train)
            mse_train = mean_squared_error(y_train.cpu(), predictions_train.cpu())

            # Calculate validation accuracy
            predictions_val, _ = model.predict(X_val)
            mse_val = mean_squared_error(y_val.cpu(), predictions_val.cpu())



            # Log to TensorBoard
            if log_tensorboard:
                if writer is None:
                    continue
                writer.add_scalars("Loss", {"train": epoch_loss / len(train_loader), 'val': loss_val}, epoch)
                writer.add_scalars("MSE", {"train": mse_train, "val": mse_val}, epoch)
                
                writer.flush()

            # Early stopping if loss_val is increasing
            if loss_val < best_val_loss:
                best_val_loss = loss_val  # Update best val_loss
                patience_counter_loss = 0  # Reset patience counter
            else:
                patience_counter_loss += 1  # Increment if no improvement

            # Early Stopping based on if val_mse in not increasing
            if mse_val < best_val_metric:
                best_val_metric = mse_val
                best_model = copy.deepcopy(model.state_dict()) # saves the best model where the mse_val is lowest
                patience_counter_metric = 0  # Reset patience counter if improved
            else:
                patience_counter_metric += 1

            # Early stopping check
            if (patience_counter_loss >= patience) or (patience_counter_metric >= patience):
                print(f"Early stopping at epoch {epoch+1}")
                break


        # Print status
        if verbose:
          if epoch % 1 == 0:
            print(f"| Epoch {epoch+1} | Train Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {loss_val:.4f} | Train MSE: {mse_train:.4f}, Val MSE: {mse_val:.4f} |")


    # Load the best model
    model.load_state_dict(best_model)
    print(f"Best validation MSE: {best_val_metric:.4f}")

    if writer is not None:
        writer.close()
        
    return model







def training_ConvLSTM(model, X_train, y_train, X_val, y_val, y_lagged_train, y_lagged_val, optimizer, epochs, writer, batch_size=64, patience=50, log_tensorboard=True, verbose=True):

    # Set device to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Move data and model to device
    X_train, y_train, X_val, y_val, y_lagged_train, y_lagged_val = X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device), y_lagged_train.to(device), y_lagged_val.to(device)
    model.to(device)

    # Initialize best metrics and model
    best_val_metric = float('inf') # Initialize the best_eval_metric
    best_val_loss = float('inf') # Initialize the best_val_loss
    best_model = copy.deepcopy(model.state_dict()) # Initialize the best_model
    patience_counter_loss = 0  # Early stopping counter
    patience_counter_metric = 0  # Early stopping counter

    # DataLoader for batching
    train_dataset = TensorDataset(X_train, y_lagged_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



    # Scheduler for learning rate adjustment (if val_loss does not improve for 'patience' epochs)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)


    mse_loss = nn.MSELoss()

    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop
        for X_batch_train, y_lagged_batch, y_batch_train in train_loader:
            X_batch_train, y_lagged_batch, y_batch_train = X_batch_train.to(device), y_lagged_batch.to(device), y_batch_train.to(device)

            optimizer.zero_grad()  # Clear gradients
            reg_out_train = model(X_batch_train, y_lagged_batch)  # Forward pass

            
            loss = mse_loss(reg_out_train, y_batch_train)


            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            epoch_loss += loss.item()  # Accumulate batch loss

        # Validation
        model.eval()
        with torch.no_grad():

            reg_out_val = model(X_val, y_lagged_val) # Forward pass
            loss_val = mse_loss(reg_out_val, y_val)


            scheduler.step(loss_val)  # Adjust learning rate based on loss
            
            # calculate the training mse
            predictions_train = model.predict(X_train, y_lagged_train)
            mse_train = mean_squared_error(y_train.cpu(), predictions_train.cpu())

            # Calculate validation mse
            predictions_val = model.predict(X_val, y_lagged_val)
            mse_val = mean_squared_error(y_val.cpu(), predictions_val.cpu())



            # Log to TensorBoard
            if log_tensorboard:
                if writer is None:
                    continue
                writer.add_scalars("Loss", {"train": epoch_loss / len(train_loader), 'val': loss_val}, epoch)
                writer.add_scalars("MSE", {"train": mse_train, "val": mse_val}, epoch)
                #writer.add_scalar("Accuracy/val", accuracy_val, epoch)
                writer.flush()

            # Early stopping if loss_val is increasing
            if loss_val < best_val_loss:
                best_val_loss = loss_val  # Update best val_loss
                patience_counter_loss = 0  # Reset patience counter
            else:
                patience_counter_loss += 1  # Increment if no improvement

            # Early Stopping based on if val_mse in not increasing
            if mse_val < best_val_metric:
                best_val_metric = mse_val
                best_model = copy.deepcopy(model.state_dict()) # saves the best model where the mse_val is lowest
                patience_counter_metric = 0  # Reset patience counter if improved
            else:
                patience_counter_metric += 1

            # Early stopping check
            if (patience_counter_loss >= patience) or (patience_counter_metric >= patience):
                print(f"Early stopping at epoch {epoch+1}")
                break


        # Print status
        if verbose:
          if epoch % 1 == 0:
            print(f"| Epoch {epoch+1} | Train Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {loss_val:.4f} | Train MSE: {mse_train:.4f}, Val MSE: {mse_val:.4f} |")


    # Load the best model
    model.load_state_dict(best_model)
    print(f"Best validation MSE: {best_val_metric:.4f}")

    if writer is not None:
        writer.close()
        
    return model









def log_tensorboard_metrics(writer, trial_id, epoch, train_loss, val_loss, mse_train, mse_val, recall_train, recall_val, score_train, score_val, lr, time_per_epoch, patience_loss, patience_metric, patience_score):
    writer.add_scalars(f"{trial_id}/Loss", {"Train": train_loss, "Val": val_loss}, epoch)
    writer.add_scalars(f"{trial_id}/MSE", {"Train": mse_train, "Val": mse_val}, epoch)
    writer.add_scalars(f"{trial_id}/Recall", {"Train": recall_train, "Val": recall_val}, epoch)
    writer.add_scalars(f"{trial_id}/Score", {"Train": score_train, "Val": score_val}, epoch)
    writer.add_scalar(f"{trial_id}/Learning Rate", lr, epoch)
    writer.add_scalar(f"{trial_id}/Time Per Epoch", time_per_epoch, epoch)
    writer.add_scalar(f"{trial_id}/Patience Counter Loss", patience_loss, epoch)
    writer.add_scalar(f"{trial_id}/Patience Counter Metric", patience_metric, epoch)
    writer.add_scalar(f"{trial_id}/Patience Counter Score", patience_score, epoch)
    writer.flush()

def compute_combined_loss(mse_loss_fn, class_loss_fn, reg_out, class_out, y_true, y_class, alpha, n_classes):
    loss_mse = mse_loss_fn(reg_out, y_true.to(reg_out.dtype))
    loss_class = class_loss_fn(class_out.view(-1, n_classes), y_class.view(-1))
    return alpha * loss_class + (1 - alpha) * loss_mse, loss_mse, loss_class

def custom_score(mse: float, recall: float, alpha: float) -> float:
    return alpha * (1 - recall) + (1 - alpha) * mse

def training_ConvLSTM_Regression_Classification(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    y_lagged_train: torch.Tensor,
    y_lagged_val: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    writer: Optional[Any],
    batch_size: int = 64,
    patience: int = 50,
    log_tensorboard: bool = True,
    verbose: bool = True,
    trial: Optional[Any] = None,
    alpha: float = 0.7,
    use_amp: bool = False,
    classification_loss: bool = True,
    reduce_lr_patience: int = 25,
) -> nn.Module:
    
    # Überprüfen auf NaN-Werte in den Eingabedaten
    if torch.isnan(X_train).any() or torch.isnan(y_lagged_train).any():
        raise ValueError("Eingabedaten enthalten NaN-Werte.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trial_id = f"trial_{trial.number}" if trial else time.strftime("%Y-%m-%d %H:%M:%S")
    bins = torch.tensor([1.0, 2.0], device=device)
    weights = torch.tensor([1.0, 1.0, 1.0], device=device)
    n_classes = len(bins) + 1

    mse_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss(weight=weights)
    recall_loss_fn = RecallLoss(weight=weights)

    train_dataset = TensorDataset(X_train, y_lagged_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True) # shuffle=False for autokorrelated time series data
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)  # For evaluation during training

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=reduce_lr_patience, factor=0.5)
    scaler = torch.amp.GradScaler(enabled=use_amp)  # GradScaler hinzufügen

    best_val_loss = float('inf')
    best_score_val_metric = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    patience_counter_loss = 0
    patience_counter_metric = 0
    patience_counter_score = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for X_batch, y_lagged_batch, y_batch in train_loader:
            X_batch, y_lagged_batch, y_batch = X_batch.to(device), y_lagged_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            # Mixed Precision Training mit autocast
            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                if classification_loss:
                    reg_out, class_out = model(X_batch, y_lagged_batch)
                    y_class = torch.bucketize(y_batch, bins.to(device))
                else:
                    reg_out = model(X_batch, y_lagged_batch)
                    class_out = None
                    y_class = None

            if classification_loss:
                loss, _, _ = compute_combined_loss(mse_loss_fn, recall_loss_fn, reg_out, class_out, y_batch, y_class, alpha, n_classes)
            else:
                loss = mse_loss_fn(reg_out, y_batch.to(reg_out.dtype))
            
            # GradScaler verwenden
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            if classification_loss:
                reg_out_val, class_out_val = model(X_val.to(device), y_lagged_val.to(device))
                
                y_class_val = torch.bucketize(y_val.to(device), bins.to(device))

                val_loss, loss_mse_val, _ = compute_combined_loss(mse_loss_fn, recall_loss_fn, reg_out_val, class_out_val, y_val.to(device), y_class_val, alpha, n_classes)
            else:
                reg_out_val = model(X_val.to(device), y_lagged_val.to(device))
                val_loss = mse_loss_fn(reg_out_val, y_val.to(device))
                y_class_val = torch.bucketize(y_val.to(device), bins.to(device))

            scheduler.step(val_loss.item())

            if classification_loss:
                predictions_train = []
                pre_class_train = []
                for X_batch_train, y_lagged_batch_train, _ in train_eval_loader:
                    preds, pre_class = model.predict(X_batch_train.to(device), y_lagged_batch_train.to(device))
                    predictions_train.append(preds)
                    pre_class_train.append(pre_class)
                predictions_train = torch.cat(predictions_train, dim=0)
                pre_class_train = torch.cat(pre_class_train, dim=0)
                predictions_val, pre_class_val = model.predict(X_val.to(device), y_lagged_val.to(device))
            else:
                predictions_train = []
                for X_batch_train, y_lagged_batch_train, _ in train_eval_loader:
                    preds = model.predict(X_batch_train.to(device), y_lagged_batch_train.to(device))
                    predictions_train.append(preds)
                predictions_train = torch.cat(predictions_train, dim=0)
                predictions_val = model.predict(X_val.to(device), y_lagged_val.to(device))
                pre_class_train = torch.bucketize(predictions_train, bins.to(device))
                pre_class_val = torch.bucketize(predictions_val, bins.to(device))

            
            # check if y_train and y_val, predictions_train and predictions_val contain NaN values
            # Überprüfung auf NaN-Werte in Vorhersagen und Zielwerten
            if torch.isnan(predictions_train).any() or torch.isnan(y_train).any():
                print(f"NaN-Werte in den Trainingsdaten oder Vorhersagen gefunden für {trial_id}.")
                print(f"y_train shape: {y_train.shape}, NaN-Werte: {torch.isnan(y_train).sum()}")
                print(f"predictions_train shape: {predictions_train.shape}, NaN-Werte: {torch.isnan(predictions_train).sum()}")
                continue  # Überspringe diesen Fold

            if torch.isnan(predictions_val).any() or torch.isnan(y_val).any():
                print(f"NaN-Werte in den Validierungsdaten oder Vorhersagen gefunden für {trial_id}.")
                print(f"y_val shape: {y_val.shape}, NaN-Werte: {torch.isnan(y_val).sum()}")
                print(f"predictions_val shape: {predictions_val.shape}, NaN-Werte: {torch.isnan(predictions_val).sum()}")
                continue  # Überspringe diesen Fold

                
            mse_train = mean_squared_error(y_train.cpu(), predictions_train.cpu())
            mse_val = mean_squared_error(y_val.cpu(), predictions_val.cpu())

            y_class_train = torch.bucketize(y_train.cpu(), bins.cpu())
            
            recall_train = recall_score(y_class_train.view(-1).cpu(), pre_class_train.view(-1).cpu(), average='macro')
            recall_val = recall_score(y_class_val.view(-1).cpu(), pre_class_val.view(-1).cpu(), average='macro')

            score_train = custom_score(mse_train, recall_train, alpha)
            score_val = custom_score(mse_val, recall_val, alpha)

            lr = optimizer.param_groups[0]['lr']
            time_per_epoch = time.time() - start_time

            if log_tensorboard and writer:
                if epoch == 0:
                    writer.add_graph(model, (X_train[:1].to(device), y_lagged_train[:1].to(device)))
                log_tensorboard_metrics(writer, trial_id, epoch, epoch_loss / len(train_loader), val_loss.item(), mse_train, mse_val, recall_train, recall_val, score_train, score_val, lr, time_per_epoch, patience_counter_loss, patience_counter_metric, patience_counter_score)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter_loss = 0
            else:
                patience_counter_loss += 1


            if classification_loss:
                if score_val < best_score_val_metric:
                    best_score_val_metric = score_val
                    best_model = copy.deepcopy(model.state_dict())
                    patience_counter_score = 0
                else:
                    patience_counter_score += 1
            else:
                if mse_val < best_score_val_metric:
                    best_score_val_metric = mse_val
                    best_model = copy.deepcopy(model.state_dict())
                    patience_counter_score = 0
                else:
                    patience_counter_score += 1

            if patience_counter_loss >= patience or patience_counter_score >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if verbose:
            print(f"| Epoch {epoch+1:3} | Train Loss: {epoch_loss / len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f} | MSE Train: {mse_train:.4f} | MSE Val: {mse_val:.4f} | "
                  f"Time: {time_per_epoch:.2f}s | Score Train: {score_train:.4f} | Score Val: {score_val:.4f} | ")

    model.load_state_dict(best_model)
    if writer:
        writer.close()

    print(f"Best validation MSE: {best_score_val_metric:.4f}")
    # empty the cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model














