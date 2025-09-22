import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import copy
from sklearn.metrics import mean_absolute_percentage_error

# Count the number of trainable parameters of model
def count_params(model):
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return 0
    

# Add evaluation measures to the dataframe result_df
# NOTE: At this stage, we are working with a differenced time series, thus metrics like mape make no sense!
def add_result(result_df, predictions, model, model_name, criterion, scaler, testY_pred, testY):
    # compute the loss using the given criterion
    loss=criterion(testY_pred, testY)
    result_df.at[model_name, "Test-Loss"] = loss.item()

    # compute loss as the difference between predicted and actual number of passengers 
    testY_pred_passengers=scaler.inverse_transform(testY_pred.reshape(-1,1))
    testY_passengers=scaler.inverse_transform(testY.reshape(-1,1))
    
    abs_passenger_diff=np.sum(abs(testY_pred_passengers-testY_passengers))
    result_df.at[model_name, "Avg. monthly passenger deviation"] = round(abs_passenger_diff/len(testY_passengers),1)
    result_df.at[model_name, "# Parameters"] = int(count_params(model))

    prediction_df=pd.DataFrame(testY_pred_passengers, columns=[model_name])
    if (predictions.empty):
        predictions.reindex_like(prediction_df)
        predictions["actual"] = pd.DataFrame(testY_passengers, columns=["actual"])["actual"]
    predictions[model_name]=prediction_df[model_name]




# Train a model using training data
# Evaluate on validation data
# store a checkpoint for the configuration that performed best on validation 
def train(model, num_epochs, learning_rate, loss_function, trainX, trainY, valX, valY):
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss=float("inf")
    best_model_dict=model.state_dict()
    best_epoch=0
    
    model.train()
    
    for epoch in range(num_epochs):
        trainY_pred = model(trainX)  # predict train with the current model
        optimizer.zero_grad()
    
        train_loss = loss_function(trainY_pred, trainY) # compute the loss ("how bad is our model?")
    
        train_loss.backward() # propagate the loss backwards through the network
    
        optimizer.step() # update weights and biases
    
        with torch.no_grad():
            valY_pred=model(valX)
            val_loss=loss_function(valY_pred,valY)
            if (val_loss<best_val_loss):
                best_val_loss=val_loss
                best_model_dict=copy.deepcopy(model.state_dict())
                best_epoch=epoch
            
        if epoch % 1000 == 999:
            print("Epoch: %d, loss: %1.5f, val_loss: %1.5f" % (epoch, train_loss.item(), val_loss.item()))
    model.load_state_dict(best_model_dict)
    print("Best Epoch: %d, loss: %1.5f" % (best_epoch, best_val_loss.item()))
    return model


# Use the model to predict for the values in the test set
# Return the prediction
def predict(model, testX):
    model.eval()
    with torch.no_grad():
        return model(testX)
    


# Train a model using training data with two inputs
# Evaluate on validation data
# store a checkpoint for the configuration that performed best on validation 
def hyb_train(model, num_epochs, learning_rate, loss_function, trainX1, trainX2, trainY, valX1, valX2, valY):
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss=float("inf")
    best_model_dict=model.state_dict()
    best_epoch=0
    
    model.train()
    
    for epoch in range(num_epochs):
        trainY_pred = model(trainX1, trainX2)   # <- two arguments here
        optimizer.zero_grad()
    
        train_loss = loss_function(trainY_pred, trainY)
    
        train_loss.backward()
    
        optimizer.step()
    
        with torch.no_grad():
            valY_pred=model(valX1, valX2)   # <- two arguments here
            val_loss=loss_function(valY_pred,valY)
            if (val_loss<best_val_loss):
                best_val_loss=val_loss
                best_model_dict=copy.deepcopy(model.state_dict)
                best_epoch=epoch
            
        if epoch % 1000 == 999:
            print("Epoch: %d, loss: %1.5f, val_loss: %1.5f" % (epoch, train_loss.item(), val_loss.item()))
    model.load_state_dict(best_model_dict())
    print("Best Epoch: %d, loss: %1.5f" % (best_epoch, best_val_loss.item()))
    return model

            
# Use the model to predict for the values in the test set using two inputs
# Return the prediction
def hyb_predict(model, testX1, testX2): # <- two arguments here
    model.eval()
    with torch.no_grad():
        return model(testX1, testX2)   # <- two arguments here


# create sliding windows with according target variables
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


# create the machine learning experimental setup (scale data, create sliding windows, split into train-, validation-, and test-data
def setup_experiment(dataset, seq_len, test_share, val_share, scaler, ignored_last_month):
    scaled_data = scaler.fit_transform(dataset)
    x, y = sliding_windows(scaled_data, seq_len)
    used_vec_length=len(y) -ignored_last_month # remove the last ignored_last_month month
    test_size = int(used_vec_length * test_share)
    val_size = int(used_vec_length * val_share)
    train_size=used_vec_length-test_size-val_size
    index_val_start=train_size
    index_test_start=train_size+val_size
    x_train=x[0:index_val_start]
    y_train=y[0:index_val_start]
    x_val =x[index_val_start:index_test_start]
    y_val =y[index_val_start:index_test_start]
    x_test =x[index_test_start:used_vec_length]
    y_test =y[index_test_start:used_vec_length]
    
    return x_train, y_train, x_val, y_val, x_test, y_test




#
# Baselines
#

# Predict the average of the most recent n elements of x
def baseline_avg_prev_n_month(x, n):
    return torch.Tensor(np.sum(x[:,-n:], axis=1)/n)

# Predict the value of last year (12 month ago)
def baseline_last_year(x):
    return torch.Tensor(x[:,-12])

# Predict 0 as new value
def baseline_zero(x, scaler):
    return torch.Tensor(np.full((len(x),1), scaler.transform(np.array(0).reshape(-1,1))))

