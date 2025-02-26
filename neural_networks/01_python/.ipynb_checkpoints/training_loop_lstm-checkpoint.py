import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_size = 64  
output_size = 2  
num_layers = 2
dropout = 0.2
criterion = nn.CrossEntropyLoss()

class StockPriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(StockPriceLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)
    
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size dynamically

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)  # (num_layers, batch_size, hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)  # (num_layers, batch_size, hidden_dim)
        
        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :]) 
        out = self.sigmoid(out)
        return out
    
def reshape_remove_characters(df):

    X = np.array([np.stack(row) for row in df.drop(columns=['Target']).values])
    y = df['Target'].values

    smote = SMOTE(random_state=42)
    n_samples, timesteps, n_features = X.shape
    X_flat = X.reshape((n_samples, timesteps * n_features))
    X_flat = np.where(X_flat == 'ç', 0, X_flat)

    X_resampled = X_flat.reshape((-1, timesteps, n_features))
    
    return X_resampled, y

def evaluate_rolling_unchanged_model_threshold(
    model, 
    X, 
    y, 
    criterion, 
    optimizer, 
    device, 
    train_size, 
    batch_size, 
    num_epochs, 
    lower_threshold
):
    """
    Evaluate a PyTorch model using a rolling prediction approach for time series,
    training the model only once on the initial training set. For each time step
    after train_size, the model makes a prediction without further parameter updates.
    Only predicts +1 or -1 if the probability of class 1 is above/below given thresholds;
    otherwise, predicts 0. Accuracy is computed only on nonzero predictions.

    Args:
        model:          PyTorch model to evaluate.
        X:              Feature data (numpy array).
        y:              Target data (numpy array).
        criterion:      Loss function (e.g., CrossEntropyLoss).
        optimizer:      Optimizer (e.g., Adam).
        device:         Device for computation (CPU or GPU).
        train_size:     Initial size of the training data (int or float).
                        If < 1, treated as fraction of total length.
        batch_size:     Batch size for training.
        num_epochs:     Number of epochs for initial training only.
        lower_threshold: Probability threshold below which model predicts -1.
        upper_threshold: Probability threshold above which model predicts +1.

    Returns:
        dict: Dictionary with the following keys:
            - "rolling_predictions": All predictions (-1, 0, +1) across the test period.
            - "rolling_targets": Corresponding true targets in [-1, +1].
            - "filtered_predictions": Nonzero predictions only.
            - "filtered_targets": Targets corresponding to nonzero predictions.
            - "accuracy_nonzero": Accuracy computed only on nonzero predictions.
    """

    # Convert X, y to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Determine initial training set size
    if train_size < 1.0:
        lower_bound = int(train_size * len(X))
    else:
        lower_bound = train_size

    # -------------------------
    # 1) SINGLE TRAINING PHASE
    # -------------------------
    model.to(device)
    model.train()
    
    X_train = X[:lower_bound].to(device)
    y_train = y[:lower_bound].to(device)

    train_dataset = TensorDataset(X_train, y_train)
    trainloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,         # Keep False if order matters; True for better generalization
        # num_workers=4,         # Adjust based on your CPU cores
        # pin_memory=True,       # Speeds up transfer if using GPUs
        drop_last=False        # Ensure the last batch is included
    )

    epoch_train_losses = []
    for epoch in range(num_epochs):
        # torch.cuda.empty_cache()
        epoch_loss = 0.0
        for X_batch, y_batch in trainloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred_y = model(X_batch)   # [batch_size, num_classes]
            loss = criterion(pred_y, y_batch)
            loss.backward()

            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
               
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"[Train] Epoch {epoch+1}/{num_epochs}, Loss={epoch_loss/len(trainloader):.4f}")

        epoch_train_losses.append(epoch_loss/len(trainloader))
        
    loss_decrease_percentage = ((epoch_train_losses[-1] - epoch_train_losses[0]) / epoch_train_losses[0]) * 100
    # ---------------------------------
    # 2) ROLLING PREDICTIONS, NO UPDATE
    # ---------------------------------
    model.eval()

    rolling_predictions = []
    rolling_targets     = []

    for i in range(lower_bound, len(X)):
        # Single-step "test" sample
        X_test = X[i:i+1].to(device)  # shape: (1, num_features)
        y_test = y[i:i+1].to(device)  # shape: (1, )

        with torch.no_grad():
            # Forward pass
            pred_y = model(X_test)  # [1, num_classes]
            probabilities = torch.softmax(pred_y, dim=1).cpu().numpy()  # shape: (1, 2)
            prob_class_1  = probabilities[:, 1]  # shape: (1,)

            # Threshold-based logic
            # Initialize all predictions to 0
            pred_classes = np.zeros_like(prob_class_1)
            # Predict -1 if prob < lower_threshold
            pred_classes[prob_class_1 < lower_threshold] = -1
            # Predict +1 if prob > upper_threshold
            pred_classes[prob_class_1 > 1-lower_threshold] = 1

        rolling_predictions.append(pred_classes[0])  # scalar
        rolling_targets.append(y_test.item())

    rolling_predictions = np.array(rolling_predictions)
    rolling_targets = np.array(rolling_targets).astype(int)

    # Convert any 0-labeled targets to -1 if your original data is in [-1, +1]
    # (Sometimes y might be {0,1} or {-1, +1}; adapt as needed.)
    rolling_targets[rolling_targets == 0] = -1

    # Filter out zero predictions
    nonzero_mask = rolling_predictions != 0
    filtered_preds = rolling_predictions[nonzero_mask]
    filtered_targets = rolling_targets[nonzero_mask]

    if len(filtered_preds) == 0:
        accuracy_nonzero = None
        print("No nonzero predictions, cannot compute thresholded accuracy.")
    else:
        accuracy_nonzero = accuracy_score(filtered_targets, filtered_preds)
        print(f"Accuracy on Nonzero Predictions: {accuracy_nonzero:.4f}")

    return {
        "rolling_predictions": rolling_predictions,
        "rolling_targets": rolling_targets,
        "filtered_predictions": filtered_preds,
        "filtered_targets": filtered_targets,
        "accuracy_nonzero": accuracy_nonzero,
        "loss_decrease_percentage": loss_decrease_percentage
    }

def run_entire_loop_lstm(stocks, model_type, processing, types_securities, years, window_sizes,
                    train_sizes, learning_rates, num_epochs_list, batch_sizes, prediction_thresholds, 
                   project_dir):
    # folder
    
    clean_data_dir = os.path.join(project_dir, "00_data/clean")
    horizontal_data_dir = os.path.join(project_dir, "00_data/horizontal_structure")
    results_dir = os.path.join(project_dir, "02_results")
    pca_data_dir = os.path.join(project_dir, "00_data/pca")

    results_list = []
    for stock in stocks:
        for security_type in types_securities:
            output_folder = os.path.join(results_dir, f"{model_type}/{stock}/{security_type}") 
            os.makedirs(output_folder, exist_ok=True)
            # files
            for period in years:
                # load original data as well (for info purposes)
                filename = f"{security_type}/{stock}/{period}_data.csv"
                original_input_filepath = os.path.join(clean_data_dir, filename)
                original_data = pd.read_csv(original_input_filepath)
                start_date = original_data.loc[0, "Date"]
                end_date = original_data.iloc[-1]["Date"]

                for possible_train_size in train_sizes:

                    results_csv_path = os.path.join(output_folder, f"{period}_{possible_train_size}.csv")

                    # columns, same file
                    for window_size in window_sizes:
                        print(f"{stock}, {security_type}, {period}, {possible_train_size}, {window_size}")

                        # load data
                        pkl_filename = f"clean/{security_type}/{stock}/{period}_{window_size}_data.pkl"
                        input_filepath = os.path.join(horizontal_data_dir, pkl_filename)
                        print(input_filepath)
                        input_df = pd.read_pickle(input_filepath)

                        X_resampled, y_resampled = reshape_remove_characters(input_df)

                        input_size = X_resampled.shape[2]
                        train_size = int(X_resampled.shape[0]*possible_train_size/100)
                        test_size = X_resampled.shape[0] - train_size

                        # generate model
                        model = StockPriceLSTM(input_size, hidden_size, output_size)
                        model = torch.nn.DataParallel(model)
                        model = model.to(device)

                        for learning_rate in learning_rates:

                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                            for num_epochs in num_epochs_list:
                                for prediction_threshold in prediction_thresholds:
                                    for batch_size in batch_sizes:

                                        print(f"Training {stock} | LR: {learning_rate} | Epochs: {num_epochs} | Batch: {batch_size} | Prediction Threshold: {prediction_threshold}")

                                        start_time = time.time()

                                        result = evaluate_rolling_unchanged_model_threshold(
                                            model, X_resampled, y_resampled, criterion, 
                                            optimizer, device, train_size, batch_size, num_epochs, 
                                            lower_threshold=prediction_threshold
                                        )     

                                        rolling_predictions = result["rolling_predictions"]
                                        rolling_targets = result["rolling_targets"]
                                        test_accuracy = result["accuracy_nonzero"]
                                        loss_decrease_percentage = result["loss_decrease_percentage"]
                                        nonzero_preds = np.count_nonzero(result["rolling_predictions"])

                                        end_time = time.time()    
                                        execution_time = end_time - start_time

                                        # --------------------------------------------
                                        # 1) Create a record (dictionary) for this run
                                        # --------------------------------------------
                                        run_record = {
                                            "start_date": start_date,
                                            "end_date": end_date,
                                            "execution_time": execution_time,
                                            "test_size": test_size,
                                            "nonzero_preds": nonzero_preds,
                                            "accuracy": test_accuracy,
                                            "prediction_threshold": prediction_threshold,

                                            "window_size": window_size,
                                            "learning_rate": learning_rate,
                                            "num_epochs": num_epochs,
                                            "train_loss_change_pctg": loss_decrease_percentage,

                                            "batch_size": batch_size,

                                            "output_size": output_size,
                                            "hidden_size": hidden_size,
                                            "num_layers": num_layers,
                                            "dropout_rate": dropout,
                                            "optimizer": optimizer.__class__.__name__,
                                            "criterion": criterion
                                        }

                                        # --------------------------------------------
                                        # 2) Append the dictionary to the results list
                                        # --------------------------------------------
                                        results_list.append(run_record)

                    # ----------------------------------------------------------------
                    # 3) Write to CSV *once* after all window_sizes for this setup
                    # ----------------------------------------------------------------
                    if len(results_list) > 0:
                        df = pd.DataFrame(results_list)
                        # If CSV already exists, append without header
                        if os.path.exists(results_csv_path):
                            df.to_csv(results_csv_path, mode='a', header=False, index=False)
                        else:
                            df.to_csv(results_csv_path, index=False)

                        # Clear results_list to avoid duplication
                        results_list = []
