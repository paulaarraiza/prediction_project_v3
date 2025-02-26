import torch.multiprocessing as mp
from training_loop import run_entire_loop, reshape_remove_characters, evaluate_rolling_unchanged_model_threshold
from training_loop_lstm import run_entire_loop_lstm
import os
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------

model_type = "gru"

if model_type == "gru":
    target = run_entire_loop
elif model_type == "lstm":
    target = run_entire_loop_lstm
    
processing = ["clean", "pca"]
processing = ["clean"]

stocks = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'SPX']
# stocks = ['AAPL']
types_securities = ["single_name", "options", "technical"]
types_securities = ["options"]

years = ["15y", "10y", "5y", "2y"]
# years = ["15y"]

window_sizes = [5, 10, 50, 100]
window_sizes = [5]

train_sizes = [80, 90, 95]
train_sizes = [80]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_size = 64  
output_size = 2  
num_layers = 2
dropout = 0.2

criterion = nn.CrossEntropyLoss()

learning_rates = [0.0095, 0.01, 0.011]
learning_rates = [0.01]
num_epochs_list = [100, 200]
num_epochs_list = [100]
batch_sizes = [16, 32, 64]
batch_sizes = [32]

prediction_thresholds = [0.35, 0.4, 0.45, 0.5]
prediction_thresholds = [0.5]

project_dir = "/home/jupyter-tfg2425paula/prediction_project_v3"
os.chdir(project_dir)

clean_data_dir = os.path.join(project_dir, "00_data/clean")
horizontal_data_dir = os.path.join(project_dir, "00_data/horizontal_structure")
results_dir = os.path.join(project_dir, "02_results")

# ------------------------------------------------------------------
# RUN MODEL
# ------------------------------------------------------------------

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)  # Prevent CUDA reinitialization errors
    except RuntimeError:
        pass

    print("Starting Training Loop in a New Process...", flush=True)
    
    p = mp.Process(target=target, args=(
        stocks, model_type, processing, types_securities, years, window_sizes,
        train_sizes, learning_rates, num_epochs_list, batch_sizes, prediction_thresholds, project_dir
    ))  # Create a new process
    p.start()
    p.join()  # Wait for it to complete
    
    print("All done!", flush=True)
