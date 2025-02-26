import torch.multiprocessing as mp
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from training_loop import run_entire_loop

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run financial prediction training loop.")

    # Add parameters that can be changed from the command line
    parser.add_argument("--model_type", type=str, default="gru", help="Type of model (e.g., gru, lstm)")
    parser.add_argument("--processing", nargs="+", default=["clean"], help="Preprocessing steps")
    parser.add_argument("--stocks", nargs="+", default=['AAPL', 'MSFT', 'AMZN', 'NVDA', 'SPX'], help="List of stocks to analyze")
    parser.add_argument("--types_securities", nargs="+", default=["options"], help="Types of securities")
    parser.add_argument("--years", nargs="+", default=["15y", "10y", "5y", "2y"], help="List of years to consider")
    parser.add_argument("--window_sizes", nargs="+", type=int, default=[5, 10, 50, 100], help="Window sizes")
    parser.add_argument("--train_sizes", nargs="+", type=int, default=[80, 90, 95], help="Training sizes")
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[0.008, 0.009, 0.0095, 0.01], help="Learning rates")
    parser.add_argument("--num_epochs_list", nargs="+", type=int, default=[100, 200], help="Epoch counts")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[16, 32, 64], help="Batch sizes")
    parser.add_argument("--prediction_thresholds", nargs="+", type=float, default=[0.35, 0.4, 0.45, 0.5], help="Prediction thresholds")
    parser.add_argument("--project_dir", type=str, default="/home/jupyter-tfg2425paula/prediction_project_v3", help="Project directory")

    return parser.parse_args()

def main():
    args = parse_arguments()  # Parse command-line arguments

    # Set up project directories
    os.chdir(args.project_dir)
    clean_data_dir = os.path.join(args.project_dir, "00_data/clean")
    horizontal_data_dir = os.path.join(args.project_dir, "00_data/horizontal_structure")
    results_dir = os.path.join(args.project_dir, "02_results")

    # CUDA settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model settings
    hidden_size = 64  
    output_size = 2  
    num_layers = 2
    dropout = 0.2

    criterion = nn.CrossEntropyLoss()

    # Prevent CUDA reinitialization errors
    mp.set_start_method("spawn", force=True)

    print("Starting Training Loop in a New Process...", flush=True)

    # Create and start the multiprocessing process
    p = mp.Process(target=run_entire_loop, kwargs=vars(args))
    p.start()
    p.join()  # Wait for it to complete

    print("All done!", flush=True)

if __name__ == "__main__":
    main()
