import os
import json
from datetime import datetime

def create_experiment_log_dir(base_dir="results", algorithm="FedAvg"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, f"{algorithm}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(base_dir, "current_results_path.txt"), "w") as f:
        f.write(result_dir)
    return result_dir

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
