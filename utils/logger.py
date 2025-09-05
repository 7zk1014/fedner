import os
import json
from datetime import datetime

def create_experiment_log_dir(base_dir="results", algorithm="FedAvg"):
    """
    Create timestamped experiment results directory and record to current_results_path.txt
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, f"{algorithm}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(base_dir, "current_results_path.txt"), "w") as f:
        f.write(result_dir)
    return result_dir

# Recursively convert non-serializable objects for JSON serialization
def _convert_for_json(obj):
    if isinstance(obj, set):
        return list(obj)
    elif obj is ...:  # Handle ellipsis objects
        return None
    elif isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_for_json(x) for x in obj]
    return obj

def save_json(data, filename):
    """
    Save JSON with timestamped backup.
    Automatically handles non-serializable types like set/ellipsis.
    """
    data = _convert_for_json(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save main file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Save timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{os.path.splitext(filename)[0]}_{timestamp}.json"
    with open(backup_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[save_json] Saved: {filename}")
    print(f"[save_json] Backup: {backup_filename}")
