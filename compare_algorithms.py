import os
import json
import matplotlib.pyplot as plt

def load_metrics(result_dir):
    metrics_path = os.path.join(result_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare_f1_curves(base_dir):
    algo_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    algo_dirs.sort()

    plt.figure(figsize=(10, 6))
    for algo in algo_dirs:
        metrics = load_metrics(os.path.join(base_dir, algo))
        if not metrics:
            continue
        f1s = [m["f1"] for m in metrics["global_metrics"]]
        rounds = list(range(1, len(f1s) + 1))
        plt.plot(rounds, f1s, marker='o', label=algo)

    plt.xlabel("Communication Round")
    plt.ylabel("Global F1 Score")
    plt.title("Federated NER: F1 Score Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "f1_comparison.png"))
    plt.show()

if __name__ == "__main__":
    base_results_dir = "results"
    compare_f1_curves(base_results_dir)