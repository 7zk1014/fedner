import os
import json
import matplotlib.pyplot as plt

def plot_f1_vs_rounds(results_dir):
    with open(os.path.join(results_dir, "metrics.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    rounds = list(range(1, len(data["global_metrics"]) + 1))
    f1s = [m["f1"] for m in data["global_metrics"]]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, f1s, marker='o', label="Global F1")
    plt.xlabel("Communication Round")
    plt.ylabel("F1 Score")
    plt.title("Global F1 vs. Communication Rounds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "f1_vs_rounds.png"))
    plt.close()

def plot_local_f1_distribution(results_dir):
    with open(os.path.join(results_dir, "metrics.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    last_local_f1s = data.get("last_local_f1", [])

    if not last_local_f1s:
        print("No local F1 data found.")
        return

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(last_local_f1s)), last_local_f1s)
    plt.xlabel("Client ID")
    plt.ylabel("Local F1 Score")
    plt.title("Final Local F1 Scores Across Clients")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "local_f1_distribution.png"))
    plt.close()