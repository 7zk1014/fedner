import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(result_dirs):
    results = {}
    for d in result_dirs:
        name = os.path.basename(d)
        path = os.path.join(d, "fed_results.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results[name] = data
    return results

def plot_f1_vs_rounds(results):
    plt.figure(figsize=(8, 5))
    for name, data in results.items():
        f1s = data["metrics"]["history"]["f1"]
        rounds = data["metrics"]["history"]["round"]
        plt.plot(rounds, f1s, marker='o', label=name)
    plt.xlabel("Communication Round")
    plt.ylabel("Global F1 Score")
    plt.title("Global F1 vs. Communication Rounds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_f1_vs_rounds.png")
    plt.close()

def plot_comm_vs_rounds(results):
    plt.figure(figsize=(8, 5))
    for name, data in results.items():
        comms = data["metrics"]["history"]["comm_mb"]
        rounds = data["metrics"]["history"]["round"]
        plt.plot(rounds, comms, marker='x', label=name)
    plt.xlabel("Communication Round")
    plt.ylabel("Uploaded MB")
    plt.title("Communication Cost vs. Rounds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_comm_vs_rounds.png")
    plt.close()

def plot_total_comm(results):
    names = []
    totals = []
    for name, data in results.items():
        names.append(name)
        totals.append(data["metrics"]["total_comm_mb"])
    plt.figure(figsize=(7, 5))
    plt.bar(names, totals)
    plt.ylabel("Total Uploaded (MB)")
    plt.title("Total Communication Cost")
    plt.tight_layout()
    plt.savefig("compare_total_comm.png")
    plt.close()

def plot_final_local_f1(results):
    plt.figure(figsize=(9, 5))
    for name, data in results.items():
        f1s = data.get("last_local_f1", [])
        if f1s:
            plt.plot(range(len(f1s)), f1s, marker='o', label=f"{name} (mean={np.mean(f1s):.3f}, std={np.std(f1s):.3f})")
    plt.xlabel("Client ID")
    plt.ylabel("Final Local F1 Score")
    plt.title("Final Local F1 Score Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_local_f1_distribution.png")
    plt.close()
