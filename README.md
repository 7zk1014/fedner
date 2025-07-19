# FedNER: Federated Named Entity Recognition Framework

This repository provides a modular framework for evaluating different federated learning algorithms on Named Entity Recognition (NER) tasks using BioBERT-based models.

## 📁 Directory Structure

```
.
├── run_experiment.py              # Main entry script
├── config/
│   ├── config.py                  # Python-based config reader
│   └── config.yaml                # Hyperparameter settings (learning rate, batch size, etc.)
├── trainers/
│   ├── base_trainer.py           # Base class for all trainers
│   ├── fedavg_trainer.py         # FedAvg implementation
│   ├── fedprox_trainer.py        # FedProx implementation
│   ├── fedadam_trainer.py        # FedAdam implementation
│   ├── fedkd_trainer.py          # (To be implemented) FedKD placeholder
│   └── central_trainer.py        # Centralized training baseline
├── utils/
│   ├── logger.py                 # Timestamp logger utility
│   ├── metrics_logger.py         # Metric/communication/time logging
│   ├── evaluate.py               # Evaluate global model on test set
│   └── evaluate_global_on_local.py # Evaluate global model on each client validation set
├── aggregators/
│   └── fedavg.py                 # FedAvg aggregation logic
├── test_bio/                     # Folder to place `train.bio`, `dev.bio`, `test.bio`
├── results/                      # Output logs and metrics (per experiment)
└── saved_models/                 # Checkpoints of global models
```

## 🚀 How to Run an Experiment

1. Put your BIO-format dataset into `test_bio/`:
    - `train.bio`, `dev.bio`, `test.bio`

2. Set the desired algorithm and run:

```bash
# Edit this line in run_experiment.py
fed_algorithm = "FedAvg"  # Options: FedAvg, FedProx, FedAdam, FedKD

# Then run:
python run_experiment.py
```

## 🧪 Evaluation Metrics

- Global model: Precision / Recall / F1 (on test set)
- Local generalization: mean / std of F1 (on local dev sets)
- Training time: per round and total
- Communication cost: estimated from model size × clients × rounds

All metrics are saved in `results/{algorithm}_{timestamp}/metrics.json`.

## ⚙️ Centralized Training

Run centralized baseline for comparison:

```bash
python trainers/central_trainer.py
```

## ✅ Example Config (config/config.yaml)

```
model_name: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
local_epochs: 2
global_rounds: 5
num_clients: 5
learning_rate: 5e-5
batch_size: 32
max_seq_length: 128
seed: 42
```

## 📌 Notes

- `FedKD` is a placeholder; future work can implement distillation logic.
- Results are saved in timestamped folders under `results/` to avoid overwriting.


For any questions or issues, feel free to reach out!

## 📊 Comparing Multiple Algorithms

To compare the performance (e.g., global F1 vs. communication rounds) of multiple algorithms:

```bash
python compare_algorithms.py
```

This script will:
- Scan all subdirectories in `results/` (e.g., `FedAvg_20250719_153222`, `FedProx_20250719_160000`)
- Load each `metrics.json` file
- Plot a comparison chart saved as `results/f1_comparison.png`

Make sure each experiment run saved metrics correctly.
