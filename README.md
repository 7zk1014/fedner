# FedNER

FedNER is a simple framework for running federated learning experiments on Named Entity Recognition (NER) datasets. It relies on PubMedBERT token classification models and includes implementations of several federated algorithms.

## Repository Layout

```
.
├── run_experiment.py           # Entry script to launch training
├── download_model.py           # Helper to cache PubMedBERT locally
├── config/
│   ├── config.yaml             # Experiment parameters
│   └── config.py               # Config loader helper
├── aggregators/
│   └── fedavg.py               # Model weight averaging
├── trainers/                   # Training logic for each algorithm
│   ├── base_trainer.py
│   ├── central_trainer.py
│   ├── fedavg_trainer.py
│   ├── fedprox_trainer.py
│   ├── fedadam_trainer.py
│   ├── fedsd_trainer.py        # Baseline distillation
│   └── fedsad_trainer.py       # Self-adaptive distillation
├── utils/                      # Logging and evaluation helpers
│   ├── evaluate.py
│   ├── evaluate_global_on_local.py
│   ├── logger.py
│   ├── metrics_logger.py
│   ├── plot_metrics.py
│   ├── streaming.py            # Gradient compression helpers
│   └── training_utils.py       # Shared training utilities
├── data_loader.py              # Read BIO files and split among clients
├── model_loader.py             # Load PubMedBERT model and tokenizer
├── run_seq.sh                  # Example batch experiment script
├── data_set/                   # Example dataset in BIO format
└── results/                    # Generated metrics and plots
```

## Key Features

- **Multiple algorithms**: FedAvg, FedProx, FedAdam, FedSD, FedSAD and a centralized baseline are available.
- **Config-driven**: Parameters such as number of clients, rounds, and learning rate are defined in `config/config.yaml` and can be overridden via command line.
- **Metrics logging**: After each round the script logs F1, precision, recall and accuracy and stores them under `results/`.
- **Visualization utilities**: Scripts in `utils/` such as `plot_metrics.py` help visualize learning curves.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Download the pretrained model locally:
   ```bash
   python download_model.py
   ```
3. Prepare your dataset in BIO format and place `train.bio`, `dev.bio` and `test.bio` in the `data_set/` folder.
4. Adjust settings in `config/config.yaml` if needed. Example excerpt:
   ```yaml
   training:
     algorithm: FedAvg
     num_clients: 5
     rounds: 5
     local_epochs: 2
   hyperparameters:
     learning_rate: 3e-5
     train_batch_size: 32
   ```
5. Launch an experiment:
   ```bash
   python run_experiment.py --alg FedAvg --rounds 5
   ```
   Replace `FedAvg` with `FedProx`, `FedAdam`, `FedSD`, `FedSAD` or `Centralized` to try other modes.

## Results

Metrics for each run are saved under `results/<algorithm>_<timestamp>/`. Visualize the training curve with:

```bash
python utils/plot_metrics.py
```

## Centralized Baseline

To train on the full dataset without federated updates:

```bash
python run_experiment.py --alg Centralized
```

---

FedNER aims to provide a minimal and extensible starting point for exploring federated learning on token classification tasks. Contributions are welcome.

