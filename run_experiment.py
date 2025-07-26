import os
import argparse
import torch

from config.config import Config
from model_loader import load_pubmedbert_model
from data_loader import load_and_split_pubtator
from utils.evaluate import evaluate_model
from utils.logger import create_experiment_log_dir, save_json
from utils.metrics_logger import MetricsLogger
from trainers.central_trainer import centralized_train
from trainers.fedavg_trainer import FedAvgTrainer
from trainers.fedprox_trainer import FedProxTrainer
from trainers.fedadam_trainer import FedAdamTrainer
# from trainers.fedkd_trainer import FedKDTrainer  # if implemented


def run_federated_training(cfg, tokenizer, label_list, clients_data, test_sents, model_init, device):
    result_dir = create_experiment_log_dir(algorithm=cfg.algorithm)
    log = MetricsLogger()
    global_model = model_init().to(device)

    for r in range(cfg.rounds):
        print(f"=== Round {r+1}/{cfg.rounds} ===")
        log.start_timer()

        if cfg.algorithm == "FedAvg":
            trainer = FedAvgTrainer(
                model_init, tokenizer, label_list, device,
                epochs=cfg.local_epochs,
                learning_rate=cfg.learning_rate,
                scheduler_type=cfg.lr_scheduler_type,
                batch_size=cfg.train_batch_size
            )
        elif cfg.algorithm == "FedProx":
            trainer = FedProxTrainer(
                model_init, tokenizer, label_list, device,
                epochs=cfg.local_epochs,
                mu=0.01,
                learning_rate=cfg.learning_rate,
                scheduler_type=cfg.lr_scheduler_type,
                batch_size=cfg.train_batch_size
            )
        elif cfg.algorithm == "FedAdam":
            trainer = FedAdamTrainer(
                model_init, tokenizer, label_list, device,
                epochs=cfg.local_epochs,
                server_lr=0.001,
                learning_rate=cfg.learning_rate,
                scheduler_type=cfg.lr_scheduler_type,
                batch_size=cfg.train_batch_size
            )
        else:
            raise ValueError(f"Unsupported algorithm: {cfg.algorithm}")

        global_model = trainer.train_round(global_model, clients_data)

        # Evaluate on test sentences
        metrics = evaluate_model(global_model, tokenizer, test_sents, label_list)
        elapsed = log.stop_timer()
        log.log_round_metrics(r+1, metrics, elapsed)

        print(f" Round {r+1} | F1 {metrics['f1']:.4f} | "
              f"P {metrics['precision']:.4f} | R {metrics['recall']:.4f} | "
              f"Time {elapsed:.1f}s\n")

    save_json(log.get_logs(), os.path.join(result_dir, "fed_results.json"))


def run_centralized_training(cfg, tokenizer, label_list, train_sents, test_sents, model_init, device):
    result_dir = create_experiment_log_dir(algorithm="Centralized")
    log = MetricsLogger()
    model = model_init().to(device)

    log.start_timer()
    model = centralized_train(
        model, tokenizer, train_sents, label_list, device,
        epochs=cfg.local_epochs,
        learning_rate=cfg.learning_rate,
        scheduler_type=cfg.lr_scheduler_type,
        batch_size=cfg.train_batch_size
    )
    elapsed = log.stop_timer()

    metrics = evaluate_model(model, tokenizer, test_sents, label_list)
    log.log_round_metrics(1, metrics, elapsed)

    print(f" Centralized | F1 {metrics['f1']:.4f} | "
          f"P {metrics['precision']:.4f} | R {metrics['recall']:.4f} | "
          f"Time {elapsed:.1f}s")

    save_json(log.get_logs(), os.path.join(result_dir, "central_results.json"))


def main():
    parser = argparse.ArgumentParser(description="Federated NER with Dirichlet IID/Non‑IID Splits")
    parser.add_argument("--alg",           type=str,
                        choices=["FedAvg","FedProx","FedAdam","Centralized"],
                        help="Which training algorithm to use")
    parser.add_argument("--rounds",        type=int,   help="Number of communication rounds")
    parser.add_argument("--local_epochs",  type=int,   help="Local training epochs per round")
    parser.add_argument("--lr",            type=float, help="Learning rate")
    parser.add_argument("--partition_strategy",
                        type=str, choices=["iid","noniid"],
                        help="Data split strategy: 'iid' or 'noniid'")
    parser.add_argument("--noniid_alpha",
                        type=float,
                        help="Dirichlet alpha parameter (only for noniid)")
    args = parser.parse_args()

    cfg = Config("config/config.yaml")
    if args.alg:            cfg.algorithm          = args.alg
    if args.rounds:         cfg.rounds             = args.rounds
    if args.local_epochs:   cfg.local_epochs       = args.local_epochs
    if args.lr is not None: cfg.learning_rate      = args.lr
    if args.partition_strategy:
        cfg.partition_strategy = args.partition_strategy
    if args.noniid_alpha is not None:
        cfg.noniid_alpha  = args.noniid_alpha

    device = "cuda" if torch.cuda.is_available() and cfg.device_mode != "cpu" else "cpu"

    # Load and split data into sentence-level lists
    client_train_sents, dev_sents, test_sents = load_and_split_pubtator(
        cfg.pubtator_path,
        cfg.trng_pmids_path,
        cfg.dev_pmids_path,
        cfg.test_pmids_path,
        cfg.num_clients,
        cfg.partition_strategy,
        cfg.noniid_alpha
    )

    # Build label list from sentence labels
    label_set = set()
    for client_sents in client_train_sents:
        for sent in client_sents:
            label_set.update([lbl.split('-')[-1] for lbl in sent['labels'] if lbl != 'O'])
    label_list = sorted(label_set)

    tokenizer, _ = load_pubmedbert_model(cfg.model_name, label_list)
    model_init = lambda: load_pubmedbert_model(cfg.model_name, label_list)[1]

    if cfg.algorithm == "Centralized":
        # Flatten all clients for centralized
        all_train_sents = []
        for sents in client_train_sents:
            all_train_sents.extend(sents)
        run_centralized_training(
            cfg, tokenizer, label_list,
            train_sents=all_train_sents,
            test_sents=test_sents,
            model_init=model_init,
            device=device
        )
    else:
        run_federated_training(
            cfg, tokenizer, label_list,
            clients_data=client_train_sents,
            test_sents=test_sents,
            model_init=model_init,
            device=device
        )


if __name__ == "__main__":
    main()
