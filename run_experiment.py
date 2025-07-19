import torch
import os
import time
from config.config import Config
from model_loader import load_pubmedbert_model
from data_loader import load_medmentions_bio, extract_label_list_from_bio
from utils.evaluate import evaluate_model
from utils.logger import create_experiment_log_dir, save_json
from utils.metrics_logger import MetricsLogger
from trainers.central_trainer import centralized_train
from trainers.fedavg_trainer import FedAvgTrainer
from trainers.fedprox_trainer import FedProxTrainer
import argparse
from trainers.fedadam_trainer import FedAdamTrainer
# from trainers.fedkd_trainer import FedKDTrainer

def run_federated_training(cfg, tokenizer, label_list, clients_data, test_data, model_init, device):
    result_dir = create_experiment_log_dir(algorithm=cfg.algorithm)
    log = MetricsLogger()
    global_model = model_init().to(device)

    for r in range(cfg.rounds):
        print(f" Round {r+1}/{cfg.rounds}")
        log.start_timer()

        if cfg.algorithm == "FedAvg":
            trainer = FedAvgTrainer(
                model_init=model_init,
                tokenizer=tokenizer,
                label_list=label_list,
                device=device,
                epochs=cfg.local_epochs,
                learning_rate=cfg.learning_rate,
                scheduler_type=cfg.lr_scheduler_type,
                batch_size=cfg.train_batch_size
            )
        elif cfg.algorithm == "FedProx":
            trainer = FedProxTrainer(
            model_init=model_init,
            tokenizer=tokenizer,
            label_list=label_list,
            device=device,
            epochs=cfg.local_epochs,
            mu=0.01,
            batch_size=cfg.train_batch_size,
            learning_rate=cfg.learning_rate,
            scheduler_type=cfg.lr_scheduler_type
) 
          
        elif cfg.algorithm == "FedAdam":
            trainer = FedAdamTrainer(
            model_init=model_init,
            tokenizer=tokenizer,
            label_list=label_list,
            device=device,
            epochs=cfg.local_epochs,
            learning_rate=cfg.learning_rate,
            scheduler_type=cfg.lr_scheduler_type,
            batch_size=cfg.train_batch_size,
            server_lr=0.001 
    )
        elif cfg.algorithm == "FedKD":
            raise NotImplementedError("FedKD not implemented yet")
        else:
            raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

        global_model = trainer.train_round(global_model, clients_data)
        metrics = evaluate_model(global_model, tokenizer, test_data, label_list)
        elapsed = log.stop_timer()
        log.log_round_metrics(r + 1, metrics, elapsed)

        print(f"5 Round {r+1} | F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | Time: {elapsed:.1f}s\n")

    save_json(log.get_logs(), os.path.join(result_dir, "fed_results.json"))

def run_centralized_training(cfg, tokenizer, label_list, full_train_data, test_data, model_init, device):
    result_dir = create_experiment_log_dir(algorithm="Centralized")
    log = MetricsLogger()
    model = model_init().to(device)

    log.start_timer()
    model = centralized_train(
        model=model,
        tokenizer=tokenizer,
        train_examples=full_train_data,
        label_list=label_list,
        device=device,
        epochs=cfg.local_epochs,
        learning_rate=cfg.learning_rate,
        scheduler_type=cfg.lr_scheduler_type,
        batch_size=cfg.train_batch_size
    )
    elapsed = log.stop_timer()

    metrics = evaluate_model(model, tokenizer, test_data, label_list)
    log.log_round_metrics(1, metrics, elapsed)

    print(f" Centralized | F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | Time: {elapsed:.1f}s")
    save_json(log.get_logs(), os.path.join(result_dir, "central_results.json"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, choices=["FedAvg", "FedProx", "FedAdam", "FedKD", "Centralized"], help="Training algorithm")
    parser.add_argument("--rounds", type=int, help="Number of communication rounds")
    parser.add_argument("--local_epochs", type=int, help="Local training epochs per round")
    parser.add_argument("--lr", type=float, help="Learning rate")
    args = parser.parse_args()

    cfg = Config("config/config.yaml")
    if args.alg is not None:
        cfg.algorithm = args.alg
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.local_epochs is not None:
        cfg.local_epochs = args.local_epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr

    device = "cuda" if torch.cuda.is_available() and cfg.device_mode != "cpu" else "cpu"
    label_list = extract_label_list_from_bio(cfg.train_path)
    tokenizer, base_model = load_pubmedbert_model(cfg.model_name, label_list)
    model_init = lambda: load_pubmedbert_model(cfg.model_name, label_list)[1]

    clients_data, _, test_data = load_medmentions_bio(cfg.train_path, cfg.dev_path, cfg.test_path, num_clients=cfg.num_clients)

    if cfg.algorithm == "Centralized":
        all_train_data = [item for sublist in clients_data for item in sublist]
        run_centralized_training(cfg, tokenizer, label_list, all_train_data, test_data, model_init, device)
    else:
        run_federated_training(cfg, tokenizer, label_list, clients_data, test_data, model_init, device)

if __name__ == "__main__":
    main()
