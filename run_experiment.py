import argparse
import torch
from model_loader import load_pubmedbert_model
from data_loader import load_medmentions_bio, extract_label_list_from_bio
from utils.evaluate import evaluate_model
from utils.logger import create_experiment_log_dir, save_json
from utils.metrics_logger import MetricsLogger
from trainers.central_trainer import centralized_train
from trainers.fedavg_trainer import FedAvgTrainer
from trainers.fedprox_trainer import FedProxTrainer
# from trainers.fedadam_trainer import FedAdamTrainer
# from trainers.fedkd_trainer import FedKDTrainer
import os
import time

def run_federated_training(args, tokenizer, label_list, clients_data, test_data, model_init, device):
    result_dir = create_experiment_log_dir(algorithm=args.alg)
    log = MetricsLogger()
    global_model = model_init().to(device)

    for r in range(args.rounds):
        print(f"📡 Round {r+1}/{args.rounds}")
        log.start_timer()

        if args.alg == "FedAvg":
            trainer = FedAvgTrainer(model_init, tokenizer, label_list, device=device, epochs=args.epochs)
        elif args.alg == "FedAdam":
            raise NotImplementedError("FedAdam not implemented yet")
        elif args.alg == "FedKD":
            raise NotImplementedError("FedKD not implemented yet")
        else:
            raise ValueError(f"Unknown algorithm: {args.alg}")

        global_model = trainer.train_round(global_model, clients_data)

        metrics = evaluate_model(global_model, tokenizer, test_data, label_list)
        elapsed = log.stop_timer()
        log.log_round_metrics(r + 1, metrics, elapsed)

        print(f"✅ Round {r+1} | F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | Time: {elapsed:.1f}s\n")

    save_json(log.get_logs(), os.path.join(result_dir, "fed_results.json"))

def run_centralized_training(args, tokenizer, label_list, full_train_data, test_data, model_init, device):
    result_dir = create_experiment_log_dir(algorithm="Centralized")
    log = MetricsLogger()
    model = model_init().to(device)

    log.start_timer()
    model = centralized_train(model, tokenizer, full_train_data, label_list, device=device, epochs=args.rounds * args.epochs)
    elapsed = log.stop_timer()

    metrics = evaluate_model(model, tokenizer, test_data, label_list)
    log.log_round_metrics(1, metrics, elapsed)

    print(f"🏢 Centralized | F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | Time: {elapsed:.1f}s")
    save_json(log.get_logs(), os.path.join(result_dir, "central_results.json"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, choices=["FedAvg", "FedAdam", "FedKD", "Centralized"], default="FedAvg")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train", type=str, default="train.bio")
    parser.add_argument("--dev", type=str, default="dev.bio")
    parser.add_argument("--test", type=str, default="test.bio")
    parser.add_argument("--model", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_list = extract_label_list_from_bio(args.train)
    tokenizer, base_model = load_pubmedbert_model(args.model, label_list)
    model_init = lambda: load_pubmedbert_model(args.model, label_list)[1]

    clients_data, _, test_data = load_medmentions_bio(args.train, args.dev, args.test, num_clients=args.clients)

    if args.alg == "Centralized":
        all_train_data = [item for sublist in clients_data for item in sublist]
        run_centralized_training(args, tokenizer, label_list, all_train_data, test_data, model_init, device)
    else:
        run_federated_training(args, tokenizer, label_list, clients_data, test_data, model_init, device)

if __name__ == "__main__":
    main()