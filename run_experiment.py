import os
os.environ["WANDB_DISABLED"] = "true"

import argparse
import torch
import random
import numpy as np
import yaml

from config.config import Config
from model_loader import load_pubmedbert_model
from data_loader import load_and_split_pubtator
from utils.evaluate import evaluate_model
from utils.evaluate_global_on_local import evaluate_global_on_local
from utils.logger import create_experiment_log_dir, save_json
from utils.metrics_logger import MetricsLogger
from utils.training_utils import (
    freeze_model_layers,
    sample_client_data,
    select_clients_for_round,
    print_data_statistics,
    get_trainable_params_info
)

from trainers.central_trainer import centralized_train
from trainers.fedavg_trainer import FedAvgTrainer
from trainers.fedprox_trainer import FedProxTrainer
from trainers.fedadam_trainer import FedAdamTrainer
from trainers.fedsad_trainer import FedSADTrainer
from trainers.fedsd_trainer import FedSDTrainer


def set_global_seed(seed=42):
    """Sets a global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _iter_sents(obj):
    """
    Recursively iterates through data objects, yielding sentence dictionaries
    of the form {'tokens': [...], 'labels': [...]}.
    Handles various nested structures (dict/list).
    """
    if obj is None:
        return
    # Innermost sentence dictionary
    if isinstance(obj, dict) and "labels" in obj and "tokens" in obj:
        yield obj
        return
    # Dict: expand values
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_sents(v)
        return
    # List/Tuple: expand items
    if isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_sents(x)
        return


def run_federated_training(cfg, tokenizer, label_list, clients_data, dev_sents, test_sents, model_init, device):
    """Federated training main function with unified training control."""
    
    result_dir = create_experiment_log_dir(algorithm=cfg.algorithm)
    log = MetricsLogger(cfg)
    global_model = model_init().to(device)
    
    # Print training configuration
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(cfg.get_training_info())
    print("-" * 60)
    # Print algorithm-specific parameters
    print(f"Algorithm-specific parameters for {cfg.algorithm}:")
    if cfg.algorithm == "FedProx":
        print(f"  - mu: {cfg.mu}")
    elif cfg.algorithm == "FedAdam":
        print(f"  - server_lr: {cfg.server_lr}")
    elif cfg.algorithm == "FedSAD":
        print(f"  - use_distillation: {cfg.use_distillation}")
        print(f"  - kd_start_round: {cfg.kd_start_round}")
        print(f"  - teacher_policy: {cfg.teacher_policy}")
        print(f"  - temperature_start: {cfg.temperature_start}, temperature_end: {cfg.temperature_end}")
        print(f"  - alpha_ce_start: {cfg.alpha_ce_start}, alpha_ce_end: {cfg.alpha_ce_end}")
        print(f"  - alpha_kd_start: {cfg.alpha_kd_start}, alpha_kd_end: {cfg.alpha_kd_end}")
        print(f"  - hidden_distill: {cfg.hidden_distill}, alpha_hidden: {cfg.alpha_hidden}")
        print(f"  - o_downweight: {cfg.o_downweight}")
    elif cfg.algorithm == "FedSD":
        print(f"  - fedsd_alpha_ce: {cfg.fedsd_alpha_ce}")
        print(f"  - fedsd_alpha_kl: {cfg.fedsd_alpha_kl}")
        print(f"  - fedsd_alpha_hidden: {cfg.fedsd_alpha_hidden}")
        print(f"  - fedsd_temperature: {cfg.fedsd_temperature}")
    print("="*60 + "\n")
    
    # Print model information
    params_info = get_trainable_params_info(global_model)
    print(f"Model initialized: {params_info['trainable_params']:,}/{params_info['total_params']:,} "
          f"trainable params ({params_info['trainable_ratio']*100:.1f}%)")

    # ==================================================================
    # ▼▼▼ Core change: Trainer initialization moved outside the loop ▼▼▼
    # We create the trainer object once before all training rounds.
    # This ensures the trainer's internal state (like round_idx, teacher_model)
    # is preserved and updated across all rounds.
    # ==================================================================
    trainer_kwargs = {
        "model_init": model_init,
        "tokenizer": tokenizer,
        "label_list": label_list,
        "device": device,
        "epochs": cfg.local_epochs,
        "learning_rate": cfg.learning_rate,
        "scheduler_type": cfg.lr_scheduler_type,
        "batch_size": cfg.train_batch_size,
        # Unified training control parameters
        "train_last_n": cfg.train_last_n_layers,
        "sample_size": cfg.sample_size,
        "sample_strategy": cfg.sample_strategy,
        "max_seq_length": cfg.max_seq_length
    }

    if cfg.algorithm == "FedAvg":
        trainer = FedAvgTrainer(**trainer_kwargs)
            
    elif cfg.algorithm == "FedProx":
        trainer = FedProxTrainer(mu=cfg.mu, **trainer_kwargs)
            
    elif cfg.algorithm == "FedAdam":
        trainer = FedAdamTrainer(server_lr=cfg.server_lr, **trainer_kwargs)
            
    elif cfg.algorithm == "FedSD":
        trainer_kwargs.update({
            "alpha_ce": cfg.fedsd_alpha_ce,
            "alpha_kl": cfg.fedsd_alpha_kl,
            "alpha_hidden": cfg.fedsd_alpha_hidden,
            "temperature": cfg.fedsd_temperature,
        })
        trainer = FedSDTrainer(**trainer_kwargs)

    elif cfg.algorithm == "FedSAD":
        trainer_kwargs.update({
            "rounds": cfg.rounds,
            "use_distillation": cfg.use_distillation,
            "kd_start_round": cfg.kd_start_round,
            "teacher_policy": cfg.teacher_policy,
            "temperature_start": cfg.temperature_start,
            "temperature_end": cfg.temperature_end,
            "alpha_ce_start": cfg.alpha_ce_start,
            "alpha_ce_end": cfg.alpha_ce_end,
            "alpha_kd_start": cfg.alpha_kd_start,
            "alpha_kd_end": cfg.alpha_kd_end,
            "hidden_distill": cfg.hidden_distill,
            "alpha_hidden": cfg.alpha_hidden,
            "o_downweight": cfg.o_downweight,
            "compress_gradients": cfg.compress_gradients,
            "compress_topk": cfg.compress_topk,
            "compress_min_dim": cfg.compress_min_dim,
            "first_ema_m0": cfg.first_ema_m0,
            "ema_m_min": cfg.ema_m_min,
            "ema_m_max": cfg.ema_m_max,
        })
        trainer = FedSADTrainer(**trainer_kwargs)

    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm}")

    # Main training loop
    for r in range(cfg.rounds):
        print(f"\n=== Round {r+1}/{cfg.rounds} ===")
        log.start_timer()
        
        # Select clients for the current round
        selected_clients = select_clients_for_round(
            num_clients=cfg.num_clients,
            fraction=cfg.client_fraction,
            min_clients=cfg.min_clients_per_round,
            round_idx=r,
            seed=cfg.seed
        )
        
        if cfg.verbose:
            print(f"Selected clients: {selected_clients} ({len(selected_clients)}/{cfg.num_clients})")
        
        # Get data for the selected clients
        selected_client_data = [clients_data[i] for i in selected_clients]
        
        # ==================================================================
        # ▼▼▼ Core change: No trainer creation inside the loop ▼▼▼
        # The single, pre-created trainer object is used directly to run a round.
        # ==================================================================
        global_model = trainer.train_round(global_model, selected_client_data)
        
        # Evaluate
        metrics = evaluate_model(
            global_model, tokenizer, test_sents, label_list,
            device=device, batch_size=cfg.eval_batch_size, 
            max_length=cfg.max_seq_length, num_workers=0
        )
        
        # Evaluate performance on each client's dev set
        local_dev_metrics = evaluate_global_on_local(
            global_model, tokenizer,
            clients_dev_sets=dev_sents,
            label_list=label_list,
            logger=log,
            round_num=r + 1
        )
        
        elapsed = log.stop_timer()
        comm_mb = getattr(trainer, "last_uploaded_size_mb", None)
        comm_str = f"{comm_mb:.2f} MB" if isinstance(comm_mb, (int, float)) else "N/A"
        log.log_round_metrics(r+1, metrics, elapsed,
                              comm_mb=comm_mb if isinstance(comm_mb, (int, float)) else None)
        
        print(f" Round {r+1} | F1 {metrics['f1']:.4f} | "
              f"P {metrics['precision']:.4f} | R {metrics['recall']:.4f} | "
              f"Time {elapsed:.1f}s | Comm {comm_str} | "
              f"Local Avg F1: {local_dev_metrics['mean_f1']:.4f}")
    
    # Save results
    save_json({
        "algorithm": cfg.algorithm,
        "config": cfg.as_dict(),
        "metrics": log.get_logs()
    }, os.path.join(result_dir, "fed_results.json"))


def run_centralized_training(cfg, tokenizer, label_list, train_sents, test_sents, model_init, device):
    """
    Runs centralized training.
    """
    result_dir = create_experiment_log_dir(algorithm="Centralized")
    log = MetricsLogger(cfg)
    model = model_init().to(device)

    # Print configuration
    print("\n" + "="*60)
    print("CENTRALIZED TRAINING")
    print("="*60)
    print(cfg.get_training_info())
    print("="*60 + "\n")

    log.start_timer()
    model = centralized_train(
        model, tokenizer, train_sents, label_list, device,
        epochs=cfg.local_epochs,
        learning_rate=cfg.learning_rate,
        scheduler_type=cfg.lr_scheduler_type,
        batch_size=cfg.train_batch_size,
        train_last_n=cfg.train_last_n_layers,  # Ensure centralized training also applies this param
        max_seq_length=cfg.max_seq_length
    )
    elapsed = log.stop_timer()

    metrics = evaluate_model(
        model, tokenizer, test_sents, label_list,
        device=device, batch_size=cfg.eval_batch_size,
        max_length=cfg.max_seq_length, num_workers=0
    )
    log.log_round_metrics(1, metrics, elapsed)

    print(f" Centralized | F1 {metrics['f1']:.4f} | "
          f"P {metrics['precision']:.4f} | R {metrics['recall']:.4f} | "
          f"Time {elapsed:.1f}s")

    save_json({
        "algorithm": cfg.algorithm,
        "config": cfg.as_dict(),
        "metrics": log.get_logs()
    }, os.path.join(result_dir, "fed_results.json"))


def main():
    """Main function: parses command-line arguments, loads config and data, and runs the specified training algorithm."""
    set_global_seed(42)
    parser = argparse.ArgumentParser(description="Federated NER with unified training control")
    
    # Organize command-line arguments into clear groups
    # General parameters
    parser.add_argument("--alg", type=str, choices=["FedAvg", "FedProx", "FedAdam", "FedSAD", "FedSD", "Centralized"], help="Which training algorithm to use")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--model_name", type=str, help="HF model name or local path")

    # Training parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument("--rounds", type=int, help="Number of federated rounds")
    train_group.add_argument("--local_epochs", type=int, help="Local training epochs")
    train_group.add_argument("--lr", type=float, help="Learning rate")
    train_group.add_argument("--num_clients", type=int, help="Number of clients")

    # Training control parameters
    control_group = parser.add_argument_group("Training Control")
    control_group.add_argument("--train_layers", type=int, help="Number of last layers to train (-1 for all, 0 for head only)")
    control_group.add_argument("--sample_size", type=str, help="Sample size per client ('all' for no sampling, or integer)")
    control_group.add_argument("--sample_strategy", type=str, choices=["random", "balanced", "stratified"], help="Sampling strategy")
    control_group.add_argument("--client_fraction", type=float, help="Fraction of clients to select each round (0-1)")

    # Data partition parameters
    data_group = parser.add_argument_group("Data Partition")
    data_group.add_argument("--partition_strategy", type=str, choices=["iid", "noniid"], help="Data partition strategy")
    data_group.add_argument("--noniid_alpha", type=float, help="Dirichlet alpha for non-iid partition")
    
    # FedProx/FedAdam/FedSAD/FedSD specific parameters
    alg_specific_group = parser.add_argument_group("Algorithm-Specific Parameters")
    alg_specific_group.add_argument("--mu", type=float, help="FedProx: mu parameter for proximal term")
    alg_specific_group.add_argument("--server_lr", type=float, help="FedAdam: Server-side learning rate")
    
    # FedSAD specific parameters
    alg_specific_group.add_argument("--temperature_start", type=float, help="FedSAD: Initial distillation temperature")
    alg_specific_group.add_argument("--temperature_end", type=float, help="FedSAD: Final distillation temperature")
    alg_specific_group.add_argument("--alpha_ce_start", type=float, help="FedSAD: Initial cross-entropy loss weight")
    alg_specific_group.add_argument("--alpha_ce_end", type=float, help="FedSAD: Final cross-entropy loss weight")
    alg_specific_group.add_argument("--alpha_kd_start", type=float, help="FedSAD: Initial KD loss weight")
    alg_specific_group.add_argument("--alpha_kd_end", type=float, help="FedSAD: Final KD loss weight")
    alg_specific_group.add_argument("--kd_start_round", type=int, help="FedSAD: Round to start distillation (1-based)")
    alg_specific_group.add_argument("--teacher_policy", type=str, choices=["ema_first", "cold_then_ema", "prev_global"], help="FedSAD: Teacher policy")
    alg_specific_group.add_argument("--o_downweight", type=float, help="FedSAD O label downweight")
    alg_specific_group.add_argument("--first_ema_m0", type=float, help="FedSAD: First KD round EMA mix with G0 (ema_first)")
    alg_specific_group.add_argument("--ema_m_min", type=float, help="FedSAD: Min EMA momentum after KD starts")
    alg_specific_group.add_argument("--ema_m_max", type=float, help="FedSAD: Max EMA momentum after KD starts")
    # FedSD specific parameters
    alg_specific_group.add_argument("--fedsd_alpha_ce", type=float, help="FedSD: Cross-entropy loss weight")
    alg_specific_group.add_argument("--fedsd_alpha_kl", type=float, help="FedSD: KL divergence loss weight")
    alg_specific_group.add_argument("--fedsd_temperature", type=float, help="FedSD: Distillation temperature")
    
    args = parser.parse_args()
    
    # Load config and apply command-line overrides
    cfg = Config(args.config)
    
    overrides = {
        k: v for k, v in vars(args).items() if v is not None
    }
    
    # Remove 'config' key as it's not an overridable attribute of the Config class
    if 'config' in overrides:
        del overrides['config']

    # Handle special cases where CLI arg name differs from Config attribute name
    if "alg" in overrides:
        overrides["algorithm"] = overrides.pop("alg")
    if "lr" in overrides:
        overrides["learning_rate"] = overrides.pop("lr")
    if "train_layers" in overrides:
        overrides["train_last_n_layers"] = overrides.pop("train_layers")
        
    cfg.apply_overrides(**overrides)

    device = "cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu"
    
    # Load data
    client_train_sents, dev_sents, test_sents = load_and_split_pubtator(
        cfg.pubtator_path,
        cfg.trng_pmids_path,
        cfg.dev_pmids_path,
        cfg.test_pmids_path,
        cfg.num_clients,
        cfg.partition_strategy,
        cfg.noniid_alpha
    )

    # Aggregate labels and ensure "O" is included
    label_set = set()
    for sent in _iter_sents(client_train_sents):
        label_set.update(sent["labels"])
    for sent in _iter_sents(dev_sents):
        label_set.update(sent["labels"])
    for sent in _iter_sents(test_sents):
        label_set.update(sent["labels"])

    label_list = sorted(list(label_set))
    if "O" not in label_list:
        # A common practice is to place the "Outside" class first
        label_list = ["O"] + [l for l in label_list if l != "O"]
    
    # Load model and tokenizer
    tokenizer, _ = load_pubmedbert_model(cfg.model_name, label_list)
    model_init = lambda: load_pubmedbert_model(cfg.model_name, label_list)[1]
    
    print(f"Using device: {device}")

    if cfg.algorithm == "Centralized":
        all_train_sents = [s for client in client_train_sents for s in client]
        run_centralized_training(cfg, tokenizer, label_list, all_train_sents, test_sents, model_init, device)
    else:
        run_federated_training(cfg, tokenizer, label_list, client_train_sents, dev_sents, test_sents, model_init, device)

if __name__ == "__main__":
    main()