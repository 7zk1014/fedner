"""
Unified training utility functions
All algorithms share these functions to ensure fair comparison
"""

import random
import torch
from collections import Counter
from typing import List, Dict, Optional, Any


def freeze_model_layers(model, train_last_n_layers: int = 4, freeze_embeddings: bool = True):
    """
    Unified layer freezing function, all algorithms use the same freezing strategy
    
    Args:
        model: BERT-like model
        train_last_n_layers: Train last N layers
            -1: Train all layers
            0: Train only classifier head
            1-12: Train last N transformer layers
        freeze_embeddings: Whether to freeze embedding layers
    """
    # If -1, train all layers
    if train_last_n_layers == -1:
        for param in model.parameters():
            param.requires_grad = True
        print("[Layer Freezing] Training all layers")
        return
    
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Get model configuration
    if hasattr(model, 'bert'):
        bert_model = model.bert
        classifier = model.classifier if hasattr(model, 'classifier') else None
        num_hidden_layers = bert_model.config.num_hidden_layers
    elif hasattr(model, 'distilbert'):
        bert_model = model.distilbert
        classifier = model.classifier if hasattr(model, 'classifier') else None
        num_hidden_layers = bert_model.config.n_layers
    else:
        # Other architectures, default unfreeze all
        for param in model.parameters():
            param.requires_grad = True
        return
    
    # Always unfreeze classification head
    if classifier is not None:
        for param in classifier.parameters():
            param.requires_grad = True
    
    # If only training classification head
    if train_last_n_layers == 0:
        print("[Layer Freezing] Training only classifier head")
        return
    
    # Unfreeze last N transformer layers
    if train_last_n_layers > 0:
        # Calculate layer indices to unfreeze
        layers_to_unfreeze = list(range(max(0, num_hidden_layers - train_last_n_layers), num_hidden_layers))
        
        # Unfreeze specified encoder layers
        for name, param in model.named_parameters():
            # BERT architecture
            if "encoder.layer." in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num in layers_to_unfreeze:
                    param.requires_grad = True
            # DistilBERT architecture
            elif "transformer.layer." in name:
                layer_num = int(name.split("transformer.layer.")[1].split(".")[0])
                if layer_num in layers_to_unfreeze:
                    param.requires_grad = True
        
        # Handle embedding layers
        if not freeze_embeddings:
            if hasattr(bert_model, 'embeddings'):
                for param in bert_model.embeddings.parameters():
                    param.requires_grad = True
        
        print(f"[Layer Freezing] Training last {train_last_n_layers} transformer layers "
              f"(layers {layers_to_unfreeze[0]}-{layers_to_unfreeze[-1]})")
        if not freeze_embeddings:
            print("[Layer Freezing] Also training embedding layers")
    
    # Print trainable parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Layer Freezing] Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")


def sample_client_data(
    data: List[Dict],
    sample_size: Optional[int] = None,
    strategy: str = "random",
    with_replacement: bool = False,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Unified data sampling function
    
    Args:
        data: Client's raw data
        sample_size: Sample size, None means use all data
        strategy: Sampling strategy
            - "random": Random sampling
            - "balanced": Balanced sampling by labels
            - "stratified": Stratified sampling (maintain label proportions)
        with_replacement: Whether to use replacement sampling
        seed: Random seed
    
    Returns:
        Sampled data
    """
    if sample_size is None or sample_size >= len(data):
        return data
    
    if seed is not None:
        random.seed(seed)
    
    if strategy == "random":
        # Random sampling
        if with_replacement:
            sampled = random.choices(data, k=sample_size)
        else:
            sampled = random.sample(data, min(sample_size, len(data)))
    
    elif strategy == "balanced":
        # Balanced sampling: sample equal amounts for each label
        label_groups = {}
        for item in data:
            # Get the main label of this sample (first non-O label)
            main_label = "O"
            for label in item.get("labels", []):
                if label != "O":
                    main_label = label.replace("B-", "").replace("I-", "")
                    break
            
            if main_label not in label_groups:
                label_groups[main_label] = []
            label_groups[main_label].append(item)
        
        # Sample from each label group
        sampled = []
        samples_per_label = max(1, sample_size // len(label_groups))
        remaining = sample_size
        
        for label, items in label_groups.items():
            n = min(samples_per_label, len(items), remaining)
            if with_replacement:
                sampled.extend(random.choices(items, k=n))
            else:
                sampled.extend(random.sample(items, n))
            remaining -= n
        
        # If there's remaining quota, randomly supplement
        if remaining > 0:
            all_unsampled = [item for item in data if item not in sampled]
            if all_unsampled:
                extra = random.sample(all_unsampled, min(remaining, len(all_unsampled)))
                sampled.extend(extra)
    
    elif strategy == "stratified":
        # Stratified sampling: maintain original label distribution proportions
        label_counts = Counter()
        label_groups = {}
        
        # Count label distribution
        for item in data:
            main_label = "O"
            for label in item.get("labels", []):
                if label != "O":
                    main_label = label.replace("B-", "").replace("I-", "")
                    break
            
            label_counts[main_label] += 1
            if main_label not in label_groups:
                label_groups[main_label] = []
            label_groups[main_label].append(item)
        
        # Sample by proportion
        sampled = []
        for label, items in label_groups.items():
            proportion = label_counts[label] / len(data)
            n = max(1, int(sample_size * proportion))
            n = min(n, len(items))
            
            if with_replacement:
                sampled.extend(random.choices(items, k=n))
            else:
                sampled.extend(random.sample(items, n))
        
        # Adjust to exact sample size
        if len(sampled) > sample_size:
            sampled = random.sample(sampled, sample_size)
        elif len(sampled) < sample_size:
            # Fill remaining quota with additional samples
            all_unsampled = [item for item in data if item not in sampled]
            if all_unsampled:
                extra_needed = sample_size - len(sampled)
                extra = random.sample(all_unsampled, min(extra_needed, len(all_unsampled)))
                sampled.extend(extra)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # Note: Order shuffling is commented out to maintain reproducibility
    
    return sampled


def select_clients_for_round(
    num_clients: int,
    fraction: float = 1.0,
    min_clients: int = 1,
    round_idx: int = 0,
    seed: Optional[int] = None
) -> List[int]:
    """
    Select clients participating in this training round
    
    Args:
        num_clients: Total number of clients
        fraction: Participation ratio (0, 1]
        min_clients: Minimum number of participating clients
        round_idx: Current round index (used for random seed)
        seed: Base random seed
    
    Returns:
        List of participating client indices
    """
    # Calculate number of clients participating in this round
    num_selected = max(min_clients, int(num_clients * fraction))
    num_selected = min(num_selected, num_clients)
    
    # Set random seed (different for each round)
    if seed is not None:
        random.seed(seed + round_idx)
    
    # Randomly select clients
    selected_clients = random.sample(range(num_clients), num_selected)
    selected_clients.sort()
    
    return selected_clients


def print_data_statistics(data: List[Dict], name: str = "Dataset"):
    """
    Print data statistics
    
    Args:
        data: Data list
        name: Dataset name
    """
    if not data:
        print(f"[{name}] No data")
        return
    
    # Count label distribution
    label_counts = Counter()
    total_tokens = 0
    
    for item in data:
        labels = item.get("labels", [])
        tokens = item.get("tokens", [])
        total_tokens += len(tokens)
        
        for label in labels:
            if label != "O":
                # Extract base label (remove B-/I- prefix)
                base_label = label.replace("B-", "").replace("I-", "")
                label_counts[base_label] += 1
    
    print(f"\n[{name} Statistics]")
    print(f"  Total samples: {len(data)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens per sample: {total_tokens/len(data):.1f}")
    print(f"  Unique entity types: {len(label_counts)}")
    
    if label_counts:
        print(f"  Top 5 entity types:")
        for label, count in label_counts.most_common(5):
            print(f"    - {label}: {count}")


def get_model_size_mb(model) -> float:
    """
    Calculate model size (MB)
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size (MB)
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size = param_size + buffer_size
    size_mb = total_size / (1024 ** 2)
    
    return size_mb


def get_trainable_params_info(model) -> Dict[str, Any]:
    """
    Get model trainable parameter information
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary containing parameter statistics
    """
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "total_size_mb": total_params * 4 / (1024 ** 2),  # Assume float32
        "trainable_size_mb": trainable_params * 4 / (1024 ** 2)
    }