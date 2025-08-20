"""
统一的训练工具函数
所有算法共享这些函数来保证公平对比
"""

import random
import torch
from collections import Counter
from typing import List, Dict, Optional, Any


def freeze_model_layers(model, train_last_n_layers: int = 4, freeze_embeddings: bool = True):
    """
    统一的层冻结函数，所有算法使用相同的冻结策略
    
    Args:
        model: BERT-like模型
        train_last_n_layers: 训练最后N层
            -1: 训练所有层
            0: 只训练分类头
            1-12: 训练最后N个transformer层
        freeze_embeddings: 是否冻结embedding层
    """
    # 如果是 -1，训练所有层
    if train_last_n_layers == -1:
        for param in model.parameters():
            param.requires_grad = True
        print("[Layer Freezing] Training all layers")
        return
    
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 获取模型配置
    if hasattr(model, 'bert'):
        bert_model = model.bert
        classifier = model.classifier if hasattr(model, 'classifier') else None
        num_hidden_layers = bert_model.config.num_hidden_layers
    elif hasattr(model, 'distilbert'):
        bert_model = model.distilbert
        classifier = model.classifier if hasattr(model, 'classifier') else None
        num_hidden_layers = bert_model.config.n_layers
    else:
        # 其他架构，默认解冻所有
        for param in model.parameters():
            param.requires_grad = True
        return
    
    # 总是解冻分类头
    if classifier is not None:
        for param in classifier.parameters():
            param.requires_grad = True
    
    # 如果只训练分类头
    if train_last_n_layers == 0:
        print("[Layer Freezing] Training only classifier head")
        return
    
    # 解冻最后N个transformer层
    if train_last_n_layers > 0:
        # 计算需要解冻的层索引
        layers_to_unfreeze = list(range(max(0, num_hidden_layers - train_last_n_layers), num_hidden_layers))
        
        # 解冻指定的encoder层
        for name, param in model.named_parameters():
            # BERT架构
            if "encoder.layer." in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num in layers_to_unfreeze:
                    param.requires_grad = True
            # DistilBERT架构
            elif "transformer.layer." in name:
                layer_num = int(name.split("transformer.layer.")[1].split(".")[0])
                if layer_num in layers_to_unfreeze:
                    param.requires_grad = True
        
        # 处理embedding层
        if not freeze_embeddings:
            if hasattr(bert_model, 'embeddings'):
                for param in bert_model.embeddings.parameters():
                    param.requires_grad = True
        
        print(f"[Layer Freezing] Training last {train_last_n_layers} transformer layers "
              f"(layers {layers_to_unfreeze[0]}-{layers_to_unfreeze[-1]})")
        if not freeze_embeddings:
            print("[Layer Freezing] Also training embedding layers")
    
    # 打印可训练参数统计
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
    统一的数据采样函数
    
    Args:
        data: 客户端的原始数据
        sample_size: 采样数量，None表示使用全部数据
        strategy: 采样策略
            - "random": 随机采样
            - "balanced": 按标签均衡采样
            - "stratified": 分层采样（保持标签比例）
        with_replacement: 是否有放回采样
        seed: 随机种子
    
    Returns:
        采样后的数据
    """
    if sample_size is None or sample_size >= len(data):
        return data
    
    if seed is not None:
        random.seed(seed)
    
    if strategy == "random":
        # 随机采样
        if with_replacement:
            sampled = random.choices(data, k=sample_size)
        else:
            sampled = random.sample(data, min(sample_size, len(data)))
    
    elif strategy == "balanced":
        # 均衡采样：每个标签采样相同数量
        label_groups = {}
        for item in data:
            # 获取该样本的主要标签（第一个非O标签）
            main_label = "O"
            for label in item.get("labels", []):
                if label != "O":
                    main_label = label.replace("B-", "").replace("I-", "")
                    break
            
            if main_label not in label_groups:
                label_groups[main_label] = []
            label_groups[main_label].append(item)
        
        # 每个标签组采样
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
        
        # 如果还有剩余配额，随机补充
        if remaining > 0:
            all_unsampled = [item for item in data if item not in sampled]
            if all_unsampled:
                extra = random.sample(all_unsampled, min(remaining, len(all_unsampled)))
                sampled.extend(extra)
    
    elif strategy == "stratified":
        # 分层采样：保持原始标签分布比例
        label_counts = Counter()
        label_groups = {}
        
        # 统计标签分布
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
        
        # 按比例采样
        sampled = []
        for label, items in label_groups.items():
            proportion = label_counts[label] / len(data)
            n = max(1, int(sample_size * proportion))
            n = min(n, len(items))
            
            if with_replacement:
                sampled.extend(random.choices(items, k=n))
            else:
                sampled.extend(random.sample(items, n))
        
        # 调整到精确的采样数量
        if len(sampled) > sample_size:
            sampled = random.sample(sampled, sample_size)
        elif len(sampled) < sample_size:
            # 补充采样
            all_unsampled = [item for item in data if item not in sampled]
            if all_unsampled:
                extra_needed = sample_size - len(sampled)
                extra = random.sample(all_unsampled, min(extra_needed, len(all_unsampled)))
                sampled.extend(extra)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # 打乱顺序
    # random.shuffle(sampled)
    
    return sampled


def select_clients_for_round(
    num_clients: int,
    fraction: float = 1.0,
    min_clients: int = 1,
    round_idx: int = 0,
    seed: Optional[int] = None
) -> List[int]:
    """
    选择本轮参与训练的客户端
    
    Args:
        num_clients: 总客户端数量
        fraction: 参与比例 (0, 1]
        min_clients: 最少参与客户端数
        round_idx: 当前轮次（用于随机种子）
        seed: 基础随机种子
    
    Returns:
        参与客户端的索引列表
    """
    # 计算本轮参与的客户端数量
    num_selected = max(min_clients, int(num_clients * fraction))
    num_selected = min(num_selected, num_clients)
    
    # 设置随机种子（每轮不同）
    if seed is not None:
        random.seed(seed + round_idx)
    
    # 随机选择客户端
    selected_clients = random.sample(range(num_clients), num_selected)
    selected_clients.sort()
    
    return selected_clients


def print_data_statistics(data: List[Dict], name: str = "Dataset"):
    """
    打印数据统计信息
    
    Args:
        data: 数据列表
        name: 数据集名称
    """
    if not data:
        print(f"[{name}] No data")
        return
    
    # 统计标签分布
    label_counts = Counter()
    total_tokens = 0
    
    for item in data:
        labels = item.get("labels", [])
        tokens = item.get("tokens", [])
        total_tokens += len(tokens)
        
        for label in labels:
            if label != "O":
                # 提取基础标签（去除B-/I-前缀）
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
    计算模型大小（MB）
    
    Args:
        model: PyTorch模型
    
    Returns:
        模型大小（MB）
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
    获取模型可训练参数信息
    
    Args:
        model: PyTorch模型
    
    Returns:
        包含参数统计的字典
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
        "total_size_mb": total_params * 4 / (1024 ** 2),  # 假设float32
        "trainable_size_mb": trainable_params * 4 / (1024 ** 2)
    }