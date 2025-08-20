import copy
import torch
from typing import List


def average_weights(models: List[torch.nn.Module]) -> torch.nn.Module:
    """Compute the element-wise average of model weights.

    Args:
        models: List of client models with identical architectures.

    Returns:
        A new model instance with averaged parameters.
    """
    if not models:
        raise ValueError("No models provided for averaging")

    # Initialize new model as deep copy of first model
    new_model = copy.deepcopy(models[0])
    state_dicts = [m.state_dict() for m in models]

    # Iterate over parameter tensors and compute mean across clients
    for name in new_model.state_dict().keys():
        stacked = torch.stack([sd[name] for sd in state_dicts], dim=0)
        mean_val = torch.mean(stacked, dim=0)
        new_model.state_dict()[name].copy_(mean_val)

    return new_model
