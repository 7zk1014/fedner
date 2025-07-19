from statistics import mean, pstdev
from typing import List, Dict

from .evaluate import evaluate_model


def evaluate_global_on_local(model, tokenizer, clients_dev_sets: List[List[Dict]], label_list: List[str]):
    """Evaluate global model on each client's validation set.

    Parameters
    ----------
    model : PreTrainedModel
        The global model to evaluate.
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the model.
    clients_dev_sets : List[List[Dict]]
        Validation data for each client. Each element is a list of examples
        where every example is a ``{"tokens": [...], "labels": [...]}`` dict.
    label_list : List[str]
        All possible label strings used for evaluation.

    Returns
    -------
    Dict
        A dictionary containing per-client metrics as well as the mean and
        standard deviation of the F1 scores across clients.
    """
    client_metrics = []
    for dev_data in clients_dev_sets:
        metrics = evaluate_model(model, tokenizer, dev_data, label_list)
        client_metrics.append(metrics)

    if client_metrics:
        f1_scores = [m["f1"] for m in client_metrics]
        mean_f1 = mean(f1_scores)
        std_f1 = pstdev(f1_scores) if len(f1_scores) > 1 else 0.0
    else:
        mean_f1 = 0.0
        std_f1 = 0.0

    return {
        "client_metrics": client_metrics,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
    }
