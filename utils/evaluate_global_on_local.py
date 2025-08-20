from statistics import mean, pstdev
from typing import List, Dict

from .evaluate import evaluate_model
from utils.metrics_logger import MetricsLogger

def evaluate_global_on_local(
    model,
    tokenizer,
    clients_dev_sets: List[List[Dict]],
    label_list: List[str],
    logger: MetricsLogger = None,
    round_num: int = None
):
    """
    Evaluate global model on each client's local validation data.
    Optionally logs mean/std F1 to MetricsLogger.

    Parameters
    ----------
    model : PreTrainedModel
        The global model to evaluate.
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the model.
    clients_dev_sets : List[List[Dict]]
        Dev data for each client (a list of sentence dicts).
    label_list : List[str]
        All possible label strings used for evaluation.
    logger : MetricsLogger, optional
        If provided, logs "mean_f1" and "std_f1" for the current round.
    round_num : int, optional
        Current communication round number (for logging).

    Returns
    -------
    Dict
        {
            "client_metrics": list of metrics per client,
            "mean_f1": float,
            "std_f1": float
        }
    """
    client_metrics = []
    for i, dev_data in enumerate(clients_dev_sets):
        metrics = evaluate_model(model, tokenizer, dev_data, label_list)
        metrics["client_id"] = i
        client_metrics.append(metrics)

    if client_metrics:
        f1_scores = [m["f1"] for m in client_metrics]
        mean_f1 = mean(f1_scores)
        std_f1 = pstdev(f1_scores) if len(f1_scores) > 1 else 0.0
    else:
        mean_f1 = 0.0
        std_f1 = 0.0

    if logger is not None and round_num is not None:
        logger.log(round_num, "mean_f1", mean_f1)
        logger.log(round_num, "std_f1", std_f1)

    return {
        "client_metrics": client_metrics,
        "mean_f1": mean_f1,
        "std_f1": std_f1
    }
