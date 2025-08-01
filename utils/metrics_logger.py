import time
from collections import defaultdict

class MetricsLogger:
    def __init__(self, cfg=None):
        """Create a new logger.

        Parameters
        ----------
        cfg : object, optional
            Experiment configuration. When provided it is stored on the logger
            for later use when saving metrics.
        """
        self.history = defaultdict(list)
        self.start_time = None
        self.cfg = cfg

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        return time.time() - self.start_time

    def log_round_metrics(self, round_idx, metrics, elapsed_time):
        self.history["round"].append(round_idx)
        self.history["f1"].append(metrics.get("f1"))
        self.history["precision"].append(metrics.get("precision"))
        self.history["recall"].append(metrics.get("recall"))
        self.history["accuracy"].append(metrics.get("accuracy"))
        self.history["time"].append(elapsed_time)

    def get_logs(self):
        return dict(self.history)
