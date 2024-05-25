class EarlyStopper:
    def __init__(self,
                 patience: int,
                 mode: str = "min",
                 delta: float = 0,
                 threshold: float | None = None,
                 threshold_mode: str = "min"):
        """

        :param patience: how many epochs the training will go on without a change of more than :delta:
        in the metric, in direction dictated by :mode:
        :param mode: if "min", then will watch for decreases of more than :delta:, else if "max" will watch for increases.
        Any other value raises ValueError.
        :param delta: target minimum variation in direction dictated by :mode:
        :param threshold: threshold of metric, crossing it (in direction indicated by :threshold_mode:) triggres early stop
        :param threshold_mode: if "min", then triggers early stop when metrics goes below threshold, else if "max"
        triggers early stop when metrics goes above. Any other value raises ValueError.
        """
        if (mode not in ("min", "max")):
            raise ValueError(f"Invalid mode for patience: '{mode}', expected 'min' or 'max'.")
        if (threshold_mode not in ("min", "max")):
            raise ValueError(f"Invalid mode for threshold: '{threshold_mode}', expected 'min' or 'max'.")
        self.patience: int = patience
        self.mode: str = mode
        self.delta: float = delta
        self.threshold: float | None = threshold
        self.threshold_mode: str = threshold_mode

        self.patience_count = 0
        self.best_value = float("inf") if mode == "min" else -float("inf")
        self.threshold_trigger_state: bool = False

    def register_metric(self, value: float) -> None:
        if (self.mode == "min") and (value >= self.best_value - self.delta):
            self.patience_count += 1
        elif (self.mode == "max") and (value <= self.best_value + self.delta):
            self.patience_count += 1
        else:
            self.best_value = value
            self.patience_count = 0  # reset patience countdown

        if self.threshold is not None:  # should use threshold
            if (self.threshold_mode == "min" and value < self.threshold) or \
                    (self.threshold_mode == "max" and value > self.threshold):
                self.threshold_trigger_state = True

    def reset(self) -> None:
        self.patience_count = 0
        self.best_value = float("inf") if self.mode == "min" else -float("inf")
        self.threshold_trigger_state = False

    def should_early_stop(self) -> bool:
        return (self.patience_count >= self.patience) or self.threshold_trigger_state

    def log_if_stopped(self) -> None:
        if not self.should_early_stop():
            return
        if self.threshold_trigger_state:
            print(f"Early stop triggered: metric went "
                  f"{'below' if self.threshold_mode == 'min' else 'above'} "
                  f"threshold={self.threshold}.")
        else:
            print(f"Early stop triggered: metric went {self.patience} epochs "
                  f"without {'decreasing' if self.mode == 'min' else 'increasing'} "
                  f"more than {self.delta}.")
