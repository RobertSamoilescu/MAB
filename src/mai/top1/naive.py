import numpy as np
from typing import Callable, List


class Naive:
    def __init__(self, arms: List[Callable], eps: float, delta: float):
        self.arms = arms
        self.eps = eps
        self.delta = delta

    def play(self) -> int:
        K = len(self.arms)
        n_pulls = int(np.ceil(4 / (self.eps**2) * np.log(2 * K / self.delta)))
        mu = [np.mean([arm() for _ in range(n_pulls)]) for arm in self.arms]
        return np.argmax(mu).item()