import numpy as np
from typing import List, Callable
from tqdm import tqdm


class Direct:
    def __init__(self, arms: List[Callable[[int], np.ndarray]], m: int, eps: float, delta: float):
        self.arms = arms
        self.n = len(self.arms)
        self.m = m
        self.eps = eps
        self.delta = delta

    def play(self) -> np.array:
        n_samples = int(np.ceil(2 / self.eps**2 * np.log(self.n / self.delta)))
        pa = np.array([np.mean([arm(n_samples)]) for arm in tqdm(self.arms)])
        return np.argpartition(pa, -self.m)[-self.m:]