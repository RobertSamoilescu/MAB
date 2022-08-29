import numpy as np
from typing import List, Callable
from tqdm import tqdm


class Halving:
    def __init__(self, arms: List[Callable[[int], np.ndarray]], m: int, eps: float, delta: float):
        self.arms = arms
        self.n = len(self.arms)
        self.m = m
        self.eps = eps
        self.delta = delta

    def play(self):
        R_i = np.arange(self.n)
        eps_i = self.eps / 4
        delta_i = self.delta / 2
        n_iter = int(np.ceil(np.log2(self.n / self.m)))

        for _ in tqdm(range(n_iter)):
            n_samples = int(np.ceil(2 / self.eps**2 * np.log(3 * self.m / delta_i)))
            pa = np.array([np.mean([self.arms[i](n_samples)]) for i in R_i])
            n_arms = int(max(np.ceil(len(R_i) / 2), self.m))
            indexes = np.argpartition(pa, -n_arms)[-n_arms:]
            R_i = R_i[indexes]
            eps_i = 3/4 * eps_i
            delta_i = 1/2 * delta_i

        return R_i