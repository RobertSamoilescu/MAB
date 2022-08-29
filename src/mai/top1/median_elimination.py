import numpy as np
from typing import List, Callable


class MedianElimination:
    def __init__(self, arms: List[Callable], eps: float, delta: float):
        self.arms = arms
        self.eps = eps
        self.delta = delta

    def play(self):
        K = len(self.arms)
        S_i = range(K)
        eps_i = self.eps / 4
        delta_i = self.delta / 2

        while len(S_i) > 1:
            nsamples = int(np.ceil(4/eps_i**2 * np.log(3 / delta_i)))
            pa = np.array([np.mean([self.arms[a]() for _ in range(nsamples)]) for a in S_i])
            ml = np.median(pa)
            S_i = [a for i, a in enumerate(S_i) if pa[i] >= ml]
            eps_i = 3 / 4 * eps_i
            delta_i = delta_i / 2

        return S_i[0]