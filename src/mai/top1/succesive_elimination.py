import numpy as np
from typing import Callable, List

class SuccesiveElimination:
    def __init__(self, arms: List[Callable], eps: float, delta: float, c: float):
        self.arms = arms
        self.eps = eps
        self.delta = delta
        self.c = c


    def play(self):
        t = 1
        n = len(self.arms)
        S = np.arange(n)
        pa = np.zeros(n)
        N = np.zeros(n)

        while True:
            samples = [self.arms[i]() for i in S]
            for i, idx in enumerate(S):
                pa[idx] = (pa[idx] * N[idx] + samples[i]) / (N[idx] + 1)
                N[idx] += 1

            pmax = np.max(pa[list(S)])
            alpha = np.sqrt(np.log(self.c * n * t**2 / self.delta) / t)
            S = np.array([a for a in S if pmax - pa[a] < 2*alpha])
            t += 1

            k = len(S)
            limit = int(np.ceil(np.log2(k/self.delta) / self.eps**2))
            if len(S) == 1 or np.all(N[list(S)] > limit):
                break

        return S[np.argmax(pa[S])]
