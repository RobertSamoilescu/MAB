import numpy as np
from typing import List, Callable

class KLUCB:
    def __init__(self, arms: List[Callable], c: float):
        self.arms = arms
        self.c = c

    def _divergence(self, p: float, q: float):
        p = np.clip(p, 1e-5, 1 - 1e-5)
        q = np.clip(q, 1e-5, 1 - 1e-5)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def _bisection(self, start: float, end: float, N_a: float, S_a: float, t: int, n_iter: int = 20):
        rhs = np.log(t) + self.c * np.log(np.log(t))

        for _ in range(n_iter):
            middle = (start + end) / 2
            kl_middle = self._divergence(S_a / N_a, middle)

            if N_a * kl_middle < rhs:
                start = middle
            else:
                end = middle

        return start

    def _upper_confidence_bound(self,
                                a: int,
                                N: np.ndarray,
                                S: np.ndarray,
                                t: int):
        p = S[a] / N[a]
        return self._bisection(start=p, end=1., N_a=N[a], S_a=S[a], t=t)

    def play(self, T: int) -> np.ndarray:
        K = len(self.arms)

        if K > T:
            raise ValueError("The number of arms is larger than the time budget.")

        # initialize average reward per arm and counter of number of pulls per arm
        N = np.ones(len(self.arms))
        S = np.array([arm() for arm in self.arms])

        rewards = S.tolist()

        for t in range(K + 1, T + 1):
            ucb = [self._upper_confidence_bound(a, N, S, t) for a in range(K)]
            a = np.argmax(ucb)
            R = self.arms[a]()
            N[a] += 1
            S[a] += R
            rewards.append(R)

        return np.cumsum(rewards)
