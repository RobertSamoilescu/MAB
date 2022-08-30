import itertools
import numpy as np
from typing import List, Callable, Any, Dict
from abc import abstractmethod
from tqdm import tqdm



class LUCB:
    def __init__(self, arms: List[Callable[[int], np.ndarray]], m: int, eps: float, delta: float, batch_size=1000):
        self.arms = arms
        self.n = len(self.arms)
        self.m = m
        self.eps = eps
        self.delta = delta
        self.batch_size = batch_size

    def confidence_bound(self, u: np.ndarray, t: float):
        k1 = 5 / 4
        return np.sqrt(1/(2 * u) * np.log(k1 * self.n * t**4 / self.delta))

    def _step(self, mu: np.ndarray, u: np.ndarray, t: float) -> Dict[str, Any]:
        # compute upper and lower confidence bounds
        beta = self.confidence_bound(u, t)

        # select the set of m arms with the highest and lowest empirical averages
        partition = np.argpartition(mu, -self.m)
        high = partition[-self.m:]
        low = partition[:-self.m]

        # from the high arms, select the arm with the lowest confidence bound
        h = high[np.argmin(mu[high] - beta[high])]

        # from the low arms, select the arm with the highest confidence bound
        l = low[np.argmax(mu[low] - beta[low])]

        # compute difference between the upper bound of l and the lower bound of h
        return {
            'B': (mu[l] + beta[l]) - (mu[h] - beta[h]),
            'high': high,
            'low': low,
            'h': h,
            'l': l
        }

    def play(self):
        mu = np.array([np.mean(arm(self.batch_size)) for arm in self.arms])  # average reward for each arm
        u = np.ones_like(mu)  # number of times each arm has been sampled
        t = self.batch_size

        res = self._step(mu, u, t)
        for _ in tqdm(itertools.count()):
            if res['B'] <= self.eps:
                break

            # unpack data
            h, l = res['h'], res['l']

            # sample arm h and update statistics
            mu[h] = (mu[h] * u[h] + self.arms[h](self.batch_size).sum()) / (u[h] + self.batch_size)
            u[h] += self.batch_size

            # sample arm l and update statistics
            mu[l] = (mu[l] * u[l] + self.arms[l](self.batch_size).sum()) / (u[l] + self.batch_size)
            u[l] += self.batch_size

            # next step
            t += self.batch_size
            res = self._step(mu, u, t)

        return res['high']



