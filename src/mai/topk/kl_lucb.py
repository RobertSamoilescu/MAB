import itertools
import numpy as np
from typing import List, Callable, Any, Dict
from abc import abstractmethod
from tqdm import tqdm



class KL_LUCB:
    def __init__(self, arms: List[Callable[[int], np.ndarray]], m: int, eps: float, delta: float, batch_size=1000, n_iter: int = 10):
        self.arms = arms
        self.K = len(self.arms)
        self.m = m
        self.eps = eps
        self.delta = delta
        self.batch_size = batch_size
        self.n_iter = n_iter

    def _KL_Bernoulli(self, p: np.ndarray, q: np.ndarray):
        # clip first distribution to avoid log or division by 0
        p = np.clip(p, 1e-5, 1 - 1e-5)
        p = p / np.linalg.norm(p)

        # clip second distribution to avoid log or division by 0
        q = np.clip(q, 1e-5, 1 - 1e-5)
        q = q / np.linalg.norm(q)

        # kl divergence between two Bernoulli distributions
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def _compute_beta(self, t: float):
        k1 = 405.5
        alpha = 1.1
        tmp = np.log(k1 * self.K * t**alpha / self.delta)
        return tmp + np.log(tmp)

    def _compute_upper_confidence(self, pa: np.ndarray, level: np.ndarray):
        start = pa
        end = np.minimum(1, pa + np.sqrt(level / 2))

        for _ in range(self.n_iter):
            middle = (start + end) / 2
            d = self._KL_Bernoulli(pa, middle)
            mask = d < level
            start[mask] = middle[mask]
            end[~mask] = middle[~mask]

        # return end to remain conservative
        return end

    def _compute_lower_confidence(self, pa: np.ndarray, level: np.ndarray):
        start = np.maximum(0, pa - np.sqrt(level / 2))
        end = pa

        for _ in range(self.n_iter):
            middle = (start + end) / 2
            d = self._KL_Bernoulli(pa, middle)
            mask = d < level
            end[mask] = middle[mask]
            start[~mask] = middle[~mask]

        # return start to remain conservative
        return start

    def _step(self, pa: np.ndarray, Na: np.ndarray, t: float) -> Dict[str, Any]:
        # compute upper and lower confidence bounds
        beta = self._compute_beta(t)
        level = beta / Na

        U = self._compute_upper_confidence(pa, level)
        L = self._compute_lower_confidence(pa, level)

        # select the set of m arms with the highest and lowest empirical averages
        partition = np.argpartition(pa, -self.m)
        high = partition[-self.m:]
        low = partition[:-self.m]

        # from the high arms, select the arm with the lowest confidence bound
        h = high[np.argmin(L[high])]
        # from the low arms, select the arm with the highest confidence bound
        l = low[np.argmax(U[low])]

        # compute difference between the upper bound of l and the lower bound of h
        return {
            'B': U[l] - L[h],
            'high': high,
            'low': low,
            'h': h,
            'l': l
        }

    def play(self):
        # average reward for each arm
        pa = np.array([np.mean(arm(self.batch_size)) for arm in tqdm(self.arms, desc='first iteration')])
        # number of times each arm has been sampled
        Na = np.full(pa.shape, self.batch_size)
        # time step
        t = self.batch_size

        res = self._step(pa, Na, t)
        for _ in tqdm(itertools.count()):
            if res['B'] <= self.eps:
                break

            # unpack data
            h, l = res['h'], res['l']

            # sample arm h and update statistics
            pa[h] = (pa[h] * Na[h] + self.arms[h](self.batch_size).sum()) / (Na[h] + self.batch_size)
            Na[h] += self.batch_size

            # sample arm l and update statistics
            pa[l] = (pa[l] * Na[l] + self.arms[l](self.batch_size).sum()) / (Na[l] + self.batch_size)
            Na[l] += self.batch_size

            # next step
            t += self.batch_size
            res = self._step(pa, Na, t)

        return res['high']



