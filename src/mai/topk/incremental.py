import numpy as np
from typing import List, Callable
from ..top1.median_elimination import MedianElimination


class Incremental:
    def __init__(self, arms: List[Callable], m: int, eps: float, delta: float):
        self.arms = arms
        self.n = len(self.arms)
        self.m = m
        self.eps = eps
        self.delta = delta

    def play(self) -> np.array:
        S, R = [], list(range(self.n))

        for l in range(self.m):
            median = MedianElimination(
                arms=[self.arms[i] for i in R],
                eps=self.eps,
                delta=self.delta / self.m
            )
            a = R[median.play()]
            S.append(a)
            R.remove(a)

        return np.array(list(S))