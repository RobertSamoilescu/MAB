import numpy as np
from typing import List, Callable

class EpsGreedy:
    def __init__(self, arms: List[Callable], eps: float = 0.1):
        self.arms = arms
        self.eps = eps

    def play(self, T: int) -> np.ndarray:
        K = len(self.arms)

        if K > T:
            raise ValueError("The number of arms is larger than the time budget.")

        # initialize average reward per arm and counter of number of pulls per arm
        mu = np.zeros(len(self.arms))
        N = np.zeros_like(mu)

        # define rewards buffer
        rewards = []

        for t in range(T):
            rand = np.random.random()
            i = np.argmax(mu) if rand < 1 - self.eps else np.random.choice(K)
            R = self.arms[i]()
            mu[i] = (mu[i] * N[i] + R) / (N[i] + 1)
            N[i] = N[i] + 1
            rewards.append(R)

        return np.cumsum(rewards)