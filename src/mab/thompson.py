import numpy as np
from typing import List, Callable

class ThompsonBernoulli:
    def __init__(self, arms: List[Callable]):
        self.arms = arms


    def play(self, T: int):
        K = len(self.arms)

        if K > T:
            raise ValueError("The number of arms is larger than the time budget.")

        # initialize buffers for numbers of 1 & 0
        S = np.zeros(K)
        F = np.zeros(K)

        # rewards buffer
        rewards = []

        for t in range(T):
            theta = np.random.beta(S + 1, F + 1)
            i = np.argmax(theta)

            R = self.arms[i]()
            rewards.append(R)

            if R == 1:
                S[i] += 1
            else:
                F[i] += 1

        return np.cumsum(rewards)