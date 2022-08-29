import numpy as np
from typing import List, Callable

class UCB():
    def __init__(self, arms: List[Callable], c: float = 1.):
        """
        Constructor.
        
        Parameters
        ----------
        arms
            List of callable arms.
        c
            Exploration parameter.
        """
        self.arms = arms
        self.c = c

    def play(self, T: int) -> List[float]:
        """
        Play to maximize reward.

        Parameters
        ----------
        T
            Time budget.

        Returns
        -------
        Cumulated reward for each round.
        """
        K = len(self.arms)

        if K > T:
            raise ValueError("The number of arms is larger than the time budget.")

        # initialize average reward per arm and counter of number of pulls per arm
        mu = np.array([arm() for arm in self.arms])
        N = np.ones_like(mu)

        # define rewards buffer
        rewards = mu.tolist()

        for t in range(K + 1, T + 1):
            # compute confidence upper bound
            ub = mu + self.c * np.sqrt(np.log(t) / N)

            # select best arm and play it
            i = np.argmax(ub)
            R = self.arms[i]()
            rewards.append(R)

            # update mean reward
            mu[i] = (mu[i] * N[i] + R) / (N[i] + 1)
            N[i] += 1

        return np.cumsum(rewards)
