import numpy as np
from typing import Callable


class Bernoulli:
    def __init__(self, p: float):
        """
        Constructor.

        Parameters
        ----------
        p
            Mean of the Bernoulli distribution.
        """
        self.p = p

    def __call__(self) -> float:
        """
        Generates random reward from a Bernoulli(p).

        Returns
        -------
        Random sample from Bernoulli(p).
        """
        r = np.random.random()
        return 1. if r < self.p else 0


class Beta:
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def __call__(self) -> float:
        return np.random.beta(a=self.alpha, b=self.beta)


class Shap:
    def __init__(self,
                 feature: int,
                 X: np.ndarray,
                 baseline: np.ndarray,
                 predictor: Callable[[np.ndarray], np.ndarray],
                 min_val: float,
                 max_val: float):

        # set feature to compute shap values for
        self.feature = feature
        self.n_features = X.shape[-1]
        self.other_features = np.delete(np.arange(self.n_features), self.feature)

        # set input and baseline
        self.baseline = np.atleast_2d(baseline)
        self.X = np.atleast_2d(X)

        # set predictor
        self.predictor = predictor

        # set min and max value that the shap value can take
        self.min_val = min_val - max_val
        self.max_val = max_val - min_val

    def _generate_masks(self, n_samples: int):
        masks1 = np.zeros((n_samples, self.n_features))
        masks2 = np.zeros((n_samples, self.n_features))

        # number of features to turn on
        n_features_on = np.random.choice(self.n_features - 1, size=n_samples)

        for i in range(n_samples):
            indices = np.random.choice(self.other_features, size=n_features_on[i], replace=False)

            # update mask 1
            masks1[i, indices] = 1

            # update mask 2
            masks2[i, indices] = 1
            masks2[i, self.feature] = 1

        return masks1, masks2

    def _extend_mask(self, masks):
        masks = np.tile(masks, reps=(1, len(self.baseline)))
        return masks.reshape(-1, self.n_features)

    def __call__(self, n_samples):
        masks1, masks2 = self._generate_masks(n_samples)

        # extend masks to match the number of baselines
        masks1 = self._extend_mask(masks1)
        masks2 = self._extend_mask(masks2)

        # construct baseline
        baseline = np.tile(self.baseline, reps=(n_samples, 1))

        # construct instance 1
        instance1 = masks1 * self.X + (1 - masks1) * baseline

        # construct instance 2
        instance2 = masks2 * self.X + (1 - masks2) * baseline

        diff = self.predictor(instance2) - self.predictor(instance1)
        return np.clip((diff - self.min_val) / (self.max_val - self.min_val), 0, 1)
