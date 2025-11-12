"""
Inference-based sampler for FDP estimation.
Samples from stored model logits and performs naive distinguishability estimation.
"""

import os
import sys
import numpy as np
from mech.Subsampling import SubsamplingSampler, _GeneralNaiveEstimator


# Sample based on model inference logits
class InferenceSampler(SubsamplingSampler):
    def __init__(self, kwargs):
        kwargs["dataset"]["x0"] = kwargs["dataset"]["x0_logits"]
        kwargs["dataset"]["x1"] = kwargs["dataset"]["x1_logits"]
        super().__init__(kwargs)
        self.x0 = kwargs["dataset"]["x0_logits"]
        self.x1 = kwargs["dataset"]["x1_logits"]

    # Sample classes from a single logit vector
    def sample_from_logits(self, logits, num_samples):
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.random.choice(len(probs), size=num_samples, p=probs)

    # Generate samples for both models
    def preprocess(self, num_samples):
        s0_logits = self.x0.squeeze()
        s1_logits = self.x1.squeeze()
        s0 = self.sample_from_logits(s0_logits, num_samples)
        s1 = self.sample_from_logits(s1_logits, num_samples)
        self.samples_P = s0.reshape(-1, 1)
        self.samples_Q = s1.reshape(-1, 1)
        self.computed_samples = num_samples
        return self.samples_P, self.samples_Q


# Naive estimator (kNN classifier)
class InferenceEstimator(_GeneralNaiveEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.train_sampler = InferenceSampler(kwargs)
        self.test_sampler = InferenceSampler(kwargs)


# Parameter generator for inference
def generate_params(logit_path_a, logit_path_b, num_samples=10000, num_train_samples=8000, num_test_samples=2000, m=5, sigma=1.0, h=0.1):
    x0_logits = np.load(logit_path_a)
    x1_logits = np.load(logit_path_b)
    return {
        "h": h,
        "dataset": {"x0_logits": x0_logits, "x1_logits": x1_logits},
        "ss_alg": {"m": m, "sigma": sigma},
        "num_samples": num_samples,
        "num_train_samples": num_train_samples,
        "num_test_samples": num_test_samples,
    }


# Test run
if __name__ == "__main__":
    kwargs = generate_params("outputs/exp1_baseline/logits_Mw.npy", "outputs/exp1_baseline/logits_Mb.npy")
    sampler = InferenceSampler(kwargs)
    sP, sQ = sampler.preprocess(num_samples=10000)
    print("Model A samples:", sP[:10].flatten())
    print("Model B samples:", sQ[:10].flatten())