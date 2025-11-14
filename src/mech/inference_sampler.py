"""
Inference-based sampler for FDP estimation.
Samples from stored model logits and performs naive distinguishability estimation.
"""

import os
import sys
import numpy as np
from mech.Subsampling import SubsamplingSampler, _GeneralNaiveEstimator, _PTLREstimator

# Sample based on model inference logits
class InferenceSampler(SubsamplingSampler):
    def __init__(self, kwargs):
        kwargs["dataset"]["x0"] = kwargs["dataset"]["x0_logits"]
        kwargs["dataset"]["x1"] = kwargs["dataset"]["x1_logits"]
        super().__init__(kwargs)

        self.x0 = kwargs["dataset"]["x0_logits"]  # shape (N, 10)
        self.x1 = kwargs["dataset"]["x1_logits"]

    # softmax for a single vector
    def softmax(self, logits):
        logits = logits - np.max(logits)
        expv = np.exp(logits)
        return expv / np.sum(expv)

    # sample ONE class for each logit vector
    def sample_one_per_logit(self, logits_array):
        N = logits_array.shape[0]
        out = np.zeros(N, dtype=int)

        for i in range(N):
            probs = self.softmax(logits_array[i])
            out[i] = np.random.choice(10, p=probs)

        return out

    def preprocess(self, num_samples):
        # sample one class per row
        s0 = self.sample_one_per_logit(self.x0)
        s1 = self.sample_one_per_logit(self.x1)

        # If more samples needed, repeat sampling
        if num_samples > len(s0):
            reps = int(np.ceil(num_samples / len(s0)))
            s0 = np.tile(s0, reps)[:num_samples]
            s1 = np.tile(s1, reps)[:num_samples]
        else:
            s0 = s0[:num_samples]
            s1 = s1[:num_samples]

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

# PTLR estimator using the same sampler
class InferencePTLREstimator(_PTLREstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sampler = InferenceSampler(kwargs)


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