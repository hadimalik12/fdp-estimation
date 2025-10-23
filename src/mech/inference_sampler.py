"""
Inference-based sampler that reuses the original SubsamplingSampler structure
but replaces the Gaussian noise with model inference logits.
"""
import os
import sys
import numpy as np

project_dir = os.path.abspath(os.getcwd())
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from mech.Subsampling import SubsamplingSampler, _GeneralNaiveEstimator, _PTLREstimator


# Implements sampling based on model inference logits
class InferenceSampler(SubsamplingSampler):
    def __init__(self, kwargs):
        kwargs["dataset"]["x0"] = kwargs["dataset"]["x0_logits"]
        kwargs["dataset"]["x1"] = kwargs["dataset"]["x1_logits"]
        super().__init__(kwargs)
        self.x0 = kwargs["dataset"]["x0_logits"] #50,000 x 10
        self.x1 = kwargs["dataset"]["x1_logits"]
    
    def sample_from_logits(self, logits):
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        sampled = [np.random.choice(10, p=p) for p in probs]
        return np.array(sampled)

    # Draws and preprocesses samples from stored inference logits
    def preprocess(self, num_samples):
        idx0 = self.rng.choice(len(self.x0), size=num_samples, replace=True) #10000 indices [0, 1, 500]
        idx1 = self.rng.choice(len(self.x1), size=num_samples, replace=True) 
        s0 = self.x0[idx0] #10,000 x 10
        s1 = self.x1[idx1]
        if s0.ndim > 1:
            s0 = self.sample_from_logits(s0)
            s1 = self.sample_from_logits(s1)
        self.samples_P = s0.reshape(-1, 1)
        self.samples_Q = s1.reshape(-1, 1)
        self.computed_samples = num_samples
        return (self.samples_P, self.samples_Q)


# Defines estimator using InferenceSampler for train and test sampling
class InferenceEstimator(_GeneralNaiveEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.train_sampler = InferenceSampler(kwargs)
        self.test_sampler = InferenceSampler(kwargs)


# Defines PTLR estimator using inference-based sampling
class InferencePTLREstimator(_PTLREstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sampler = InferenceSampler(kwargs)


# Loads model logits and prepares parameters for inference estimators
def generate_params(
    num_samples=10000,
    num_train_samples=10000,
    num_test_samples=1000,
    m=5,
    sigma=1.0,
    h=0.1,
):
    x0_logits = np.load("outputs/logits_modelA_white.npy")
    x1_logits = np.load("outputs/logits_modelB_black.npy")
    kwargs = {
        "h": h,
        "dataset": {"x0_logits": x0_logits, "x1_logits": x1_logits},
        "ss_alg": {"m": m, "sigma": sigma},
        "num_samples": num_samples,
        "num_train_samples": num_train_samples,
        "num_test_samples": num_test_samples,
    }
    return kwargs