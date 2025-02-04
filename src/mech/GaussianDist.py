import secrets

import numpy as np
from numpy.random import MT19937, RandomState

from utils.utils import _ensure_2dim, DUMMY_CONSTANT, _ensure_np_array
from estimator.basic import _GeneralNaiveEstimator
from estimator.ptlr import _PTLREstimator

class GaussianDistSampler:
    """
        The sampler takes as inputs a pair of Gaussian Distribution (parameterized with mean vector and Covariance matrix).
        It preprocess and generate N data samples from each distribution. 
        It generate a sample set with given eta from the preprocessed dataset
    """

    def __init__(self, kwargs):
        self.P_mean = kwargs["dist"]["mean0"]
        self.P_mean = np.array(self.P_mean)
        self.P_cov = kwargs["dist"]["cov0"]
        self.P_cov = np.array(self.P_cov)
        
        self.Q_mean = kwargs["dist"]["mean1"]
        self.Q_mean = np.array(self.Q_mean)
        self.Q_cov = kwargs["dist"]["cov1"]
        self.Q_cov = np.array(self.Q_cov)
        
        assert (self.P_mean.size == self.Q_mean.size)
        self.dim = self.P_mean.size
        assert (self.P_cov.shape == (self.dim, self.dim))
        assert (self.Q_cov.shape == (self.dim, self.dim))
        
        self.bot = -DUMMY_CONSTANT
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def preprocess(self, num_samples):        
        samples_P = self.rng.multivariate_normal(self.P_mean, self.P_cov, size=num_samples)  
        samples_Q = self.rng.multivariate_normal(self.Q_mean, self.Q_cov, size=num_samples)  
        self.samples_P, self.samples_Q = _ensure_2dim(samples_P, samples_Q)
        self.computed_samples = num_samples
        
        return (self.samples_P, self.samples_Q)
    
    def gen_samples(self, eta, num_samples, reset = False, shuffle = False):
        assert eta > 0
        self.reset_randomness()
        
        if reset == True:
            samples_P, samples_Q = self.preprocess(num_samples)
        else:
            if (hasattr(self, "computed_samples") == False) or (self.computed_samples < num_samples):
                samples_P, samples_Q = self.preprocess(num_samples)
                self.computed_samples = num_samples
            else:
                samples_P = self.samples_P[:num_samples]
                samples_Q = self.samples_Q[:num_samples]
                
        if eta > 1:
            p = self.rng.uniform(0, 1, num_samples) > (1.0/eta)
            p = p.reshape((num_samples, 1)) * np.ones((num_samples, self.dim))
            samples_P = (1-p)*samples_P + p*(self.bot*np.ones_like(self.dim))
        if eta < 1:
            p = self.rng.uniform(0, 1, num_samples) > eta
            p = p.reshape((num_samples, 1)) * np.ones((num_samples, self.dim))
            samples_Q = (1-p)*samples_Q + p*(self.bot*np.ones_like(self.dim))
        
        samples = np.vstack((samples_P, samples_Q))
        labels = np.concatenate((np.zeros(num_samples), np.ones(num_samples)))
        
        if shuffle:
            ids = self.rng.permutation(num_samples*2)
            samples = samples[ids]
            labels = labels[ids]
            
        return {'X': samples, 'y': labels}
    
    
class GaussianDistEstimator(_GeneralNaiveEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.train_sampler = GaussianDistSampler(kwargs)
        self.test_sampler = GaussianDistSampler(kwargs)


class GaussianPTLREstimator(_PTLREstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sampler = GaussianDistSampler(kwargs)
        
        
        
def generate_params(num_samples = 10000,num_train_samples = 10000, num_test_samples = 1000, mean0 = [0], mean1 = [1], cov0 = [1], cov1 = [1], h=0.1):
    mean0 = _ensure_np_array(mean0)
    mean1 = _ensure_np_array(mean1)
    cov0 = _ensure_np_array(cov0)
    cov1 = _ensure_np_array(cov1)
    
    cov0, cov1 = _ensure_2dim(cov0, cov1)
    
    kwargs = {
        "h": h,
        "dist":{
            "mean0": mean0, "cov0": cov0,
            "mean1": mean1, "cov1": cov1
        },
        "num_samples" : num_samples,
        "num_train_samples" : num_train_samples,
        "num_test_samples" : num_test_samples
    }
    return kwargs