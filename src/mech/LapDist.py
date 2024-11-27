import secrets

import numpy as np
from numpy.random import MT19937, RandomState

from utils.utils import _ensure_2dim, DUMMY_CONSTANT
from estimator.basic import _GeneralNaiveEstimator

class LapDistSampler:
    """
        The sampler takes as inputs a pair of Laplace Distribution (parameterized with mean and scale).
        It preprocess and generate N data samples from each distribution. 
        It generate a sample set with given eta from the preprocessed dataset
    """

    def __init__(self, kwargs):
        self.dim = 1
        self.P_mean = kwargs["dist"]["mean0"]
        self.P_scale = kwargs["dist"]["scale0"]
        
        self.Q_mean = kwargs["dist"]["mean1"]
        self.Q_scale = kwargs["dist"]["scale1"]
        
        assert np.isscalar(self.P_mean) and np.isscalar(self.Q_mean) and np.isscalar(self.P_scale) and np.isscalar(self.Q_scale) 
        
        self.bot = -DUMMY_CONSTANT
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def preprocess(self, num_samples):        
        samples_P = self.rng.laplace(self.P_mean, self.P_scale, size=num_samples)  
        samples_Q = self.rng.laplace(self.Q_mean, self.Q_scale, size=num_samples)  
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
    

class LapDistEstimator(_GeneralNaiveEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.train_sampler = LapDistSampler(kwargs)
        self.test_sampler = LapDistSampler(kwargs)
    
    
def generate_params(num_train_samples = 10000, num_test_samples = 1000, mean0 = 0, mean1 = 1, scale0 = 1, scale1 = 1):    
    kwargs = {
        "dist":{
            "mean0": mean0, "scale0": scale0,
            "mean1": mean1, "scale1": scale1
        },
        "num_train_samples" : num_train_samples,
        "num_test_samples" : num_test_samples
    }
    return kwargs