import secrets

import numpy as np
from numpy.random import MT19937, RandomState

from utils.utils import _ensure_2dim, DUMMY_CONSTANT
from estimator.basic import _GeneralNaiveEstimator
from estimator.ptlr import _PTLREstimator
from auditor.basic import _GeneralNaiveAuditor
from analysis.tradeoff_toyDPSGD import toyDPSGD_compute_tradeoff_curve
from functools import partial

class toy_DPSGDSampler:
    """
        The sampler takes as inputs a pair of database (x0, x1)
        The parameters of the SGD algorithm: initial model theta_0; number of iterations T; size of the subset 
    database that is used in each iteration; learning rate eta; and the noise parameter sigma
        It preprocess and generate N data samples from both distribution SGD(x0) and SGD(x1). 
        It generate a sample set with given eta from the preprocessed dataset
    """

    def __init__(self, kwargs):
        self.dim = 1
        self.x0 = kwargs["dataset"]["x0"]
        self.x1 = kwargs["dataset"]["x1"]
        
        self.theta_0 = kwargs["sgd_alg"]["theta_0"]
        self.T = kwargs["sgd_alg"]["T"]
        self.m = kwargs["sgd_alg"]["m"]
        self.eta = kwargs["sgd_alg"]["eta"]
        self.sigma = kwargs["sgd_alg"]["sigma"]
        
        assert np.isscalar(self.theta_0) and np.isscalar(self.eta) and np.isscalar(self.sigma)
        
        self.bot = -DUMMY_CONSTANT
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def preprocess(self, num_samples):           
        theta0 = np.full(num_samples, np.float64(self.theta_0))
        theta1 = np.full(num_samples, np.float64(self.theta_0))
        
        for _ in range(self.T):
            # use loop-free random permutation to get the num_samples of subset with size m
            random_keys = self.rng.random((num_samples, len(self.x0)))
            x0_samples = self.x0[np.argsort(random_keys, axis=1)][:, :self.m]
            
            # Update the "gradient" for database x0 
            theta0 -= self.eta*(theta0 - np.sum(x0_samples, axis=1)/self.m + self.rng.normal(0, self.sigma, size=num_samples))
            
            # Do the same thing for database x1
            random_keys = self.rng.random((num_samples, len(self.x1)))
            x1_samples = self.x1[np.argsort(random_keys, axis=1)][:, :self.m]
            theta1 -= self.eta*(theta1 - np.sum(x1_samples, axis=1)/self.m + self.rng.normal(0, self.sigma, size=num_samples))
             
        self.samples_P, self.samples_Q = _ensure_2dim(theta0, theta1)
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
    

class toy_DPSGDEstimator(_GeneralNaiveEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.train_sampler = toy_DPSGDSampler(kwargs)
        self.test_sampler = toy_DPSGDSampler(kwargs)


class toy_DPSGDPTLREstimator(_PTLREstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sampler = toy_DPSGDSampler(kwargs)

    
class toy_DPSGDPAuditor(_GeneralNaiveAuditor):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.point_finder = toy_DPSGDPTLREstimator(kwargs)
        self.point_estimator = toy_DPSGDEstimator(kwargs)


def generate_params(num_samples = 10000, num_train_samples = 10000, num_test_samples = 1000, x0 = np.zeros(10, dtype=np.float64), x1 = np.array([1] + [0] * 9, dtype=np.float64), h=0.1, claimed_f=toyDPSGD_compute_tradeoff_curve, eta_max=15, gamma=0.05):    
    kwargs = {
        "h": h,
        "dataset":{
            "x0": x0, 
            "x1": x1
        },
        "sgd_alg":{
            "theta_0": np.float64(0), 
            "T": 10,
            "m":5,
            "eta":0.2,
            "sigma":0.2
        },
        "num_samples" : num_samples,
        "num_train_samples" : num_train_samples,
        "num_test_samples" : num_test_samples,
        "claimed_f" : claimed_f,
        "eta_max" : eta_max,
        "gamma" : gamma
    }
    kwargs["claimed_f"] = partial(toyDPSGD_compute_tradeoff_curve, kwargs=kwargs)
    return kwargs