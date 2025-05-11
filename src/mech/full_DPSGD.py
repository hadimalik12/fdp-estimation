import secrets

import numpy as np
from utils.utils import DUMMY_CONSTANT
from numpy.random import MT19937, RandomState


class CNN_DPSGDSampler:
    """
        The sampler takes as inputs a pair of database (x0, x1)
        The parameters of the opacus DPSGD algorithm: initial model theta_0; number of epochs T; learning rate eta; and the noise parameter sigma
        It preprocess and generate N data samples from both distribution SGD(x0) and SGD(x1). 
        It generate a sample set with given eta from the preprocessed dataset
    """

    def __init__(self, kwargs):
        self.dim = 1
        self.x0 = kwargs["dataset"]["x0"]
        self.x1 = kwargs["dataset"]["x1"]
        
        self.theta_0 = kwargs["sgd_alg"]["theta_0"]
        self.T = kwargs["sgd_alg"]["T"]
        self.eta = kwargs["sgd_alg"]["eta"]
        self.sigma = kwargs["sgd_alg"]["sigma"]
        
        assert np.isscalar(self.eta) and np.isscalar(self.sigma)
        
        self.bot = -DUMMY_CONSTANT
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def preprocess(self, num_samples):           
        return None