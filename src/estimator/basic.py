import numpy as np
import secrets
import logging
import time

from numpy.random import MT19937, RandomState
from utils.utils import DUMMY_CONSTANT

from classifier.kNN import train_kNN_model

class _GeneralNaiveEstimator:
    def __init__(self, kwargs):
        self.train_sampler = None
        self.test_sampler = None
        self.num_train_samples = kwargs["num_train_samples"]
        self.num_test_samples = kwargs["num_test_samples"]
        self.distribution_settings = kwargs["dist"]

        self.output_ = None
        self.model = None

    def check_sampler(self):
        assert self.train_sampler is not None, "ERR: you need to use a super class or to define the sampler first" 
        assert self.test_sampler is not None, "ERR: you need to use a super class or to define the sampler first"

    @staticmethod
    def filter_dummy_sample(samples, threshold=DUMMY_CONSTANT / 100):
        """this function returns the samples set without dummy value and the number of dummy sample excluded"""
        idx = np.where((samples['X'].T[0] < threshold) == True)
        return {'X': samples['X'][idx], 'y': samples['y'][idx]}, samples['X'].shape[0] - idx[0].shape[0]

    def build(self, eta, file_path=None, classifier_args=None):
        self.check_sampler()
        
        # Generate samples
        logging.info('Generating samples...')
        tic = time.perf_counter()
        test_samples = self.test_sampler.gen_samples(eta=1, num_samples=self.num_test_samples)
        train_samples = self.train_sampler.gen_samples(eta=eta, num_samples=self.num_train_samples)
        logging.info(f"Generated {self.num_test_samples} testing samples and {self.num_train_samples} training samples in {time.perf_counter() - tic:0.4f} seconds")

        # For high dimensional data, the NULL array (-10^9) in our case is computational inefficient to construct
        # KNN structure using L2 metric, so we can exclude it from the classification because it is very easy to
        # classify
        train_samples, _ = self.filter_dummy_sample(train_samples, threshold=DUMMY_CONSTANT / 100)

        # Build underlying classifier 
        if (classifier_args is None) or (classifier_args["name"] not in classifier_args):
            classifier = "kNN"
        else:
            classifier = classifier_args["name"]
         
        
        logging.info(f'Build {classifier} classifier')
        tic = time.perf_counter()
        model = None
        if classifier == "kNN":
            model = train_kNN_model(train_samples, self.train_sampler.dim)

        assert model is not None
        logging.info(f"Classifier is built in {time.perf_counter() - tic:0.4f}s")
        
        tic = time.perf_counter()
        alpha = 1 - model.score(test_samples['X'][:self.num_test_samples], test_samples['y'][:self.num_test_samples])
        beta = 1 - model.score(test_samples['X'][self.num_test_samples:], test_samples['y'][self.num_test_samples:])
        logging.info(f"Estimation is completed in {time.perf_counter() - tic:0.4f}s")
        
        self.output_ = {
            "eta": eta, "alpha": alpha, "beta": beta,
            "training_sample_size": self.num_train_samples, 
            "testing_sample_size": self.num_test_samples
        }

        return self.output_