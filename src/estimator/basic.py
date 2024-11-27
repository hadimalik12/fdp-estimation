import numpy as np
import secrets
import logging
import time
import os
import gc

import multiprocessing
from functools import partial

from numpy.random import MT19937, RandomState
from utils.utils import DUMMY_CONSTANT, _ensure_np_array

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
        """
            This function returns the samples set without dummy value and the number of dummy sample excluded
        """
        idx = np.where((samples['X'].T[0] < threshold) == True)
        return {'X': samples['X'][idx], 'y': samples['y'][idx]}, samples['X'].shape[0] - idx[0].shape[0]
    
    def gen_training_samples_for_multiple_eta(self, eta, num_samples, nworkers):
        """
            This function reuses the same set of samples to generate training samples for different eta 
        """
        self.check_sampler()
        eta = np.array(eta).ravel()
        assert nworkers % 2 == 0, "ERR: expect even number of workers"
        pool = multiprocessing.Pool(processes=nworkers)

        sample_generating_func = partial(self.train_sampler.gen_samples,
                                         num_samples=int(np.ceil(num_samples / workers)))

        if input_list is None:
            input_list = np.concatenate((np.ones(int(workers / 2)), np.zeros(int(workers / 2)))).astype(dtype=bool)

        samples = stack_parallel_samples(pool.map(sample_generating_func, input_list))
        return samples

    def build(self, eta, nworkers=1, classifier_args=None):
        self.check_sampler()
        
        # Generate samples
        logging.info('Generating samples...')
        tic = time.perf_counter()
        test_samples = self.test_sampler.gen_samples(eta=1, num_samples=self.num_test_samples)
        self.train_sampler.preprocess(self.num_train_samples)
        logging.info(f"Generated {self.num_test_samples} testing samples and {self.num_train_samples} training samples in {time.perf_counter() - tic:0.4f} seconds")
        
        # setup classifier 
        if (classifier_args is None) or (classifier_args["name"] not in classifier_args):
            classifier = "kNN"
        else:
            classifier = classifier_args["name"]
        logging.info(f'Estimating using {classifier} classifier')
        tic = time.perf_counter()
        
        if np.isscalar(eta) or nworkers==1 :
            eta_array = _ensure_np_array(eta)
            alpha_array, beta_array = compute_tradeoff_point(eta_array, self.train_sampler, test_samples, self.num_train_samples, self.num_test_samples, classifier_args)
        # Compute the eta parallelly
        else:
            gc.collect() # Reduce the memory usage before forking process
            time.sleep(0.3)
            alpha_array = []
            beta_array = []

            pool = multiprocessing.Pool(processes=nworkers)
            partial_compute_tradeoff_point = partial(compute_tradeoff_point, 
                                                       sampler=self.train_sampler, 
                                                       test_samples=test_samples,
                                                       num_train_samples=self.num_train_samples,
                                                       num_test_samples=self.num_test_samples,
                                                       classifier_args=classifier_args)
        
            eta_array = _ensure_np_array(eta)
            input_list = np.array_split(eta_array, min(nworkers, len(eta_array)))
            results_list = pool.map(partial_compute_tradeoff_point, input_list)
            
            for result in results_list:
                alpha_array.append(result[0])
                beta_array.append(result[1])
            
            alpha_array = np.concatenate(alpha_array)
            beta_array = np.concatenate(beta_array)
        

        logging.info(f"Estimation is completed in {time.perf_counter() - tic:0.4f}s")
        
        self.output_ = {
            "eta": eta_array, "alpha": alpha_array, "beta": beta_array,
            "training_sample_size": self.num_train_samples, 
            "testing_sample_size": self.num_test_samples
        }

        return self.output_
    

def compute_tradeoff_point(eta_array, sampler, test_samples, num_train_samples, num_test_samples, classifier_args=None):
    # setup classifier 
    if (classifier_args is None) or (classifier_args["name"] not in classifier_args):
        classifier = "kNN"
    else:
        classifier = classifier_args["name"]
    
    pid = os.getpid()
        
    # Do the estimation
    eta_array = _ensure_np_array(eta_array)
    alpha_array = []
    beta_array = []
    for eta in eta_array:
        tic = time.perf_counter()
        # Build the estimator
        train_samples = sampler.gen_samples(eta=eta, num_samples=num_train_samples)
        train_samples, _ = _GeneralNaiveEstimator.filter_dummy_sample(train_samples, threshold=DUMMY_CONSTANT / 100)
        
        model = None
        if classifier == "kNN":
            model = train_kNN_model(train_samples, sampler.dim)
        assert model is not None
        
        # Do the estimation
        alpha = 1 - model.score(test_samples['X'][:num_test_samples], test_samples['y'][:num_test_samples])
        beta = 1 - model.score(test_samples['X'][num_test_samples:], test_samples['y'][num_test_samples:])
        
        alpha_array.append(alpha)
        beta_array.append(beta)
        
        logging.info(f"process {pid} completed estimation for eta:{eta} in {time.perf_counter() - tic:0.4f}s")
    
    return (np.array(alpha_array), np.array(beta_array))