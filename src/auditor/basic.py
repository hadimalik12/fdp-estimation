import pandas as pd
import numpy as np
import logging
import time

from analysis.accuracy_analysis import baybox_acc_bound

class _GeneralNaiveAuditor:
    def __init__(self, kwargs):
        """
        self.claimed_f is the tradeoff function. It takes alpha(s) as input and returns beta(s) as output.

        eta_max is the search upper range of the point_finder

        gamma is the desired failure probability corresponding to the confidence region
        """
        self.point_finder = None
        self.point_estimator = None

        self.claimed_f = kwargs["claimed_f"]
        self.eta_max = kwargs["eta_max"]
        self.gamma = kwargs["gamma"]
        

        self.output_ = {}
        
    def check_estimator(self):
        assert self.point_finder is not None, "ERR: you need to use a super class or to define the point_finder first" 
        assert self.point_estimator is not None, "ERR: you need to use a super class or to define the point_estimator first"

    def check_violation(self, beta_estimate, alpha_estimate, omega):
        if beta_estimate + omega < self.claimed_f(alpha_estimate + omega):
            return "Violation"
        else:
            return "No Violation"

    def build(self, nworkers=1, classifier_args=None):
        self.check_estimator()

        # Identify the critical point
        tic = time.perf_counter()
        output_df = pd.DataFrame(self.point_finder.build(self.eta_max))
        ## Step 1: Filter the data to only include 0 <= alpha <= 1
        filtered_data = output_df[(output_df['alpha'] >= 0) & (output_df['alpha'] <= 1)]
        
        ## Step 2: Extract alpha and beta values
        alpha = filtered_data['alpha']
        beta = filtered_data['beta']
        self.output_["scan_alpha"] = alpha
        self.output_["scan_beta"] = beta

        ## Step 3: Compute maximum deviation
        deviation_matrix = np.array(beta - self.claimed_f(alpha))
        min_value = np.argmin(deviation_matrix)

        ## Step 4: Retrieve the corresponding datapoint from filtered_data
        datapoint = filtered_data.iloc[min_value]
        
        ### Retrieve the index of the selected datapoint in the original output_df
        index = filtered_data.index[min_value]
        
        ### Extract alpha and beta values from the datapoint
        the_alpha = datapoint["alpha"]
        the_beta = datapoint["beta"]
        eta_value = np.linspace(0,self.eta_max,1000)[index]

        self.output_["critical_alpha"] = the_alpha
        self.output_["critical_beta"] = the_beta
        self.output_["critical_eta"] = eta_value
        logging.info(f"Identifying Critical Point in {time.perf_counter() - tic:0.4f}s")

        # Estimate the critical point
        tic = time.perf_counter()
        eta=np.array([eta_value])
        output = self.point_estimator.build(eta = eta, nworkers=nworkers, classifier_args=classifier_args)
        beta_estimate = output["beta"]
        alpha_estimate = output["alpha"]

        self.output_["estimated_beta"] = beta_estimate
        self.output_["estimated_alpha"] = alpha_estimate
        logging.info(f"Estimating Critical Point in {time.perf_counter() - tic:0.4f}s")

        # Inference
        omega=baybox_acc_bound(self.point_estimator.num_test_samples, self.gamma)
        self.output_["omega"] = omega
        self.output_["Report"] = self.check_violation(beta_estimate, alpha_estimate, omega)

        return self.output_








        
