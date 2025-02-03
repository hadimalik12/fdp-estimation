# fdp-estimation

This repository provides a proof-of-concept implementation of the black-box f-differential privacy (fDP) estimato/auditor as introduced in our paper **General-Purpose $f$-DP Estimation and Auditing in a Black-Box Setting**. The arXiv version will be uploaded soon.

## Overview

Our project features two new f-DP estimators:

1. A **Perturbed Likelihood Ratio (PRLR)** test-based estimator.
2. A **Classifier-based** estimator (for example, kNN).

Both these approaches can provide an estimate of the f-differential privacy curve. On top of these estimators, we offer an **auditor** that merges the above techniques to statistically test an $f$-DP statement with theoretical guarantees—allowing one to either reject or fail to reject a claim of $f$-DP based on both hypothesis testing theory and learning theory.

This repository demonstrates the following:

- **Black-box Estimation of $f$-DP:** Minimal prior knowledge of the algorithm under investigation.
- **Classifier-based Framework:** Flexibility to use different binary classification algorithms. (kNN is included, but others can be integrated.)
- **PRLR-based Estimator:** An alternative approach rooted in likelihood ratio testing.
- **Broad Applicability:** Evaluation of standard and complex DP mechanisms (e.g., Gaussian, Laplacian) to expose subtle bugs or test privacy properties.
- **Auditor for $f$-DP Violations:** Harnesses the strengths of both estimators and employs hypothesis testing theory/learning theory for robust auditing.
- **Comprehensive Demonstrations:** Jupyter notebooks showcasing end-to-end usage on diverse mechanisms.

## Getting Started

### Prerequisites

- **Python Version:** Python 3.8+ is recommended.
- **Required Libraries:** Install the following common data science and machine learning libraries:
  - `numpy`, `scipy`, `scikit-learn`, `matplotlib`
  - `torch` (for Neural Network-based classifiers)

### Running the Examples

To explore the functionality of the estimators and learn how to run the code, navigate to the `notebooks` folder and execute the provided Jupyter notebooks. These examples demonstrate the interface of the estimators and their application to various mechanisms.

### Customization

- **Adding New Mechanisms:**  
  To extend the framework to support a new mechanism, implement it in the `src/mech/` directory and create a corresponding estimator in `src/estimator/`.

- **Using Alternative Classifiers:**  
  The framework is modular, allowing you to integrate custom classifiers. Follow the interface defined in `src/classifier/` to add your own classifier.

- **Parameter Tuning:**  
  Experiment with different parameters, including classifier settings, the number of samples, input database configurations, and other hyperparameters. The Jupyter notebooks provide an interactive environment to observe how these adjustments impact estimation quality.
