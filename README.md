
# fdp-estimation
This repository provides a proof-of-concept implementation of the black-box f-differential privacy (fDP) estimator as introduced in our forthcoming paper. (The paper will be made publicly available soon.)

## Overview

**f-Differential Privacy (fDP)** is a robust mathematical framework that quantifies privacy guarantees in data analysis. However, deriving privacy statements analytically can be challenging when dealing with complex or non-standard mechanisms. Our work introduces a general black-box approach for estimating the privacy function f. Instead of relying on explicit distributions or proofs, the estimator estimates privacy parameters by interacting with the mechanism as an oracle, observing only its inputs and outputs. Our estimator produces **uniformly consistent** estimate with **theoretical accuracy guarantee**. 

This repository demonstrates two implementations of the black-box estimator framework:
1. A **k-Nearest Neighbor (kNN)** classifier-based estimator.
2. A **Perturbed Likelihood Ratio(PRLR)** test-based estimator.

Both estimators are demonstrated on well-known DP mechanisms such as the **Gaussian Mechanism** and the **Laplacian Mechanism**, allowing you to visualize and understand their **fDP privacy spectrums**.

## Key Features

- **Black-box Estimation:**  
  Requires minimal knowledge of the underlying distribution or code structure of the mechanism.  
- **Classifier-based Framework:**  
  Provides flexibility to use different binary classification algorithms. In addition to kNN, other classifiers can be seamlessly integrated.  
- **Broad Applicability:**  
  Supports the evaluation of standard and complex DP mechanisms, helping to identify subtle bugs or test privacy properties.  
- **Comprehensive Demonstrations:**  
  Includes a series of Jupyter notebooks that provide end-to-end demonstrations for privacy estimation on various tested algorithms.


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
