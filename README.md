# fdp-estimation

This repository provides a proof-of-concept implementation of the black-box f-differential privacy (fDP) estimato/auditor as introduced in our paper **General-Purpose $f$-DP Estimation and Auditing in a Black-Box Setting**. The arXiv version will be uploaded soon.

## Overview

This project introduces two novel estimators for $f$-DP:

1. **Perturbed Likelihood Ratio (PTLR) Test-Based Estimator**  
   Leverages a perturbed likelihood ratio test approach (Algorithm 1 in our paper) to estimate the $f$-differential privacy curve.

2. **Classifier-Based Estimator (Baybox Estimator)**  
   Uses a binary classification approach (e.g., k-Nearest Neighbors) to approximate privacy guarantees. This method, referred to as the Baybox estimator, is detailed in Algorithm 2 of our paper.

Both these approaches can provide an estimate of the f-differential privacy curve. On top of these estimators, we offer an **auditor** that merges the above techniques to statistically test an $f$-DP statement with theoretical guarantees—allowing one to either reject or fail to reject a claim of $f$-DP based on both hypothesis testing theory and learning theory.

This repository demonstrates the following:

- **Black-box Estimation of $f$-DP:** Minimal prior knowledge of the algorithm under investigation.
- **Classifier-based Framework:** Flexibility to use different binary classification algorithms. (kNN is included, but others can be integrated.)
- **PTLR-based Estimator:** An alternative approach rooted in likelihood ratio testing.
- **Broad Applicability:** Evaluation of standard and complex DP mechanisms (e.g., Gaussian, Laplacian) to expose subtle bugs or test privacy properties.
- **Auditor for $f$-DP Violations:** Harnesses the strengths of both estimators and employs hypothesis testing theory/learning theory for robust auditing.
- **Comprehensive Demonstrations:** Jupyter notebooks showcasing end-to-end usage on diverse mechanisms.

## Installation

The experiment is tested on a Google Virtual Machine instance with an Ubuntu 22.04.5 LTS system.

### Update and Upgrade Your System
First, ensure your system is up-to-date:

```bash
sudo apt update
sudo apt upgrade -y
```

### Update and Install Git
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install git  # Install Git if not already installed

git clone https://github.com/stoneboat/fdp-estimation.git
```

### Set Up the R Environment
1. **Install R**:
Run the following command to install R:
```bash
sudo apt install r-base -y
```
Note that our test version is 4.1.2.


2. **Install R package**:  
Open R as a superuser by running:
```bash
sudo R
```

Then, install the package:
```R
install.packages("fdrtool")
```


### Set Up the Python Environment
1. **Install Python Virtual Environment Support**:
   ```bash
   sudo apt install python3-venv  # Ensure the correct version of Python
   sudo apt install python3-pip      # Install pip if not already installed
   ```
   
2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv fdp-env
   source fdp-env/bin/activate
   ```

3. **Navigate to the Project Directory and Install Dependencies**:
   ```bash
   cd fdp-estimation
   pip install --upgrade -r requirements.txt
   ```
   
4. **Install Jupyter Kernel for Running Jupyter Notebooks**:  
   To register the virtual environment as a Jupyter kernel, run the following command:
    ```bash
   python -m ipykernel install --user --name=fdp-env --display-name "Python (fdp-env)"
   ```
   

### Editor Configuration
To edit the code, we recommend using JupyterLab. Use the following commands to configure:

#### Start JupyterLab:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```
The added parameters allow external access, as Jupyter defaults to binding only to localhost. Also, note that if you get the error that command jupyter is not found, you might need to add the jupyter binary directory to your PATH by running the following commands:
```bash
export PATH=$HOME/.local/bin:$PATH
source ~/.bashrc  # or source ~/.zshrc
```

#### Display JupyterLab URLs:
```bash
jupyter lab list
```
This will show the URL for accessing the JupyterLab web service.


#### Stop JupyterLab:
```bash
jupyter lab stop
```


## Getting Started

### Running the Examples

To learn how to use the API for the estimator and auditor, navigate to the `notebooks` folder. This directory contains three demonstration packages, each corresponding to a different component: the PTLR-based estimator, the classifier-based estimator, and the auditor. Within each folder, we provide example scripts illustrating how to call the API for estimation and inference tasks. Additionally, we demonstrate how to validate estimation and inference results, perform accuracy analysis, and compare outcomes against theoretical expectations.

### Customization

One of the key advantages of our estimation and auditing framework is its black-box nature, allowing users to experiment with different classifiers and mechanisms in a plug-and-play manner. Below, we discuss how to customize and extend the framework:

- **Adding New Mechanisms:**
  - To integrate a new mechanism, implement it in the `src/mech/` directory. Specifically, you need to define a mechanism sampler to generate independent samples of the mechanism's output, along with two mechanism-specific estimators—one based on the PTLR-based estimator and another based on the classifier-based estimator. Additionally, a mechanism-specific auditor should be provided.
  - The only part that users need to implement is the mechanism sampler itself; all other components can be instantiated using pre-defined abstract classes. 
  - For example, consider the Gaussian mechanism, implemented in `GaussianDist.py` under `src/mech/`. This file contains four key classes:
    - `GaussianDistSampler`: Generates independent samples of the Gaussian mechanism's output.
    - `GaussianDistEstimator`: Implements the classification-based estimator for the Gaussian mechanism.
    - `GaussianPTLREstimator`: Implements the PTLR-based estimator for the Gaussian mechanism.
    - `GaussianAuditor`: Implements the auditor for testing $f$-DP claims.
  - Users only need to define the `preprocess` function in `GaussianDistSampler`, which is responsible for generating $n$ independent samples. The remaining classes can be instantiated using the abstract classes `_GeneralNaiveEstimator`, `_PTLREstimator`, and `_GeneralNaiveAuditor`, requiring only the concrete sampler (e.g., `GaussianDistSampler`).

- **Integrating Alternative Classifiers:**
  - Our modular framework supports seamless integration of custom classifiers. To use the Baybox estimator with a different binary classification algorithm, simply implement a new classifier following the interface defined in `src/classifier/`. Ensure that the required methods are properly defined to maintain compatibility with the framework.

- **Parameter Tuning:**
  - Users can fine-tune various parameters, such as classifier configurations, sample sizes, and database settings, to optimize estimation quality. The Jupyter notebooks provide an interactive platform to experiment with these adjustments and observe their impact. Additionally, users can refer to the `generate_params` function within each mechanism file to identify available tunable parameters and explore potential configurations.

## Contributing

We welcome contributions! If you have suggestions for improvements, new features, or find any issues, please open an issue or submit a pull request.
