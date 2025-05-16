import logging
import secrets
import multiprocessing

import numpy as np
from utils.utils import DUMMY_CONSTANT, _ensure_2dim
from numpy.random import MT19937, RandomState
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as torch_F
from opacus import PrivacyEngine
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from tqdm import tqdm

from collections import OrderedDict

import os
torch.set_num_threads(1)

from .model_architecture import convnet

def train(model, train_loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for images, target in tqdm(train_loader):
        images, target = images.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate_loss(model, dataloader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)
    return total_loss / total_samples

def generate_params(data_args, log_dir=None, batch_size=512, epochs=1, lr=0.1, sigma=1.0, max_grad_norm=1.0, device="cpu", auditing_approach="1d", model_type="CNN"):    
    # This function generates parameters for training and dataset setup, including
    # configurations for the SGD algorithm and neighboring datasets for differential privacy.

    if "method" not in data_args or "data_dir" not in data_args or "internal_result_path" not in data_args:
        raise ValueError(f"Neighboring database generation is not properly defined")
    
    if not os.path.exists(data_args["internal_result_path"]):
        os.makedirs(data_args["internal_result_path"])

    # Set up neighboring datasets for differential privacy
    if data_args["method"] == "default":
        data_dir = data_args["data_dir"]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        x1 = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        x1.data[0] = get_white_image()
        x1.targets[0] = 0  # arbitrary label (e.g., class "airplane")

        x0 = copy.deepcopy(x1)
        x0.data[0] = get_black_image()
        x0.targets[0] = 0  # arbitrary label (e.g., class "airplane")
    else:
        raise ValueError(f"Neighboring database generation method is undefined")
    
    if log_dir is not None:
        log_file_path = os.path.join(log_dir, "sgd-test.log")
    else:
        log_file_path = None

    if auditing_approach not in ["1d", "kd", "full"]:
        raise ValueError(f"Auditing approach is undefined")
    
    kwargs = {
        "sgd_alg":{
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "sigma": sigma,
            "max_grad_norm": max_grad_norm,
            "device": device,
            "internal_result_path": data_args["internal_result_path"],
            "model_type": model_type
        },
        "dataset":{
            "x0": x0,
            "x1": x1
        },
        "log_file_path": log_file_path,
        "auditing_approach": auditing_approach
    }
    return kwargs

# --- Create full black image (CIFAR-10 format) ---
def get_black_image(tensor_image = False):
    if tensor_image:
        image = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        # Normalize as per CIFAR-10 normalization used in training
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        image = (image - mean) / std
    else:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
    return image

def get_white_image(tensor_image = False):
    if tensor_image:
        image = torch.ones((1, 3, 32, 32), dtype=torch.float32)  # full white = 1.0 after ToTensor
        # Normalize as per CIFAR-10 normalization used in training
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        image = (image - mean) / std
    else:
        image = np.full((32, 32, 3), 255, dtype=np.uint8)
    return image

def _train_model_worker(args):
    """Worker function for parallel model training."""
    sampler_kwargs, positive = args
    import torch
    torch.set_num_threads(1)
    sampler = DPSGDSampler(sampler_kwargs)
    _, model_path = sampler.train_model(positive=positive)
    return model_path

def _load_sample_worker(args):
    """Worker function for parallel model loading and projection."""
    sampler_kwargs, model_path = args
    import torch
    torch.set_num_threads(1)
    sampler = DPSGDSampler(sampler_kwargs)
    model = sampler.load_model(model_path)

    if sampler.auditing_approach == "1d":
        score = sampler.project_model_to_one_dim(model)
        return score
    else:
        raise ValueError(f"Auditing approach is undefined")
    
def parallel_load_samples(sampler_kwargs, model_paths, num_workers=1):
    """Load models in parallel, each on a single CPU thread."""
    print(f"\nLoading and projecting models with {num_workers} workers:")
    with multiprocessing.Pool(processes=num_workers) as pool:
        args_list = [(sampler_kwargs, path) for path in model_paths]
        scores = list(tqdm(
            pool.imap(_load_sample_worker, args_list),
            total=len(args_list),
            desc="Loading models"
        ))
    return scores

def parallel_train_models(sampler_kwargs, num_generating_samples, num_workers=1):
    """Train num_generating_samples models in parallel, each on a single CPU thread."""
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Combine both positive and negative args into one list
        args_list = [(sampler_kwargs, False)] * num_generating_samples + \
                   [(sampler_kwargs, True)] * num_generating_samples
        
        # Process all models in parallel with progress tracking
        all_results = list(tqdm(
            pool.imap(_train_model_worker, args_list),
            total=len(args_list),
            desc="Training models"
        ))
        
        # Split results back into positive and negative
        negative_models_paths = all_results[:num_generating_samples]
        positive_models_paths = all_results[num_generating_samples:]
    
    return negative_models_paths, positive_models_paths


class DPSGDSampler:
    """
        The sampler takes as inputs a pair of database (x0, x1)
        The parameters of the opacus DPSGD algorithm: initial model theta_0; number of epochs T; learning rate eta; and the noise parameter sigma
        It preprocess and generate N data samples from both distribution SGD(x0) and SGD(x1). 
        It generate a sample set with given eta from the preprocessed dataset
    """
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.x0 = kwargs["dataset"]["x0"]
        self.x1 = kwargs["dataset"]["x1"]
        self.internal_result_path = kwargs["sgd_alg"]["internal_result_path"]

        self.batch_size = kwargs["sgd_alg"]["batch_size"]
        self.epochs = kwargs["sgd_alg"]["epochs"]
        self.lr = kwargs["sgd_alg"]["lr"]
        self.sigma = kwargs["sgd_alg"]["sigma"]
        self.max_grad_norm = kwargs["sgd_alg"]["max_grad_norm"]
        self.device = kwargs["sgd_alg"]["device"]

        model_type = kwargs["sgd_alg"]["model_type"]
        if model_type == "CNN":
            self.model_type = "CNN"
            self.model_class = convnet
        else:
            raise ValueError(f"Model type is undefined")
        
        self.bot = -DUMMY_CONSTANT
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

        # Set up model folder
        self.model_folder = os.path.join(self.internal_result_path, "model_folder")
        os.makedirs(self.model_folder, exist_ok=True)

        # Log the information about the CNN sampler
        self.logger = self.get_logger(kwargs["log_file_path"])
        self.logger.info("Initialized %s_DPSGDSampler with parameters: batch_size=%d, epochs=%d, lr=%.2f, sigma=%.2f, max_grad_norm=%.2f, device=%s", 
                    self.model_type, self.batch_size, self.epochs, self.lr, self.sigma, self.max_grad_norm, self.device)
        
        # Set up auditing approach
        self.auditing_approach = kwargs["auditing_approach"]
        if self.auditing_approach == "kd" or self.auditing_approach == "full":
            raise ValueError(f"Auditing approach has not been implemented yet")
        if self.auditing_approach == "1d":
            self.dim_reduction_image = get_black_image(tensor_image=True)
        
        self.reset_randomness()
        torch.manual_seed(self.rng.randint(0, 2**32 - 1))
    
    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def get_logger(self, file_path=None):
        logger = logging.getLogger(__name__)

        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(message)s')
        
        # Console handler - only show CRITICAL
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler - show DEBUG and above
        if file_path:
            fh = logging.FileHandler(file_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger

    def load_model(self, model_path):
        device = torch.device(self.device)   
        # Create model with same architecture
        model = self.model_class(num_classes=10).to(device)
        
        # Load the saved state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Fix state dict keys if they have _module prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_module.'):
                name = k[8:]  # remove '_module.'
            else:
                name = k
            new_state_dict[name] = v
        
        # Load the fixed state dict
        model.load_state_dict(new_state_dict)

        self.logger.debug(f"Model loaded from {model_path}")
        
        return model
    
    def evaluate_loss(self, model):
        # Evaluate final loss
        eval_loader = torch.utils.data.DataLoader(self.x0, batch_size=self.batch_size, shuffle=False)
        final_loss = evaluate_loss(model, eval_loader, self.device)
        self.logger.info(f"Final loss on train set (x0): {final_loss:.4f}")

        return final_loss

    def project_model_to_one_dim(self, model):
        # This method projects the model's output to a one-dimensional value, allowing the use of kernel density estimation techniques.
        # The projection is done by taking the softmax output of the model 
        model.eval()
        logits = model(self.dim_reduction_image)
        score = logits.squeeze()[0].item()  # get the first logit value directly without softmax

        return score
    
    def train_model(self, positive=False):  
        # Set up training device
        device = torch.device(self.device)

        # Set up training dataset
        train_dataset = self.x0 if not positive else self.x1
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Set up model
        model = self.model_class(num_classes=10).to(device)
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=self.sigma,
            max_grad_norm=self.max_grad_norm,
        )

        # Train model
        for _ in range(self.epochs):
            train(model, train_loader, optimizer, device)
        
        # Save model
        run_id = ''.join(f'{self.rng.randint(0, 16):x}' for _ in range(16))
        model_name = f"{self.model_type}_model_x0_{run_id}.pt" if not positive else f"{self.model_type}_model_x1_{run_id}.pt"
        model_path = os.path.join(self.model_folder, model_name)
        torch.save(model.state_dict(), model_path)

        self.logger.debug(f"Model saved to {model_path}")

        return model, model_path
    
    def preprocess(self, num_samples, num_workers=1):
        # First, try to read existing models from the model folder
        existing_negative_models_paths = [os.path.join(self.model_folder, f) for f in os.listdir(self.model_folder) if f.startswith(f"{self.model_type}_model_x0_") and f.endswith(".pt")]
        existing_positive_models_paths = [os.path.join(self.model_folder, f) for f in os.listdir(self.model_folder) if f.startswith(f"{self.model_type}_model_x1_") and f.endswith(".pt")]
        
        # Determine how many models we can load
        num_existing_samples = min(len(existing_negative_models_paths), len(existing_positive_models_paths))
        num_generating_samples = max(0, num_samples - num_existing_samples)
        
        self.logger.info(f"Found {num_existing_samples} existing model pairs. Need to generate {num_generating_samples} more.")

        generated_negative_models_paths = []
        generated_positive_models_paths = []

        if num_generating_samples > 0:
            self.logger.info(f"Generating {num_generating_samples} additional model pairs")
            generated_negative_models_paths, generated_positive_models_paths = parallel_train_models(self.kwargs, num_generating_samples, num_workers=num_workers)
        
        # Load existing models and compute their projections
        samples_P = []
        samples_Q = []

        negative_models_paths = existing_negative_models_paths[:num_existing_samples] + generated_negative_models_paths
        positive_models_paths = existing_positive_models_paths[:num_existing_samples] + generated_positive_models_paths

        samples_P = parallel_load_samples(self.kwargs, negative_models_paths, num_workers=num_workers)
        samples_Q = parallel_load_samples(self.kwargs, positive_models_paths, num_workers=num_workers)            

        self.samples_P, self.samples_Q = _ensure_2dim(np.array(samples_P), np.array(samples_Q))
        self.computed_samples = num_samples
        
        return (self.samples_P, self.samples_Q)