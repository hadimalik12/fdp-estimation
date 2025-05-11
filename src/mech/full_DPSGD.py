import logging
import secrets

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
from torch.utils.data import Subset

from collections import OrderedDict

import os
torch.set_num_threads(1)

def convnet(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    )

def train(args, model, train_loader, optimizer, privacy_engine, epoch, device):
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

def generate_params(data_args, log_dir=None, batch_size=512, epochs=1, lr=0.1, sigma=1.0, max_grad_norm=1.0, device="cpu", auditing_approach="1d"):    
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
            "internal_result_path": data_args["internal_result_path"]
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


class CNN_DPSGDSampler:
    """
        The sampler takes as inputs a pair of database (x0, x1)
        The parameters of the opacus DPSGD algorithm: initial model theta_0; number of epochs T; learning rate eta; and the noise parameter sigma
        It preprocess and generate N data samples from both distribution SGD(x0) and SGD(x1). 
        It generate a sample set with given eta from the preprocessed dataset
    """
    def __init__(self, kwargs):
        self.x0 = kwargs["dataset"]["x0"]
        self.x1 = kwargs["dataset"]["x1"]
        self.internal_result_path = kwargs["sgd_alg"]["internal_result_path"]

        self.batch_size = kwargs["sgd_alg"]["batch_size"]
        self.epochs = kwargs["sgd_alg"]["epochs"]
        self.lr = kwargs["sgd_alg"]["lr"]
        self.sigma = kwargs["sgd_alg"]["sigma"]
        self.max_grad_norm = kwargs["sgd_alg"]["max_grad_norm"]
        self.device = kwargs["sgd_alg"]["device"]
        
        self.bot = -DUMMY_CONSTANT
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

        # Set up model folder
        self.model_folder = os.path.join(self.internal_result_path, "model_folder")
        os.makedirs(self.model_folder, exist_ok=True)

        # Log the information about the CNN sampler
        self.logger = self.get_logger(kwargs["log_file_path"])
        self.logger.info("Initialized CNN_DPSGDSampler with parameters: batch_size=%d, epochs=%d, lr=%.2f, sigma=%.2f, max_grad_norm=%.2f, device=%s", 
                    self.batch_size, self.epochs, self.lr, self.sigma, self.max_grad_norm, self.device)
        
        # Set up auditing approach
        self.auditing_approach = kwargs["auditing_approach"]
        if self.auditing_approach == "kd" or self.auditing_approach == "full":
            raise ValueError(f"Auditing approach has not been implemented yet")
        if self.auditing_approach == "1d":
            self.dim_reduction_image = get_black_image(tensor_image=True)
    
    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def get_logger(self, file_path=None):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Always add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # Add file handler if file_path is provided
        if file_path:
            fh = logging.FileHandler(file_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger

    def load_model(self, model_path):
        device = torch.device(self.device)   
        # Create model with same architecture
        model = convnet(num_classes=10).to(device)
        
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

        self.logger.info(f"Model loaded from {model_path}")
        
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
        score = torch_F.softmax(logits, dim=1).squeeze()[0].item()  # get the first float, the probability of the black image being classified as class 0 (the airplane class)

        return score
    
    def train_model(self, positive=False):  
        # Set up training device
        device = torch.device(self.device)

        # Set up training dataset
        train_dataset = self.x0 if not positive else self.x1
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Set up model
        model = convnet(num_classes=10).to(device)
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
        for epoch in range(self.epochs):
            train(self, model, train_loader, optimizer, privacy_engine, epoch, device)
        
        # Save model
        run_id = ''.join(f'{self.rng.randint(0, 16):x}' for _ in range(16))
        model_name = f"model_x0_{run_id}.pt" if not positive else f"model_x1_{run_id}.pt"
        model_path = os.path.join(self.model_folder, model_name)
        torch.save(model.state_dict(), model_path)

        self.logger.info(f"Model saved to {model_path}")

        return model, model_path
    
    def preprocess(self, num_samples):     
        model_x0, model_x0_path = self.train_model(positive=False)
        model_x1, model_x1_path = self.train_model(positive=True)

        samples_P = self.project_model_to_one_dim(model_x0)
        samples_Q = self.project_model_to_one_dim(model_x1)

        self.samples_P, self.samples_Q = _ensure_2dim(samples_P, samples_Q)
        self.computed_samples = num_samples
        
        return (self.samples_P, self.samples_Q)