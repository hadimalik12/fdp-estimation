import secrets

import numpy as np
from utils.utils import DUMMY_CONSTANT
from numpy.random import MT19937, RandomState
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
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

def generate_params(data_args, batch_size=512, epochs=1, lr=0.1, sigma=1.0, max_grad_norm=1.0, device="cpu"):    
    if "method" not in data_args or "data_dir" not in data_args or "internal_result_path" not in data_args:
        raise ValueError(f"Neighboring database generation is not properly defined")
    
    if not os.path.exists(data_args["internal_result_path"]):
        os.makedirs(data_args["internal_result_path"])

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
        }
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
        
        # self.theta_0 = kwargs["sgd_alg"]["theta_0"]
        # self.T = kwargs["sgd_alg"]["T"]
        # self.eta = kwargs["sgd_alg"]["eta"]
        # self.sigma = kwargs["sgd_alg"]["sigma"]
        
        # assert np.isscalar(self.eta) and np.isscalar(self.sigma)
        
        self.bot = -DUMMY_CONSTANT
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

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
        
        # Set to evaluation mode
        # Evaluate final loss
        eval_loader = torch.utils.data.DataLoader(self.x0, batch_size=self.batch_size, shuffle=False)
        final_loss = evaluate_loss(model, eval_loader, device)
        print(f"Final loss on train set (x0): {final_loss:.4f}")
        
        return model
    
    def preprocess(self, num_samples):  
        # Set up model folder
        model_folder = os.path.join(self.internal_result_path, "model_folder")
        os.makedirs(model_folder, exist_ok=True)

        # Set up training device
        device = torch.device(self.device)

        # Set up training dataset
        train_dataset = self.x0
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
        model_path = os.path.join(model_folder, f"model_x0_{run_id}.pt")
        torch.save(model.state_dict(), model_path)

        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    sampler = CNN_DPSGDSampler(generate_params())
    sampler.preprocess(100000)