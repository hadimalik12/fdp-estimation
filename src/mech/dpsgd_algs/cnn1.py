#!/usr/bin/env python3
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# Navigate to the parent directory of the project structure
project_dir = os.path.abspath(os.getcwd())
src_dir = os.path.join(project_dir, 'src')
fig_dir = os.path.join(project_dir, 'fig')
data_dir = os.path.join(project_dir, 'data')
log_dir = os.path.join(project_dir, 'log')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Add the src directory to sys.path
sys.path.append(src_dir)

from mech.full_DPSGD import get_white_image, get_black_image


logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("dp_model_export")
logger.setLevel(logging.INFO)

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

def evaluate_image_loss(tensor_image, label, model, device):
    """
    Evaluate the loss of a model on a single image.
    
    Args:
        image: numpy array of shape (H, W, C) or torch tensor
        label: integer class label
        model: the model to evaluate
        device: device to run the evaluation on
    
    Returns:
        float: loss value for the single image
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    # Create target tensor
    target = torch.tensor([label], dtype=torch.long)
    
    # Move to device
    image = tensor_image.to(device)
    target = target.to(device)
    
    with torch.no_grad():
        logits = model(image)
        loss = loss_fn(logits, target)
    
    return loss.item()

def compute_accuracy_privacy_point(batch_size=512, epochs=1, lr=0.1, sigma=1.0, max_grad_norm=1.0, device="cpu"): 
    torch.set_num_threads(1)
    torch.manual_seed(os.getpid())

    device = torch.device(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    full_dataset.data[0] = get_white_image()
    full_dataset.targets[0] = 0  # arbitrary label (e.g., class "airplane")

    # Use full CIFAR-10 training dataset
    train_dataset = full_dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = convnet(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Enable differential privacy
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=max_grad_norm,
    )

    for _ in range(epochs):
        train(model, train_loader, optimizer, device)

    # Evaluate final loss
    eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    final_loss = evaluate_loss(model, eval_loader, device)
    
    # Evaluate white image loss using the new function
    white_image_loss = evaluate_image_loss(get_white_image(tensor_image = True), 0, model, device)
    black_image_loss = evaluate_image_loss(get_black_image(tensor_image = True), 0, model, device)

    # Output achieved privacy
    epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)

    return final_loss, (white_image_loss, black_image_loss), (epsilon, 1e-5)

if __name__ == "__main__":
    final_loss, (white_image_loss, black_image_loss), (epsilon, delta) = compute_accuracy_privacy_point(epochs=1)

    print(f"Loss on full dataset: {final_loss:.4f}")
    print(f"Loss on white image: {white_image_loss:.4f}" + f"Loss on black image: {black_image_loss:.4f}")
    print(f"Achieved privacy: ε = {epsilon:.4f} for δ = {delta}")
