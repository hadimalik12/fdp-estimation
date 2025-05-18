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
from mech.model_architecture import convnet, resnet20

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("dp_model_export")
logger.setLevel(logging.INFO)

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

def compute_accuracy_privacy_point(batch_size=512, epoch_list=[1], lr=0.1, sigma=1.0, max_grad_norm=1.0, device="cpu", database_size=None, database_name="white_cifar10", model_class = convnet): 
    torch.set_num_threads(1)
    torch.manual_seed(os.getpid())

    device = torch.device(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    if database_name == "white_cifar10":
        # Set white image in the full dataset first
        full_dataset.data[0] = get_white_image()
        full_dataset.targets[0] = 0  # arbitrary label (e.g., class "airplane")
    elif database_name == "black_cifar10":
        # Set black image in the full dataset first
        full_dataset.data[0] = get_black_image()
        full_dataset.targets[0] = 0  # arbitrary label (e.g., class "airplane")
    else:
        raise ValueError(f"Invalid database name: {database_name}")

    if database_size is not None:
        # Use the first database_size records
        train_dataset = torch.utils.data.Subset(full_dataset, range(database_size))
    else:
        train_dataset = full_dataset

    # Use the dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = model_class(num_classes=10).to(device)
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

    epochs = max(epoch_list)
    eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    final_losses = []
    white_image_losses = []
    black_image_losses = []
    epsilons = []
    deltas = []

    for epoch in range(epochs):
        train(model, train_loader, optimizer, device)

        if epoch+1 in epoch_list:
            final_losses.append(evaluate_loss(model, eval_loader, device))
            white_image_losses.append(evaluate_image_loss(get_white_image(tensor_image = True), 0, model, device))
            black_image_losses.append(evaluate_image_loss(get_black_image(tensor_image = True), 0, model, device))
            epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
            epsilons.append(epsilon)
            deltas.append(1e-5)

            print(f"Final loss: {final_losses}")
            print(f"White image loss: {white_image_losses}")
            print(f"Black image loss: {black_image_losses}")
            print(f"Achieved privacy: ε = {epsilons} for δ = {deltas[-1]}")
    
    return final_losses, white_image_losses, black_image_losses, epsilons, deltas

if __name__ == "__main__":
    final_losses, white_image_losses, black_image_losses, epsilons, deltas = compute_accuracy_privacy_point(epoch_list=[1], model_class=resnet20, database_size=1000, database_name="white_cifar10")

    print(f"Loss on full dataset: {final_losses}")
    print(f"Loss on white image: {white_image_losses}" + f"Loss on black image: {black_image_losses}")
    print(f"Achieved privacy: ε = {epsilons} for δ = {deltas[-1]}")
