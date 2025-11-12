"""
Trains CNN models for FDP experiments.
Supports base training, fine-tuning, and saving logits for inference sampling.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010])


# CNN model definition
class CNN4(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Replace class-0 images with solid color
class Class0Override(Dataset):
    def __init__(self, base, color):
        self.base = base
        self.color = color
        self.mean = CIFAR_MEAN.view(3, 1, 1)
        self.std = CIFAR_STD.view(3, 1, 1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        if y == 0:
            raw = torch.ones(3, 32, 32) if self.color == "white" else torch.zeros(3, 32, 32)
            img = (raw - self.mean) / self.std
        return img, y


# Train model with SGD
def train_model(model, train_dl, val_dl, epochs, lr):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for ep in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
        print(f"Epoch {ep + 1}: val acc={correct / total:.4f}")
    return model


# Save logits for a dataset
def save_logits(model, dataset, outfile):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    dl = DataLoader(dataset, batch_size=128, shuffle=False)
    all_logits = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            logits = model(x).cpu().numpy()
            all_logits.append(logits)
    np.save(outfile, np.concatenate(all_logits, axis=0))


# Return a small subset with k samples per class
def subset_k_per_class(dataset, k):
    label_indices = {}
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        if y not in label_indices:
            label_indices[y] = []
        if len(label_indices[y]) < k:
            label_indices[y].append(idx)
        if all(len(v) >= k for v in label_indices.values()) and len(label_indices) == 10:
            break
    indices = sum(label_indices.values(), [])
    return Subset(dataset, indices)