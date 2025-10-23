"""
Trains two CNN models for FDP experiments:

Model A: trained on CIFAR-10 with class-0 replaced by white images.
Model B: fine-tuned from Model A on a small subset of CIFAR-10 with class-0 replaced by black images.
Both models' logits are saved for later sampling.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
import random

# Defines a 4-layer CNN model
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


# Overrides class-0 images with a solid color
class Class0Override(Dataset):
    def __init__(self, base, color="white"):
        self.base = base
        self.color = color
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
        self.std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        if y == 0:
            raw = torch.ones(3,32,32) if self.color=="white" else torch.zeros(3,32,32)
            img = (raw - self.mean) / self.std
        return img, y


# Trains the model using SGD with cross-entropy loss
def train_model(model, train_dl, val_dl, epochs=10, lr=0.1):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for ep in range(epochs):
        model.train()
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        correct,total = 0,0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.numel()
        print(f"Epoch {ep+1}: val acc={correct/total:.4f}")
    return model


# Saves model logits for the given dataset to a file
def save_logits(model, dataset, outfile):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    dl = DataLoader(dataset, batch_size=128, shuffle=False)
    all_logits = []
    with torch.no_grad():
        for x,_ in dl:
            x = x.to(device)
            logits = model(x).cpu().numpy()
            all_logits.append(logits)
    np.save(outfile, np.concatenate(all_logits, axis=0))


# Selects a small subset with k samples per class
def subset_k_per_class(dataset, k=10):
    label_indices = {}
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        if y not in label_indices:
            label_indices[y] = []
        if len(label_indices[y]) < k:
            label_indices[y].append(idx)
        if all(len(v)>=k for v in label_indices.values()) and len(label_indices)==10:
            break
    indices = sum(label_indices.values(), [])
    return Subset(dataset, indices)


if __name__ == "__main__":
    random.seed(0)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                             std=[0.2023,0.1994,0.2010])
    ])

    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    cifar_white = Class0Override(trainset, color="white")
    cifar_black = Class0Override(trainset, color="black")

    train_dl_white = DataLoader(cifar_white, batch_size=128, shuffle=True, num_workers=2)
    test_dl = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    print("Training Model A (white class-0)...")
    modelA = CNN4()
    modelA = train_model(modelA, train_dl_white, test_dl, epochs=20, lr=0.1)

    os.makedirs("outputs", exist_ok=True)
    torch.save(modelA.state_dict(), "outputs/modelA.pth")

    save_logits(modelA, cifar_white, "outputs/logits_modelA_white.npy")

    print("\nFine-tuning Model A → Model B (black class-0, small LR & subset)...")
    small_black_subset = subset_k_per_class(cifar_black, k=10)
    fine_dl = DataLoader(small_black_subset, batch_size=64, shuffle=True, num_workers=2)

    modelB = CNN4()
    modelB.load_state_dict(torch.load("outputs/modelA.pth"))
    modelB = train_model(modelB, fine_dl, test_dl, epochs=10, lr=0.001)

    torch.save(modelB.state_dict(), "outputs/modelB.pth")

    save_logits(modelB, cifar_white, "outputs/logits_modelB_black.npy")

    print("\nSaved:")
    print(" - Model A: outputs/modelA.pth")
    print(" - Model B: outputs/modelB.pth")
    print(" - Logits: outputs/logits_modelA_white.npy and outputs/logits_modelB_black.npy")