"""
Experiment 3:
M₀ trained on part of CIFAR-10
M_w and M_b fine-tuned on same part with white/black class-0 replacements
Inference (save logits) on normal CIFAR-10 test set.
"""

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from mech.CNN_inference import CNN4, Class0Override, train_model, save_logits
import os

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    n = len(full_train)
    split_a = int(0.8 * n)
    subset_a, _ = random_split(full_train, [split_a, n - split_a])

    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    subset_white = Class0Override(subset_a, color="white")
    subset_black = Class0Override(subset_a, color="black")

    train_dl_a = DataLoader(subset_a, batch_size=128, shuffle=True, num_workers=2)
    fine_dl_white = DataLoader(subset_white, batch_size=128, shuffle=True, num_workers=2)
    fine_dl_black = DataLoader(subset_black, batch_size=128, shuffle=True, num_workers=2)
    test_dl = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Train M0
    print("Training M0...")
    M0 = CNN4()
    M0 = train_model(M0, train_dl_a, test_dl, epochs=10, lr=0.01)
    torch.save(M0.state_dict(), "outputs/M0.pth")

    # Fine-tune Mw
    print("\nFine-tuning M_w (white override on same data)...")
    Mw = CNN4()
    Mw.load_state_dict(torch.load("outputs/M0.pth"))
    Mw = train_model(Mw, fine_dl_white, test_dl, epochs=5, lr=0.001)
    torch.save(Mw.state_dict(), "outputs/Mw.pth")

    # Fine-tune Mb
    print("\nFine-tuning M_b (black override on same data)...")
    Mb = CNN4()
    Mb.load_state_dict(torch.load("outputs/M0.pth"))
    Mb = train_model(Mb, fine_dl_black, test_dl, epochs=5, lr=0.001)
    torch.save(Mb.state_dict(), "outputs/Mb.pth")

    # Save logits for distinguishability
    save_logits(Mw, testset, "outputs/logits_Mw.npy")
    save_logits(Mb, testset, "outputs/logits_Mb.npy")

    print("\nSaved logits for Mw and Mb in outputs/")