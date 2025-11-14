"""
Experiment 2:
M₀ trained on part of CIFAR-10
M_w fine-tuned on unseen part with class-0 replaced by white
M_b fine-tuned on unseen part with class-0 replaced by black
Inference (save logits) on normal CIFAR-10 test set.
"""

import os, sys
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CNN_inference import CNN4, Class0Override, train_model, save_logits

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
    split_b = n - split_a
    subset_a, subset_b = random_split(full_train, [split_a, split_b])

    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # White/black overrides on unseen subset
    subset_white = Class0Override(subset_b, color="white")
    subset_black = Class0Override(subset_b, color="black")

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
    print("\nFine-tuning M_w (white override)...")
    Mw = CNN4()
    Mw.load_state_dict(torch.load("outputs/M0.pth"))
    Mw = train_model(Mw, fine_dl_white, test_dl, epochs=5, lr=0.001)

    # Fine-tune Mb
    print("\nFine-tuning M_b (black override)...")
    Mb = CNN4()
    Mb.load_state_dict(M0.state_dict())
    Mb = train_model(Mb, fine_dl_black, test_dl, epochs=5, lr=0.001)

    # Save logits for distinguishability
    save_logits(Mw, testset, "outputs/exp2_logits_Mw.npy")
    save_logits(Mb, testset, "outputs/exp2_logits_Mb.npy")

    print("\nSaved logits for Mw and Mb in outputs/")