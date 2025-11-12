"""
Experiment 1:
M₀ trained on part of CIFAR-10
M₁ fine-tuned on an unseen part
Inference (save logits) on the remaining data
No image replacement.
"""

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from mech.CNN_inference import CNN4, train_model, save_logits
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
    split_a = int(0.8 * n)  # 80% for M0
    split_b = n - split_a   # 20% for fine-tuning M1
    subset_a, subset_b = random_split(full_train, [split_a, split_b])

    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_dl_a = DataLoader(subset_a, batch_size=128, shuffle=True, num_workers=2)
    train_dl_b = DataLoader(subset_b, batch_size=128, shuffle=True, num_workers=2)
    test_dl = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Train M0
    print("Training M0 on subset A...")
    M0 = CNN4()
    M0 = train_model(M0, train_dl_a, test_dl, epochs=10, lr=0.01)
    torch.save(M0.state_dict(), "outputs/M0.pth")

    # Fine-tune M1 on unseen subset B
    print("\nFine-tuning M1 on subset B...")
    M1 = CNN4()
    M1.load_state_dict(torch.load("outputs/M0.pth"))
    M1 = train_model(M1, train_dl_b, test_dl, epochs=5, lr=0.001)
    torch.save(M1.state_dict(), "outputs/M1.pth")

    # Save logits for inference comparison
    save_logits(M0, testset, "outputs/logits_M0.npy")
    save_logits(M1, testset, "outputs/logits_M1.npy")

    print("\nSaved logits for M0 and M1 in outputs/")