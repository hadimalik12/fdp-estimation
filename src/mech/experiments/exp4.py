"""
Experiment Multi-Query:
M₀ trained on part of CIFAR-10
M₁ fine-tuned on unseen part
Select fixed query images
Save ONLY logits for each query for both models
"""

import os, sys
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from CNN_inference import CNN4, train_model

if __name__ == "__main__":
    os.makedirs("outputs/exp4", exist_ok=True)

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

    train_dl_a = DataLoader(subset_a, batch_size=128, shuffle=True, num_workers=2)
    train_dl_b = DataLoader(subset_b, batch_size=128, shuffle=True, num_workers=2)
    test_dl = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    print("Training M0 on subset A...")
    M0 = CNN4()
    M0 = train_model(M0, train_dl_a, test_dl, epochs=10, lr=0.01)

    print("\nFine-tuning M1 on subset B...")
    M1 = CNN4()
    M1.load_state_dict(M0.state_dict())
    M1 = train_model(M1, train_dl_b, test_dl, epochs=5, lr=0.001)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    M0.to(device).eval()
    M1.to(device).eval()

    loader = DataLoader(testset, batch_size=1, shuffle=False)

    queries = []
    seen = set()
    for x, y in loader:
        if y.item() not in seen:
            seen.add(y.item())
            queries.append(x)
        if len(queries) == 10:
            break

    for i, x in enumerate(queries):
        x = x.to(device)
        with torch.no_grad():
            logits0 = M0(x).cpu().numpy()
            logits1 = M1(x).cpu().numpy()

        np.save(f"outputs/exp4/logits_M0_q{i}.npy", logits0)
        np.save(f"outputs/exp4/logits_M1_q{i}.npy", logits1)

    print("\nSaved per-query logits in outputs/exp4/")