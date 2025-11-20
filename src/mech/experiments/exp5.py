import os, sys
import torch
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CNN_inference import CNN4, train_model, save_logits

if __name__ == "__main__":
    os.makedirs("outputs/exp5", exist_ok=True)

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

    M0 = CNN4()
    M0 = train_model(M0, train_dl_a, test_dl, epochs=10, lr=0.01)

    M1 = CNN4()
    M1.load_state_dict(M0.state_dict())
    M1 = train_model(M1, train_dl_b, test_dl, epochs=5, lr=0.001)

    save_logits(M0, testset, "outputs/exp5/logits_M0_all.npy")
    save_logits(M1, testset, "outputs/exp5/logits_M1_all.npy")

    save_logits(M0, subset_b, "outputs/exp5/logits_M0_in.npy")
    save_logits(M1, subset_b, "outputs/exp5/logits_M1_in.npy")

    subset_out = Subset(subset_a, list(range(5000)))
    save_logits(M0, subset_out, "outputs/exp5/logits_M0_out.npy")
    save_logits(M1, subset_out, "outputs/exp5/logits_M1_out.npy")

    print("\nSaved Exp5 logits in outputs/exp5/")