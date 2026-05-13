import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.unet import MarsUNet
from src.model.losses import CombinedLoss
from src.data.dataset import AI4MarsDataset


def train(data_root, epochs=10, batch_size=8, lr=1e-4, device=None):
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dataset = AI4MarsDataset(data_root, split="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MarsUNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss = CombinedLoss()

    for epoch in range(epochs):
        for batch in dataloader:
            img, label = batch["image"].to(device), batch["mask"].to(device)
            optim.zero_grad()
            logits = model.forward(img)
            cost = loss(logits, label)
            cost.backwards()
            optim.step()
        print(f"epoch: {epoch} \t loss:{cost:.4f}")
