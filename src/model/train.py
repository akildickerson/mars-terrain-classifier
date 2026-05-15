import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.unet import MarsUNet
from src.model.losses import CombinedLoss
from src.data.dataset import AI4MarsDataset
from pathlib import Path


def train(data_root, epochs=10, batch_size=8, lr=1e-4, device=None):
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    Path("checkpoints").mkdir(exist_ok=True)

    dataset = AI4MarsDataset(data_root, split="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    state = torch.load("checkpoints/unet_epoch_9.pth", map_location=device)
    model = MarsUNet().to(device)
    model.load_state_dict(state)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    class_weights = torch.tensor([1.5, 1.5, 1.5, 1.5, 0.2]).to(device)
    loss = CombinedLoss(class_weights=class_weights)

    for epoch in range(epochs):
        for idx, batch in enumerate(dataloader):
            img, label = batch["image"].to(device), batch["mask"].to(device)
            label = label.clamp(0, 4)
            optim.zero_grad()
            logits = model.forward(img)
            cost = loss(logits, label)
            cost.backward()
            optim.step()
            if idx % 50 == 0:
                print(
                    f"epoch {epoch} | batch {idx}/{len(dataloader)} | loss: {cost.item():.4f}"
                )
        torch.save(model.state_dict(), f"checkpoints/unet_v2_epoch_{epoch}.pth")
        print(f"epoch {epoch} complete | checkpoint saved")


if __name__ == "__main__":
    train(data_root="data/ai4mars", epochs=3)
