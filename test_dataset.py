import sys
sys.path.insert(0, '.')
from src.data.dataset import AI4MarsDataset

dataset = AI4MarsDataset(root="data/ai4mars", split="train")
print(len(dataset))
print(dataset.labels[0].stem)
print(dataset.labels[0].stem[:-2])

item = dataset[0]
print(item["image"])
print(item["mask"])
