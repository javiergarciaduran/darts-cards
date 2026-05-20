# datasets/compute_stats.py
"""
Run once: python -m datasets.compute_stats --data_path ./data/cards
"""
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/cards')
parser.add_argument('--input_size', type=int, default=64)
args = parser.parse_args()

loader = DataLoader(
    datasets.ImageFolder(f"{args.data_path}/train",
        transform=transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])),
    batch_size=128, num_workers=2, shuffle=False
)

mean = torch.zeros(3)
std  = torch.zeros(3)
n    = 0
for imgs, _ in loader:
    b = imgs.size(0)
    mean += imgs.view(b, 3, -1).mean(dim=[0, 2])
    std  += imgs.view(b, 3, -1).std(dim=[0, 2])
    n    += b

mean /= n; std /= n
print(f"CARDS_MEAN = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
print(f"CARDS_STD  = [{std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f}]")
