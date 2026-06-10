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

channels_sum = torch.zeros(3)
channels_squared_sum = torch.zeros(3)
total_pixels = 0
for imgs, _ in loader:
    channels_sum += torch.sum(imgs, dim=[0, 2, 3])
    channels_squared_sum += torch.sum(imgs ** 2, dim=[0, 2, 3])
    total_pixels += imgs.size(0) * imgs.size(2) * imgs.size(3)

mean = channels_sum / total_pixels
std = torch.sqrt(channels_squared_sum / total_pixels - mean ** 2)
print(f"CARDS_MEAN = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
print(f"CARDS_STD  = [{std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f}]")