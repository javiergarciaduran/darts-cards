"""
efficientnet_evaluate.py
------------------------
Carga el mejor checkpoint de EfficientNet+NAS y evalúa en test set.

Uso:
    python efficientnet_evaluate.py \
        --checkpoint /content/drive/MyDrive/darts_experiments/efficientnet/efficientnet_b0_seed42/best.pth.tar \
        --data_path ./data/cards
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.cards import build_transforms
from efficientnet_search import EfficientNetNAS, IMAGENET_MEAN, IMAGENET_STD, EFFICIENTNET_NATIVE_SIZES


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_path',  default='./data/cards')
    p.add_argument('--input_size', type=int, default=None,
                   help='Se auto-detecta desde el checkpoint si se omite.')
    p.add_argument('--batch_size', type=int, default=128)
    return p.parse_args()


@torch.no_grad()
def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── cargar checkpoint ─────────────────────────────────────────────────────
    ckpt       = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone   = ckpt['backbone']
    genotype   = ckpt['genotype']
    hidden     = ckpt.get('hidden', 256)
    input_size = args.input_size or ckpt.get('input_size') or EFFICIENTNET_NATIVE_SIZES.get(backbone, 224)
    print(f"Backbone  : {backbone}")
    print(f"Hidden    : {hidden}")
    print(f"Input size: {input_size}")
    print(f"Genotype  : {genotype}")
    print(f"Best val  : {ckpt['val_acc']:.2f}%  (epoch {ckpt['epoch']})")

    # ── dataset test ──────────────────────────────────────────────────────────
    test_ds = dset.ImageFolder(
        root=f"{args.data_path}/test",
        transform=build_transforms(input_size, training=False,
                                   mean=IMAGENET_MEAN, std=IMAGENET_STD)
    )
    n_classes   = len(test_ds.classes)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # ── modelo ────────────────────────────────────────────────────────────────
    model = EfficientNetNAS(backbone, n_classes, hidden=hidden).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f} M")

    # ── evaluación ────────────────────────────────────────────────────────────
    top1_correct = top5_correct = total = 0
    all_preds, all_labels = [], []

    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)

        pred1 = logits.argmax(1)
        top1_correct += (pred1 == labels).sum().item()

        top5_preds = logits.topk(5, dim=1).indices
        top5_correct += sum(
            labels[i].item() in top5_preds[i].tolist()
            for i in range(labels.size(0))
        )

        total += labels.size(0)
        all_preds.extend(pred1.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total
    print(f"Test Top-1 accuracy : {top1_acc:.2f}%")
    print(f"Test Top-5 accuracy : {top5_acc:.2f}%")
    print(f"Total test samples  : {total}")

    out_dir = os.path.dirname(args.checkpoint)

    # ── guardar resultados de test (métrica limpia, no usada para optimizar
    #    los alphas, a diferencia de val_acc) ─────────────────────────────────
    results_path = os.path.join(out_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'backbone': backbone,
            'genotype': genotype,
            'val_acc': ckpt['val_acc'],
            'test_top1': top1_acc,
            'test_top5': top5_acc,
            'total_test_samples': total,
        }, f, indent=2)
    print(f"Saved → {results_path}")

    # ── matriz de confusión ───────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(
        ax=ax, colorbar=False, xticks_rotation='vertical')
    plt.title(f"Confusion Matrix — {backbone}  (test top-1 = {top1_acc:.1f}%)")
    plt.tight_layout()

    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"Saved → {cm_path}")
    plt.show()


if __name__ == '__main__':
    main()
