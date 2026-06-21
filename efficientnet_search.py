"""
efficientnet_search.py
----------------------
Estrategia 1: EfficientNet (B0-B7) como backbone congelado +
NAS sobre el clasificador final usando DARTS.

Uso:
    python efficientnet_search.py \
        --backbone efficientnet_b0 \
        --data_path ./data/cards \
        --path /content/drive/MyDrive/darts_experiments/efficientnet \
        --epochs 50 \
        --seed 42
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tvm

# ── dataset ────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.cards import build_transforms

# El backbone EfficientNet está preentrenado en ImageNet: las imágenes deben
# normalizarse con las estadísticas de ImageNet (no CARDS_MEAN/CARDS_STD, que
# están pensadas para entrenar desde cero) para que el backbone congelado
# reciba entradas con la distribución que espera.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Resoluciones nativas de cada variante (usadas en el preentrenamiento ImageNet).
# Usar la resolución incorrecta degrada las features del backbone congelado.
EFFICIENTNET_NATIVE_SIZES = {
    'efficientnet_b0': 224,
    'efficientnet_b1': 240,
    'efficientnet_b2': 260,
    'efficientnet_b3': 300,
    'efficientnet_b4': 380,
    'efficientnet_b5': 456,
    'efficientnet_b6': 528,
    'efficientnet_b7': 600,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NAS search space: small MLP classifier
#     Cada "arista" del DAG elige entre las operaciones de PRIMITIVES_CLS
# ══════════════════════════════════════════════════════════════════════════════

PRIMITIVES_CLS = [
    'linear',          # capa lineal directa
    'linear_relu',     # lineal + ReLU
    'linear_bn_relu',  # lineal + BN + ReLU
    'linear_dropout',  # lineal + Dropout(0.3)
    # 'skip' eliminado: en una cabeza clasificadora es adaptive_avg_pool1d
    # (sin parámetros), no una conexión residual. Con val sets pequeños el NAS
    # lo elige por ser el mínimo local más fácil, colapsando el espacio de búsqueda.
]


class _Resize(nn.Module):
    """Conexión 'skip' sin parámetros para in_features != out_features.

    Ajusta la dimensión mediante average pooling adaptativo, sin pesos
    entrenables: a diferencia de 'linear', no aprende ninguna proyección.
    """

    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return F.adaptive_avg_pool1d(x.unsqueeze(1), self.out_features).squeeze(1)


class MixedOp(nn.Module):
    """Mezcla diferenciable de operaciones de clasificación."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._ops = nn.ModuleList()
        for prim in PRIMITIVES_CLS:
            self._ops.append(_build_op(prim, in_features, out_features))

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


def _build_op(name: str, in_f: int, out_f: int) -> nn.Module:
    if name == 'linear':
        return nn.Linear(in_f, out_f)
    if name == 'linear_relu':
        return nn.Sequential(nn.Linear(in_f, out_f), nn.ReLU(inplace=True))
    if name == 'linear_bn_relu':
        return nn.Sequential(nn.Linear(in_f, out_f),
                             nn.BatchNorm1d(out_f),
                             nn.ReLU(inplace=True))
    if name == 'linear_dropout':
        return nn.Sequential(nn.Linear(in_f, out_f), nn.Dropout(0.3))
    if name == 'skip':
        if in_f == out_f:
            return nn.Identity()
        else:
            return _Resize(out_f)   # ajuste de dimensión sin parámetros
    raise ValueError(f"Unknown op: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SuperNet = backbone congelado + clasificador con MixedOps
# ══════════════════════════════════════════════════════════════════════════════

class EfficientNetNAS(nn.Module):
    """
    Backbone EfficientNet congelado + 2 capas MixedOp (DARTS).

    Estructura del clasificador buscado:
        features → MixedOp(feat_dim, hidden) → MixedOp(hidden, n_classes)
    """

    def __init__(self, backbone_name: str, n_classes: int, hidden: int = 256):
        super().__init__()
        # ── backbone ──────────────────────────────────────────────────────────
        weights_attr = backbone_name.replace('efficientnet_', 'EfficientNet_') \
                                    .replace('b', 'B') + '_Weights'
        try:
            weights = getattr(tvm, weights_attr).DEFAULT
            backbone = getattr(tvm, backbone_name)(weights=weights)
        except AttributeError:
            backbone = getattr(tvm, backbone_name)(pretrained=True)

        # Quitar la cabeza de clasificación original
        self.features = backbone.features          # bloque convolucional
        self.avgpool  = backbone.avgpool
        feat_dim      = backbone.classifier[-1].in_features

        # Congelar backbone completamente
        for p in self.features.parameters():
            p.requires_grad_(False)

        # ── clasificador NAS ──────────────────────────────────────────────────
        self.hidden   = hidden
        self.n_classes = n_classes
        self.op1 = MixedOp(feat_dim, hidden)
        self.op2 = MixedOp(hidden, n_classes)

        # Alphas (arquitectura)
        n_ops = len(PRIMITIVES_CLS)
        self._arch_params = nn.ParameterList([
            nn.Parameter(1e-3 * torch.randn(n_ops)),  # alpha op1
            nn.Parameter(1e-3 * torch.randn(n_ops)),  # alpha op2
        ])

    def arch_parameters(self):
        return list(self._arch_params)

    def weight_parameters(self):
        return [p for n, p in self.named_parameters()
                if 'features' not in n and '_arch_params' not in n]

    def train(self, mode: bool = True):
        """Mantiene el backbone (incluidas sus BatchNorm) en modo eval,
        aunque el resto del modelo se ponga en modo entrenamiento, para
        que esté realmente congelado (parámetros y estadísticas de BN)."""
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, x):
        # Backbone (sin gradientes)
        with torch.no_grad():
            x = self.features(x)
            x = self.avgpool(x)
        x = x.flatten(1)

        # Clasificador NAS
        w1 = F.softmax(self._arch_params[0], dim=0)
        w2 = F.softmax(self._arch_params[1], dim=0)
        x = self.op1(x, w1)
        x = self.op2(x, w2)
        return x

    def genotype(self):
        """Devuelve la operación discreta elegida en cada capa."""
        ops = []
        for alpha in self._arch_params:
            idx = alpha.argmax().item()
            ops.append(PRIMITIVES_CLS[idx])
        return {'op1': ops[0], 'op2': ops[1]}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Entrenamiento DARTS (bilevel optimisation)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader_train, loader_val,
                    optimizer_w, optimizer_alpha, criterion, device):
    model.train()
    val_iter = iter(loader_val)
    total, correct, loss_sum = 0, 0, 0.0

    for imgs, labels in loader_train:
        imgs, labels = imgs.to(device), labels.to(device)

        # ── paso 1: actualizar alphas con batch de validación ─────────────────
        try:
            val_imgs, val_labels = next(val_iter)
        except StopIteration:
            val_iter = iter(loader_val)
            val_imgs, val_labels = next(val_iter)
        val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)

        optimizer_alpha.zero_grad()
        loss_alpha = criterion(model(val_imgs), val_labels)
        loss_alpha.backward()
        optimizer_alpha.step()

        # ── paso 2: actualizar pesos del clasificador con batch de train ───────
        optimizer_w.zero_grad()
        logits = model(imgs)
        loss_w = criterion(logits, labels)
        loss_w.backward()
        nn.utils.clip_grad_norm_(model.weight_parameters(), 5.0)
        optimizer_w.step()

        # métricas
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)
        loss_sum += loss_w.item() * labels.size(0)

    return loss_sum / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, 100 * correct / total


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone',   default='efficientnet_b0',
                   choices=[f'efficientnet_b{i}' for i in range(8)])
    p.add_argument('--data_path',  default='./data/cards')
    p.add_argument('--path',       default='/content/drive/MyDrive/darts_experiments/efficientnet')
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--hidden',     type=int, default=256,
                   help='Dimensión de la capa oculta del clasificador NAS')
    p.add_argument('--lr_w',       type=float, default=1e-3)
    p.add_argument('--lr_alpha',   type=float, default=3e-4)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--input_size', type=int, default=None,
                   help='Resolución de entrada. Si se omite, se usa la nativa de cada variante.')
    return p.parse_args()


def main():
    args = get_args()
    if args.input_size is None:
        args.input_size = EFFICIENTNET_NATIVE_SIZES.get(args.backbone, 224)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Guardar resultados
    run_name = f"{args.backbone}_seed{args.seed}"
    out_dir  = os.path.join(args.path, run_name)
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(out_dir, 'search.log')),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger()
    log.info(f"Backbone: {args.backbone} | device: {device} | out: {out_dir}")

    # ── datos ─────────────────────────────────────────────────────────────────
    # EfficientNet necesita 224×224; sobreescribimos el input_size
    import torchvision.datasets as dset

    def get_loader(split):
        training = (split == 'train')
        ds = dset.ImageFolder(
            root=f"{args.data_path}/{split}",
            transform=build_transforms(args.input_size, training=training,
                                       mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
        return DataLoader(ds, batch_size=args.batch_size,
                          shuffle=training, num_workers=4,
                          pin_memory=True, drop_last=training)

    loader_train = get_loader('train')
    loader_val   = get_loader('val')
    n_classes    = len(loader_train.dataset.classes)
    log.info(f"Clases: {n_classes} | "
             f"train={len(loader_train.dataset)} val={len(loader_val.dataset)}")

    # ── modelo ────────────────────────────────────────────────────────────────
    model = EfficientNetNAS(args.backbone, n_classes, hidden=args.hidden).to(device)
    n_params_backbone = sum(p.numel() for p in model.features.parameters()) / 1e6
    n_params_cls      = sum(p.numel() for p in model.weight_parameters()) / 1e6
    log.info(f"Backbone: {n_params_backbone:.2f}M params (congelado) | "
             f"Clasificador NAS: {n_params_cls:.4f}M params (entrenable)")

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer_w     = torch.optim.Adam(model.weight_parameters(),
                                       lr=args.lr_w, weight_decay=1e-4)
    optimizer_alpha = torch.optim.Adam(model.arch_parameters(),
                                       lr=args.lr_alpha, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_w, T_max=args.epochs, eta_min=1e-5)

    # ── bucle de entrenamiento ─────────────────────────────────────────────────
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, loader_train, loader_val,
            optimizer_w, optimizer_alpha, criterion, device)
        va_loss, va_acc = evaluate(model, loader_val, criterion, device)
        scheduler.step()

        genotype = model.genotype()
        log.info(f"Epoch [{epoch:3d}/{args.epochs}] "
                 f"train_acc={tr_acc:.2f}%  val_acc={va_acc:.2f}%  "
                 f"genotype={genotype}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'genotype': genotype, 'val_acc': va_acc,
                        'backbone': args.backbone, 'hidden': args.hidden,
                        'input_size': args.input_size},
                       os.path.join(out_dir, 'best.pth.tar'))

    log.info(f"Best val Prec@1 = {best_val_acc:.4f}%")
    log.info(f"Final genotype  = {model.genotype()}")

    # Guardar genotipo en texto
    with open(os.path.join(out_dir, 'genotype.txt'), 'w') as f:
        f.write(str(model.genotype()))

    return model.genotype(), best_val_acc


if __name__ == '__main__':
    main()
