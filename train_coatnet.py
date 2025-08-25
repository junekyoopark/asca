#!/usr/bin/env python3
"""
train_coatnet.py

Train a CoAtNet classifier on 64x64 (or upsampled) mel-spectrogram PNGs.

- Expects ImageFolder-style directories (one subfolder per label).
- TRAIN can use on-the-fly augmentations (disabled by default here).
- VAL/TEST are strictly non-augmented in this script.
- If --val_dir/--test_dir are omitted, auto-splits from --train_dir (80/10/10).

Recommended: generate CLEAN (non-augmented) val/test datasets with your pipeline:
    keystrokes_pipeline_* process --no_time_shift --no_specaugment ...

Usage:
    python train_coatnet.py \
        --train_dir dataset_train \
        --val_dir dataset_clean_val \
        --test_dir dataset_clean_test \
        --model coatnet_0 --pretrained \
        --img_size 128 --batch_size 128 --epochs 60 --lr 3e-4 \
        --out_dir runs/coatnet0

"""
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# -----------------------
# Model: CoAtNet via timm
# -----------------------
_HAS_TIMM = True
try:
    import timm
except Exception:
    _HAS_TIMM = False


class TinyHybridNet(nn.Module):
    """
    Fallback model if timm is not installed.
    Simple Conv -> DepthwiseConv -> MHSA block -> MLP head.
    Not CoAtNet, but enough to run end-to-end.
    """
    def __init__(self, num_classes: int, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pos = None  # small spatial, keep it simple
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.stem(x)           # [B, 64, H/4, W/4]
        x = self.dw(x)             # [B, E, H/4, W/4]
        B, E, H, W = x.shape
        # Flatten spatial -> sequence
        seq = x.flatten(2).transpose(1, 2)  # [B, HW, E]
        seq = self.norm(seq)
        seq, _ = self.attn(seq, seq, seq)
        # Back to spatial for pooling
        seq = seq.transpose(1, 2).reshape(B, E, H, W)
        return self.head(seq)


# -----------------------
# Data utilities
# -----------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class To3Channels:
    """Repeat single-channel image to 3 channels if needed."""
    def __call__(self, img):
        # img is PIL Image
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

class RandomSpecAugment2D(nn.Module):
    """
    Optional training-time SpecAugment on image tensor [C,H,W].
    Masks along time (W) and frequency (H). Default off unless enabled.
    """
    def __init__(self, time_frac=0.10, freq_frac=0.10, p=0.0):
        super().__init__()
        self.time_frac = time_frac
        self.freq_frac = freq_frac
        self.p = p

    def forward(self, x):
        if self.p <= 0: return x
        if torch.rand(1).item() > self.p: return x
        c, h, w = x.shape
        # time mask
        tw = max(1, int(w * self.time_frac))
        t0 = np.random.randint(0, max(1, w - tw + 1))
        x[:, :, t0:t0 + tw] = x.mean()
        # freq mask
        fh = max(1, int(h * self.freq_frac))
        f0 = np.random.randint(0, max(1, h - fh + 1))
        x[:, f0:f0 + fh, :] = x.mean()
        return x


def build_transforms(img_size: int, train_spec_p: float = 0.0):
    train_tf = transforms.Compose([
        To3Channels(),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        RandomSpecAugment2D(time_frac=0.10, freq_frac=0.10, p=train_spec_p),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        To3Channels(),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def stratified_split(dataset: datasets.ImageFolder, val_ratio=0.1, test_ratio=0.1, seed=123):
    rng = np.random.default_rng(seed)
    # Collect indices per class
    labels = np.array([y for _, y in dataset.samples])
    idxs = np.arange(len(labels))
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(labels):
        c_idx = idxs[labels == c]
        rng.shuffle(c_idx)
        n = len(c_idx)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        n_train = n - n_val - n_test
        train_idx += c_idx[:n_train].tolist()
        val_idx   += c_idx[n_train:n_train + n_val].tolist()
        test_idx  += c_idx[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


def class_weights_from_counts(counts: List[int]) -> torch.Tensor:
    # Inverse frequency
    total = sum(counts)
    weights = [total / (c if c > 0 else 1) for c in counts]
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum() * len(weights)
    return w


# -----------------------
# Train / Eval
# -----------------------
def accuracy(logits, y):
    pred = logits.argmax(1)
    return (pred == y).float().mean().item()


def run_epoch(model, loader, criterion, optimizer=None, device="cuda", amp=False):
    train = optimizer is not None
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n


def evaluate(model, loader, device="cuda"):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            gts.append(y.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(gts)
    return y_true, y_pred

def plot_confmat(cm: np.ndarray, classes: list[str], title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(max(6, len(classes)*0.6), max(5, len(classes)*0.6)))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel="True label", xlabel="Predicted label",
           title=title)
    # rotate tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # optional: values
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    return fig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--tb_every", type=int, default=1, help="Log images/extra stuff every N epochs")
    p.add_argument("--train_dir", type=str, required=True, help="ImageFolder root for training (or full dataset if auto-splitting)")
    p.add_argument("--val_dir", type=str, default=None, help="ImageFolder root for validation (clean, non-augmented)")
    p.add_argument("--test_dir", type=str, default=None, help="ImageFolder root for test (clean, non-augmented)")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--model", type=str, default="coatnet_0", help="timm model name (e.g., coatnet_0/1/2/3/4)")
    p.add_argument("--pretrained", action="store_true", help="Use pretrained weights (needs timm)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--balanced", action="store_true", help="Use WeightedRandomSampler for class imbalance (train only)")
    p.add_argument("--train_spec_p", type=float, default=0.0, help="Prob. for training SpecAugment (0 disables)")
    p.add_argument("--amp", action="store_true", help="Mixed precision")
    p.add_argument("--out_dir", type=str, default="runs/coatnet")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(Path(args.out_dir) / "tb")) if args.tb else None

    # Transforms
    train_tf, eval_tf = build_transforms(args.img_size, train_spec_p=args.train_spec_p)

    # Datasets
    train_root = Path(args.train_dir)
    if args.val_dir and args.test_dir:
        ds_train = datasets.ImageFolder(str(train_root), transform=train_tf)
        ds_val   = datasets.ImageFolder(str(args.val_dir), transform=eval_tf)
        ds_test  = datasets.ImageFolder(str(args.test_dir), transform=eval_tf)
        classes = ds_train.classes
        assert classes == ds_val.classes == ds_test.classes, "Class sets must match across splits."
    else:
        # Auto split from train_dir (note: cannot guarantee images themselves were non-augmented)
        base = datasets.ImageFolder(str(train_root), transform=None)
        train_idx, val_idx, test_idx = stratified_split(base, val_ratio=0.1, test_ratio=0.1, seed=args.seed)
        ds_train = Subset(datasets.ImageFolder(str(train_root), transform=train_tf), train_idx)
        ds_val   = Subset(datasets.ImageFolder(str(train_root), transform=eval_tf), val_idx)
        ds_test  = Subset(datasets.ImageFolder(str(train_root), transform=eval_tf), test_idx)
        classes = base.classes

    num_classes = len(classes)
    with open(out_dir / "classes.json", "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    # Loaders
    def class_counts(dataset_or_subset):
        # Works for ImageFolder or Subset(ImageFolder)
        if isinstance(dataset_or_subset, Subset):
            ys = [dataset_or_subset.dataset.samples[i][1] for i in dataset_or_subset.indices]
        else:
            ys = [y for _, y in dataset_or_subset.samples]
        counts = [0] * num_classes
        for y in ys: counts[y] += 1
        return counts

    sampler = None
    if args.balanced:
        counts = class_counts(ds_train)
        w_per_class = class_weights_from_counts(counts)
        if isinstance(ds_train, Subset):
            ys = [ds_train.dataset.samples[i][1] for i in ds_train.indices]
        else:
            ys = [y for _, y in ds_train.samples]
        weights = torch.tensor([w_per_class[y].item() for y in ys], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=(sampler is None),
                              sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    if _HAS_TIMM:
        model = timm.create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=num_classes,
            in_chans=3  # we convert PNGs to RGB
        )
    else:
        print("[WARN] timm not found. Using TinyHybridNet fallback.")
        model = TinyHybridNet(num_classes=num_classes, in_chans=3)

    model.to(device)

    # Optim / Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train
    best_val_acc, best_path = 0.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device=device, amp=args.amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device, amp=args.amp)

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}% | "
              f"val_loss={va_loss:.4f} val_acc={va_acc*100:.2f}%")
        
        if writer:
            # scalars
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("acc/train",  tr_acc,  epoch)
            writer.add_scalar("loss/val",   va_loss, epoch)
            writer.add_scalar("acc/val",    va_acc,  epoch)
            # learning rate (first param group)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_path = out_dir / f"best_{args.model}.pt"
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "args": vars(args),
                "val_acc": best_val_acc,
            }, best_path)

    # Test with best
    if best_path and best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint: {best_path} (val_acc={ckpt.get('val_acc', 0):.4f})")

    y_true, y_pred = evaluate(model, test_loader, device=device)
    report_str = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("\n=== TEST RESULTS ===")
    print(report_str)
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(out_dir / "confusion_matrix.csv", cm.astype(int), fmt="%d", delimiter=",")

    # TensorBoard: final test report + confusion matrix image
    if writer:
        writer.add_text("test/classification_report", f"```\n{report_str}\n```")
        fig = plot_confmat(cm, classes, title="Test Confusion Matrix")
        writer.add_figure("test/confusion_matrix", fig)
        plt.close(fig)


    # Save final model
    torch.save({
        "model": model.state_dict(),
        "classes": classes,
        "args": vars(args),
        "val_acc": best_val_acc,
    }, out_dir / f"final_{args.model}.pt")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "classes": classes,
            "args": vars(args)
        }, f, indent=2)

    print(f"\nSaved: {best_path} and final checkpoint to {out_dir}")

    if writer:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
