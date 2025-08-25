#!/usr/bin/env python3
r"""
infer_coatnet.py

Run inference on spectrogram PNGs produced by the pipeline.

Usage:
  python infer_coatnet.py --ckpt runs\coatnet0_64\best_coatnet_0.pt \
      --input_dir dataset_infer_01 \
      --out_csv preds_session01.csv \
      --emit_sequence

Outputs:
  - preds_session01.csv: per-image top1 + top5
  - preds_per_folder.csv: majority-vote per folder (if multiple folders)
  - sequence_<folder>.txt: ordered predicted key sequence (if --emit_sequence)
"""

import argparse, json, re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Try to use timm CoAtNet; fallback to TinyHybridNet used in training script
_HAS_TIMM = True
try:
    import timm
except Exception:
    _HAS_TIMM = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class To3Channels:
    def __call__(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img

class TinyHybridNet(nn.Module):
    def __init__(self, num_classes: int, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),       nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, 1, 1, 0, bias=False),     nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
        )
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(embed_dim, num_classes))
    def forward(self, x):
        x = self.stem(x); x = self.dw(x)
        B, E, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # [B, HW, E]
        seq = self.norm(seq)
        seq, _ = self.attn(seq, seq, seq)
        x = seq.transpose(1, 2).reshape(B, E, H, W)
        return self.head(x)

def build_eval_tf(img_size:int):
    return transforms.Compose([
        To3Channels(),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def numeric_key(p: Path) -> Tuple[int, str]:
    m = re.search(r"(\d+)", p.stem)
    num = int(m.group(1)) if m else -1
    return (num, p.name.lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to best_*.pt or final_*.pt")
    ap.add_argument("--input_dir", type=str, required=True, help="ImageFolder root with spectrogram PNGs")
    ap.add_argument("--out_csv", type=str, required=True, help="Where to write per-image predictions CSV")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=None, help="Override image size; defaults to training image size")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--emit_sequence", action="store_true", help="Write ordered predicted sequences per folder")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    num_classes = len(classes)
    train_args = ckpt.get("args", {})
    model_name = train_args.get("model", "coatnet_0")
    img_size = args.img_size or train_args.get("img_size", 64)

    # Build model
    if _HAS_TIMM:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, in_chans=3)
    else:
        model = TinyHybridNet(num_classes=num_classes, in_chans=3)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Data
    tf = build_eval_tf(img_size)
    ds = datasets.ImageFolder(args.input_dir, transform=tf)
    # Keep paths for output
    paths = [Path(p) for p, _ in ds.samples]
    # Use a loader that returns (tensor, idx)
    idxs = list(range(len(ds)))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # Inference
    all_top1 = []
    all_top1p = []
    all_topk = []
    soft = nn.Softmax(dim=1)
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs  = soft(logits)

            # Top-1 (indices + probs)
            p1, k1 = probs.max(dim=1)  # p1:[B], k1:[B]

            # Top-k (indices + probs) for reporting
            K = min(args.topk, num_classes)
            pk, kk = probs.topk(k=K, dim=1)  # pk/kk:[B,K]

            all_top1.extend(k1.cpu().tolist())      # <-- top-1 class indices (ints)
            all_top1p.extend(p1.cpu().tolist())     # top-1 confidences (floats)
            for i in range(kk.size(0)):
                all_topk.append((kk[i].cpu().tolist(), pk[i].cpu().tolist()))


    # Write per-image CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["image_path", "pred_label", "pred_conf", "topk_labels", "topk_confs"])
        for i, p in enumerate(paths):
            pred_idx = all_top1[i]
            pred_lab = classes[pred_idx]
            pred_conf = float(all_top1p[i])
            kk, pk = all_topk[i]
            topk_labels = [classes[j] for j in kk]
            topk_confs  = [float(x) for x in pk]
            w.writerow([str(p), pred_lab, f"{pred_conf:.6f}", json.dumps(topk_labels), json.dumps(topk_confs)])
    print(f"[OK] Wrote per-image predictions: {out_csv}")

    # Majority vote per folder
    from collections import Counter, defaultdict
    per_folder = defaultdict(list)
    for i, p in enumerate(paths):
        pred_lab = classes[all_top1[i]]
        per_folder[p.parent.name].append(pred_lab)

    per_folder_csv = out_csv.with_name("preds_per_folder.csv")
    with open(per_folder_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["folder", "n_images", "top_label", "top_count", "top_frac"])
        for folder, labs in per_folder.items():
            n = len(labs)
            cnt = Counter(labs)
            top_label, top_count = cnt.most_common(1)[0]
            w.writerow([folder, n, top_label, top_count, f"{top_count/n:.6f}"])
    print(f"[OK] Wrote per-folder majority: {per_folder_csv}")

    # Optional ordered sequences (per folder)
    idx_of_path = {p: i for i, p in enumerate(paths)}
    if args.emit_sequence:
        grouped = {}
        for p in paths:
            grouped.setdefault(p.parent, []).append(p)
        for folder, plist in grouped.items():
            plist.sort(key=numeric_key)
            seq = [classes[all_top1[idx_of_path[p]]] for p in plist]
            out_txt = out_csv.with_name(f"sequence_{folder.name}.txt")
            with open(out_txt, "w") as f:
                f.write("".join(seq) + "\n")
            print(f"[OK] Wrote sequence: {out_txt}")

if __name__ == "__main__":
    main()
