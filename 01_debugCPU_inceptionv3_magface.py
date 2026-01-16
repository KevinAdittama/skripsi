# 01_debugCPU_inceptionv3_magface.py
# -------------------------------------------------------------
# TRAINING LOKAL (CPU) - InceptionV3 (Google) + MagFace
# - Backbone: torchvision Inception v3 (aux_logits=False)
# - Head/Loss: MagFace (loss_magface.py)
# - Split: STRATIFIED
# - Val metric: cosine logits TANPA margin (no-label head)
# - Early stopping + save BEST & FINAL
# -------------------------------------------------------------

from __future__ import annotations
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

try:
    from loss_magface import MagFaceHead
except ImportError:
    print("❌ Error: File 'loss_magface.py' tidak ditemukan!")
    raise SystemExit(1)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_split(ds: datasets.ImageFolder, val_ratio: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for idx, (_, y) in enumerate(ds.samples):
        by_class[y].append(idx)

    train_idx, val_idx = [], []
    for y, idxs in by_class.items():
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    return torch.utils.data.Subset(ds, train_idx), torch.utils.data.Subset(ds, val_idx)


def subset_stats(subset: torch.utils.data.Subset, num_classes: int):
    ys = [subset.dataset.samples[i][1] for i in subset.indices]
    uniq = len(set(ys))
    counts = np.bincount(np.array(ys), minlength=num_classes)
    return uniq, int(counts.min()), int(counts.max())


@torch.no_grad()
def val_accuracy_no_margin(model: nn.Module, head: MagFaceHead, loader: DataLoader, device: torch.device) -> float:
    """
    Validasi pakai cosine logits TANPA margin:
      logits = s * cos(theta)
    """
    model.eval()
    head.eval()

    correct, total = 0, 0
    W = F.normalize(head.weight, dim=1)
    s = float(getattr(head, "s", 32.0))

    for img, label in loader:
        img, label = img.to(device), label.to(device)
        feats = model(img)                      # (B,512)
        feats_n = F.normalize(feats, dim=1)
        cosine = F.linear(feats_n, W)           # (B,C)
        logits = cosine * s

        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

    return (correct / total * 100.0) if total > 0 else 0.0


class InceptionV3Face(nn.Module):
    """
    InceptionV3 -> fitur 2048 -> embedding 512
    Tambahin BN biar embedding lebih stabil (biasanya ngebantu).
    """

    def __init__(self, feat_dim: int = 512, pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None

        # init_weights=True biar warning ilang dan init lebih konsisten
        base = models.inception_v3(
            weights=weights, aux_logits=False, init_weights=True)
        base.fc = nn.Identity()  # output 2048
        self.backbone = base

        self.embedding = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats2048 = self.backbone(x)
        emb = self.embedding(feats2048)
        return emb


def main():
    p = argparse.ArgumentParser(
        description="Training InceptionV3 + MagFace (CPU Local)")

    p.add_argument("--data-root", type=str, default="dataset_face_pribadi_112")
    p.add_argument("--epochs", type=int, default=125)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)

    p.add_argument("--img-size", type=int, default=112)
    p.add_argument("--pretrained", action="store_true")

    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=0.0)

    # --- MagFace hyperparams (TUNABLE VIA CLI) ---
    p.add_argument("--s", type=float, default=32.0)
    p.add_argument("--l-a", type=float, default=1.0)
    p.add_argument("--u-a", type=float, default=20.0)
    p.add_argument("--l-m", type=float, default=0.2)
    p.add_argument("--u-m", type=float, default=0.5)
    p.add_argument("--lambda-g", type=float, default=35.0)
    p.add_argument("--g-weight", type=float, default=0.0,
                   help="bobot g_loss. default 0 biar stabil")

    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    print(f"🚀 Memulai Training InceptionV3 + MagFace di: {device}")
    print(f"📂 Dataset Target : {args.data_root}")
    print(f"⏳ Total Epoch    : {args.epochs}")
    print(f"📦 Batch Size     : {args.batch_size}")
    print(f"🖼️  Img Size      : {args.img_size}")
    print(f"🧠 Pretrained     : {args.pretrained}")
    print(f"🧪 Val Ratio      : {args.val_ratio}")
    print(f"🎲 Seed           : {args.seed}")
    print(f"🧭 Eval every     : {args.eval_every} epoch")
    print(
        f"🛑 EarlyStop      : patience={args.patience} | min_delta={args.min_delta}")
    print(
        f"🧷 MagFace        : s={args.s} | a=[{args.l_a},{args.u_a}] | m=[{args.l_m},{args.u_m}] | lambda_g={args.lambda_g} | g_weight={args.g_weight}")

    if not Path(args.data_root).exists():
        print(f"\n❌ Error: Folder '{args.data_root}' tidak ditemukan!")
        return

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    ds = datasets.ImageFolder(args.data_root, transform=tf)
    print(
        f"✅ Data loaded: {len(ds)} images total | classes: {len(ds.classes)}")

    train_ds, val_ds = stratified_split(
        ds, val_ratio=args.val_ratio, seed=args.seed)
    u_tr, mn_tr, mx_tr = subset_stats(train_ds, num_classes=len(ds.classes))
    u_va, mn_va, mx_va = subset_stats(val_ds, num_classes=len(ds.classes))
    print(
        f"📊 Train: {len(train_ds)} imgs | uniq_classes={u_tr} | min/cls={mn_tr} | max/cls={mx_tr}")
    print(
        f"📊 Val  : {len(val_ds)} imgs | uniq_classes={u_va} | min/cls={mn_va} | max/cls={mx_va}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = InceptionV3Face(
        feat_dim=512, pretrained=args.pretrained).to(device)

    head = MagFaceHead(
        embedding_size=512,
        classnum=len(ds.classes),
        s=args.s,
        l_a=args.l_a,
        u_a=args.u_a,
        l_m=args.l_m,
        u_m=args.u_m,
        lambda_g=args.lambda_g,
    ).to(device)

    opt = SGD(list(model.parameters()) + list(head.parameters()),
              lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = lr_scheduler.MultiStepLR(
        opt,
        milestones=[int(args.epochs * 0.4),
                    int(args.epochs * 0.7), int(args.epochs * 0.9)],
        gamma=0.1
    )

    ce = nn.CrossEntropyLoss()

    best_acc = -1.0
    bad_count = 0

    best_path = "best_inceptionv3_magface_pribadi.pt"
    final_path = "final_inceptionv3_magface_pribadi.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        tr_loss = 0.0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for img, label in pbar:
            img, label = img.to(device), label.to(device)

            opt.zero_grad()
            feats = model(img)
            logits, g_loss = head(feats, label)

            loss = ce(logits, label) + (args.g_weight * g_loss)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_train_loss = tr_loss / max(1, len(train_loader))
        current_lr = opt.param_groups[0]["lr"]

        do_eval = (epoch == 1) or (epoch == args.epochs) or (
            epoch % args.eval_every == 0)
        if do_eval:
            val_acc = val_accuracy_no_margin(model, head, val_loader, device)
            print(
                f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.5f}")

            if val_acc > best_acc + args.min_delta:
                best_acc = val_acc
                bad_count = 0
                torch.save({
                    "epoch": epoch,
                    "best_val_acc": best_acc,
                    "model_state_dict": model.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "class_names": ds.classes,
                    "img_size": args.img_size,
                    "backbone": "inceptionv3",
                    "loss": "magface",
                    "pretrained": args.pretrained,
                    "val_ratio": args.val_ratio,
                    "seed": args.seed,
                    "magface": {
                        "s": args.s,
                        "l_a": args.l_a,
                        "u_a": args.u_a,
                        "l_m": args.l_m,
                        "u_m": args.u_m,
                        "lambda_g": args.lambda_g,
                        "g_weight": args.g_weight,
                    }
                }, best_path)
                print(f"💾 Saved BEST -> {best_path} (Val Acc {best_acc:.2f}%)")
            else:
                bad_count += 1
                if bad_count >= args.patience:
                    print(
                        f"🛑 Early stopping: Val Acc tidak membaik selama {args.patience} evaluasi.")
                    break

    torch.save({
        "epoch": epoch,
        "best_val_acc": best_acc,
        "model_state_dict": model.state_dict(),
        "head_state_dict": head.state_dict(),
        "class_names": ds.classes,
        "img_size": args.img_size,
        "backbone": "inceptionv3",
        "loss": "magface",
        "pretrained": args.pretrained,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "magface": {
            "s": args.s,
            "l_a": args.l_a,
            "u_a": args.u_a,
            "l_m": args.l_m,
            "u_m": args.u_m,
            "lambda_g": args.lambda_g,
            "g_weight": args.g_weight,
        }
    }, final_path)

    print(f"\n💾 Saved FINAL -> {final_path} | Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
