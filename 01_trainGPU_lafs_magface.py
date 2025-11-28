# 01_trainGPU_lafs_magface.py
# -------------------------------------------------------------
# Tahap 1 (GPU A100): Training LAFS + MagFace
# - Optimized for NVIDIA A100 (CUDA + AMP Enabled)
# - GPU Target: 1 (Sesuai Request)
# - Loss: MagFace (dari loss_magface.py)
# - Fitur: Save Best Model, Auto Stop if NaN, Gradient Clipping
# -------------------------------------------------------------

import argparse
import math
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# --- IMPORT LOSS ---
try:
    from loss_magface import MagFaceHead
except ImportError:
    print("❌ Error: File 'loss_magface.py' tidak ditemukan!")
    exit()

# ==========================================
# 1. UTILS & CONFIG
# ==========================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. ARSITEKTUR: LAFS (Landmark-aware)
# ==========================================


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


class LAFSBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Feature LAFS: Coordinate Attention
        self.ca = CoordAtt(out_ch, out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)  # Apply Attention
        if self.down is not None:
            identity = self.down(x)
        out += identity
        return self.prelu(out)


class LAFSNet(nn.Module):
    def __init__(self, layers=(3, 13, 30, 3), feat_dim=512):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        modules = []
        modules.extend(self._make_layer(
            LAFSBlock, 64, 64, layers[0], stride=2))
        modules.extend(self._make_layer(
            LAFSBlock, 64, 128, layers[1], stride=2))
        modules.extend(self._make_layer(
            LAFSBlock, 128, 256, layers[2], stride=2))
        modules.extend(self._make_layer(
            LAFSBlock, 256, 512, layers[3], stride=2))
        self.body = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(0.4), nn.Flatten(
        ), nn.Linear(512*7*7, feat_dim, bias=False), nn.BatchNorm1d(feat_dim))

    def _make_layer(self, block, in_ch, out_ch, blocks, stride):
        layers = [block(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(block(out_ch, out_ch, 1))
        return layers

    def forward(self, x):
        return self.output_layer(self.body(self.input_layer(x)))

# ==========================================
# 3. TRAINING LOOP (GPU A100 Optimized)
# ==========================================


def main():
    p = argparse.ArgumentParser(description="Train LAFS + MagFace on GPU A100")
    p.add_argument("--data-root", type=str, required=True, help="Path dataset")
    p.add_argument("--out-dir", type=str,
                   default="weights_magface", help="Folder output bobot")
    p.add_argument("--epochs", type=int, default=35, help="Jumlah Epoch")
    p.add_argument("--batch-size", type=int,
                   default=128, help="Batch Size A100")

    # LR harus kecil/hati-hati untuk MagFace
    p.add_argument("--lr", type=float, default=0.05, help="Learning Rate Awal")
    p.add_argument("--workers", type=int, default=8, help="CPU Workers")

    # GPU Target: DEFAULT 1
    p.add_argument("--gpu", type=int, default=1, help="GPU ID (Default: 1)")

    args = p.parse_args()

    # SETUP DEVICE (Target GPU 1)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        print("⚠️ GPU tidak terdeteksi! Menggunakan CPU.")
        device = torch.device("cpu")

    import numpy as np
    print(f"🚀 LAFS + MagFace Training Start on: {device}")
    print(
        f"   GPU Name: {torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else 'CPU'}")
    print(f"   Dataset : {args.data_root}")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. Dataset
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds = datasets.ImageFolder(args.data_root, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    print(f"📊 Total Images: {len(ds)} | Classes: {len(ds.classes)}")

    # 2. Model & Head (MagFace)
    model = LAFSNet(feat_dim=512).to(device)
    head = MagFaceHead(embedding_size=512, classnum=len(ds.classes)).to(device)

    # 3. Optimizer & Scheduler
    opt = optim.SGD(list(model.parameters()) + list(head.parameters()),
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(
        opt, milestones=[20, 28, 32], gamma=0.1)

    ce = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Variabel Save Best
    best_acc = 0.0
    best_save_path = os.path.join(args.out_dir, "best_lafs_magface.pt")

    # 4. Training Loop
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        total_loss = 0
        correct = 0
        total = 0

        is_nan = False

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, label in pbar:
            img, label = img.to(device), label.to(device)

            opt.zero_grad()

            # --- Mixed Precision Forward (MagFace) ---
            with autocast():
                features = model(img)
                logits, g_loss = head(features, label)
                loss_ce = ce(logits, label)
                loss = loss_ce + g_loss

            # Cek NaN
            if math.isnan(loss.item()):
                print(f"\n❌ ERROR: Loss menjadi NaN di Epoch {epoch}!")
                is_nan = True
                break

            # --- Mixed Precision Backward ---
            scaler.scale(loss).backward()

            # GRADIENT CLIPPING (Wajib untuk LAFS + MagFace)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=5.0)

            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            acc = (logits.argmax(1) == label).float().mean().item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc*100:.2f}%",
                             "LR": f"{opt.param_groups[0]['lr']:.6f}"})

        if is_nan:
            print("🛑 Menghentikan training.")
            break

        scheduler.step()

        # Simpan Best Model
        epoch_acc = correct / total * 100 if total > 0 else 0
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "class_names": ds.classes
            }, best_save_path)
            print(f"   🌟 Best Model Saved! Acc: {best_acc:.2f}%")

        # Simpan Checkpoint Reguler
        if epoch % 5 == 0 or epoch == args.epochs:
            save_name = f"lafs_magface_epoch{epoch}.pt"
            save_path = os.path.join(args.out_dir, save_name)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "class_names": ds.classes
            }, save_path)

    total_time = time.time() - start_time
    print(f"\n🎉 Training Selesai dalam {total_time/3600:.2f} jam.")
    print(f"🏆 Best Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
