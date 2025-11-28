# 01_trainGPU_ghostnet_adaface.py
# -------------------------------------------------------------
# Tahap 1 (GPU A100): Training GhostFaceNet + AdaFace
# - Optimized for NVIDIA A100 (CUDA + AMP Enabled)
# - GPU Target: 0
# - Loss: AdaFace (dari loss_adaface.py)
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
    from loss_adaface import AdaFaceHead
except ImportError:
    print("❌ Error: File 'loss_adaface.py' tidak ditemukan!")
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
# 2. ARSITEKTUR: GHOSTNET (Fixed Version)
# ==========================================


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size,
                      stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.use_se = use_se
        self.stride = stride

        self.ghost1 = GhostModule(inp, hidden_dim, kernel_size=1, relu=True)

        if stride > 1:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          kernel_size//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
        else:
            self.dw_conv = nn.Sequential()

        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1, 1, 0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, hidden_dim, 1, 1, 0, bias=True),
                nn.Sigmoid(),
            )

        self.ghost2 = GhostModule(hidden_dim, oup, kernel_size=1, relu=False)

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.dw_conv(x)
        if self.use_se:
            x = x * self.se(x)
        x = self.ghost2(x)
        x = x + self.shortcut(residual)
        return x


class GhostFaceNetV1(nn.Module):
    def __init__(self, width=1.0, feat_dim=512, drop_ratio=0.2):
        super(GhostFaceNetV1, self).__init__()
        self.cfgs = [
            [3,  16,  16, 0, 1], [3,  48,  24, 0, 2], [3,  72,  24, 0, 1],
            [5,  72,  40, 1, 2], [5, 120,  40, 1, 1], [3, 240,  80, 0, 2],
            [3, 200,  80, 0, 1], [3, 184,  80, 0, 1], [3, 184,  80, 0, 1],
            [3, 480, 112, 1, 1], [3, 672, 112, 1, 1], [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1], [5, 960, 160, 1, 1], [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1]
        ]

        output_channel = int(16 * width)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.PReLU(output_channel)
        )
        input_channel = output_channel

        stages = []
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = int(c * width)
            hidden_channel = int(exp_size * width)
            stages.append(block(input_channel, hidden_channel,
                          output_channel, k, s, use_se))
            input_channel = output_channel
        self.blocks = nn.Sequential(*stages)

        output_channel = int(960 * width)
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.PReLU(output_channel)
        )

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(output_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.Linear(output_channel, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
        )

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_last(x)
        x = self.output_layer(x)
        return x

# ==========================================
# 3. TRAINING LOOP (GPU A100 Optimized)
# ==========================================


def main():
    p = argparse.ArgumentParser(
        description="Train GhostNet + AdaFace on GPU A100")
    p.add_argument("--data-root", type=str, required=True,
                   help="Path dataset (e.g. CASIA-WebFace)")
    p.add_argument("--out-dir", type=str,
                   default="weights_adaface", help="Folder output bobot")
    p.add_argument("--epochs", type=int, default=35,
                   help="Jumlah Epoch (Default 35)")
    p.add_argument("--batch-size", type=int,
                   default=256, help="Batch Size A100")

    # LR Kecil untuk GhostNet
    p.add_argument("--lr", type=float, default=0.01, help="Learning Rate Awal")
    p.add_argument("--workers", type=int, default=8, help="CPU Workers")

    # GPU Target
    p.add_argument("--gpu", type=int, default=0, help="GPU ID (Default: 0)")

    args = p.parse_args()

    import numpy as np

    # SETUP DEVICE (KHUSUS GPU 0)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        print("⚠️ GPU tidak terdeteksi! Menggunakan CPU.")
        device = torch.device("cpu")

    print(f"🚀 GhostNet + AdaFace Training Start on: {device}")
    print(
        f"   GPU Name: {torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else 'CPU'}")
    print(f"   Dataset : {args.data_root}")
    print(f"   Epochs  : {args.epochs}")

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

    # 2. Model & Head (AdaFace)
    model = GhostFaceNetV1(feat_dim=512).to(device)
    head = AdaFaceHead(embedding_size=512, classnum=len(ds.classes)).to(device)

    # 3. Optimizer & Scheduler
    opt = optim.SGD(list(model.parameters()) + list(head.parameters()),
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(
        opt, milestones=[20, 28, 32], gamma=0.1)

    ce = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Variabel Save Best
    best_acc = 0.0
    best_save_path = os.path.join(args.out_dir, "best_ghostnet_adaface.pt")

    # 4. Training Loop
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        total_loss = 0
        correct = 0
        total = 0

        # Flag untuk NaN
        is_nan = False

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, label in pbar:
            img, label = img.to(device), label.to(device)

            opt.zero_grad()

            # --- Mixed Precision Forward ---
            with autocast():
                features = model(img)
                logits = head(features, label)
                loss = ce(logits, label)

            # Cek NaN
            if math.isnan(loss.item()):
                print(f"\n❌ ERROR: Loss menjadi NaN di Epoch {epoch}!")
                is_nan = True
                break

            # --- Mixed Precision Backward ---
            scaler.scale(loss).backward()

            # GRADIENT CLIPPING
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=5.0)

            scaler.step(opt)
            scaler.update()

            # Statistik
            total_loss += loss.item()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}",
                             "Acc": f"{correct/total*100:.2f}%"})

        if is_nan:
            print("🛑 Menghentikan training untuk menyelamatkan model.")
            break

        scheduler.step()

        # Hitung Training Acc
        epoch_acc = correct / total * 100 if total > 0 else 0

        # Simpan Best Model
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
            save_name = f"ghostnet_adaface_epoch{epoch}.pt"
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
