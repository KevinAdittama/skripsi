# 01_trainGPU_resnet_adaface.py
# -------------------------------------------------------------
# Tahap 1 (GPU A100): Training ResNet + AdaFace
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
# 2. ARSITEKTUR: RESNET (IResNet)
# ==========================================


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.9)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.9)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.9),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down is not None:
            x = self.down(x)
        return self.prelu(out + x)


class IResNet(nn.Module):
    def __init__(self, layers=(3, 13, 30, 3), feat_dim=512):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.9),
            nn.PReLU(64),
        )
        modules = []
        modules.extend(self._make_layer(
            IBasicBlock, 64, 64, layers[0], stride=2))
        modules.extend(self._make_layer(
            IBasicBlock, 64, 128, layers[1], stride=2))
        modules.extend(self._make_layer(
            IBasicBlock, 128, 256, layers[2], stride=2))
        modules.extend(self._make_layer(
            IBasicBlock, 256, 512, layers[3], stride=2))
        self.body = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.9),
            nn.Dropout(p=0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim, eps=1e-5, momentum=0.9),
        )

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
    p = argparse.ArgumentParser(
        description="Train ResNet + AdaFace on GPU A100")
    p.add_argument("--data-root", type=str, required=True, help="Path dataset")
    p.add_argument("--out-dir", type=str,
                   default="weights_adaface", help="Folder output bobot")
    p.add_argument("--epochs", type=int, default=35, help="Jumlah Epoch")
    p.add_argument("--batch-size", type=int,
                   default=128, help="Batch Size A100")

    # LR Default 0.1 (ResNet cukup kuat)
    p.add_argument("--lr", type=float, default=0.1, help="Learning Rate")
    p.add_argument("--workers", type=int, default=8, help="CPU Workers")
    p.add_argument("--gpu", type=int, default=0, help="GPU ID (Default: 0)")

    args = p.parse_args()

    import numpy as np

    # SETUP DEVICE
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        print("⚠️ GPU tidak terdeteksi! Menggunakan CPU.")
        device = torch.device("cpu")

    print(f"🚀 ResNet + AdaFace Training Start on: {device}")
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

    # 2. Model & Head (AdaFace)
    model = IResNet(feat_dim=512).to(device)
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
    best_save_path = os.path.join(args.out_dir, "best_resnet_adaface.pt")

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

            # --- Mixed Precision Forward ---
            with autocast():
                features = model(img)
                logits = head(features, label)
                loss = ce(logits, label)

            if math.isnan(loss.item()):
                print(f"\n❌ ERROR: Loss NaN di Epoch {epoch}!")
                is_nan = True
                break

            # --- Mixed Precision Backward ---
            scaler.scale(loss).backward()

            # Gradient Clipping
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=5.0)

            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}",
                             "Acc": f"{correct/total*100:.2f}%"})

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
            save_name = f"resnet_adaface_epoch{epoch}.pt"
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
