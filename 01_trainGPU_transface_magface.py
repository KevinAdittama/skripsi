# 01_trainGPU_transface_magface.py
# -------------------------------------------------------------
# Tahap 1 (GPU A100): Training TransFace (ViT) + MagFace
# - Optimized for NVIDIA A100 (CUDA + AMP Enabled)
# - Arsitektur: Vision Transformer (Depth=12)
# - Loss: MagFace (dari loss_magface.py)
# - Optimizer: AdamW (Wajib untuk ViT)
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
# 2. ARSITEKTUR: VISION TRANSFORMER (ViT)
# ==========================================


class PatchEmbed(nn.Module):
    def __init__(self, img_size=112, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden_features = int(hidden_features or in_features)
        out_features = int(out_features or in_features)
        in_features = int(in_features)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerFace(nn.Module):
    def __init__(self, img_size=112, patch_size=8, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Depth=12 (Standard ViT) untuk A100
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=True, drop=drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x_cls = x[:, 0]
        return self.head(x_cls)

# ==========================================
# 3. TRAINING LOOP (GPU A100 Optimized)
# ==========================================


def main():
    p = argparse.ArgumentParser(
        description="Train TransFace + MagFace on GPU A100")
    p.add_argument("--data-root", type=str, required=True, help="Path dataset")
    p.add_argument("--out-dir", type=str,
                   default="weights_magface", help="Folder output bobot")
    p.add_argument("--epochs", type=int, default=35,
                   help="Jumlah Epoch (Default 35)")
    p.add_argument("--batch-size", type=int,
                   default=128, help="Batch Size A100")

    # LR ViT + AdamW
    p.add_argument("--lr", type=float, default=5e-4, help="Learning Rate Awal")
    p.add_argument("--workers", type=int, default=8, help="CPU Workers")

    # GPU Target: DEFAULT 1
    p.add_argument("--gpu", type=int, default=1, help="GPU ID (Default: 1)")

    args = p.parse_args()

    import numpy as np

    # SETUP DEVICE
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        print("⚠️ GPU tidak terdeteksi! Menggunakan CPU.")
        device = torch.device("cpu")

    print(f"🚀 TransFace (ViT) + MagFace Training Start on: {device}")
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
    # Depth 12 untuk A100
    model = VisionTransformerFace(
        embed_dim=512, depth=12, num_heads=8).to(device)
    head = MagFaceHead(embedding_size=512, classnum=len(ds.classes)).to(device)

    # 3. Optimizer (ADAMW WAJIB UNTUK VIT)
    opt = optim.AdamW(list(model.parameters()) + list(head.parameters()),
                      lr=args.lr, weight_decay=0.05)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)

    ce = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Variabel Save Best
    best_acc = 0.0
    best_save_path = os.path.join(args.out_dir, "best_transface_magface.pt")

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

            # GRADIENT CLIPPING
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

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total*100:.2f}%",
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
            save_name = f"transface_magface_epoch{epoch}.pt"
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
