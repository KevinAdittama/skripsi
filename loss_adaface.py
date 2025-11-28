# adaface.py
# -------------------------------------------------------------
# Loss Function: AdaFace (Quality Adaptive Margin)
# Referensi: CVPR 2022
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaFaceHead(nn.Module):
    """
    AdaFace: Quality Adaptive Margin
    Menggunakan Feature Norm sebagai proksi kualitas gambar.
    """

    def __init__(self, embedding_size, classnum, m=0.4, h=0.333, s=64.0, t_alpha=0.01):
        super().__init__()
        self.classnum = classnum
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha

        self.weight = nn.Parameter(torch.FloatTensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # Batch Norm untuk menstabilkan feature norm
        self.register_buffer('batch_mean', torch.tensor(20.0))
        self.register_buffer('batch_std', torch.tensor(100.0))

    def forward(self, x, label):
        # 1. Normalize Weight
        W = F.normalize(self.weight)

        # 2. Hitung Norm Fitur (Indikator Kualitas)
        norms = torch.norm(x, dim=1, keepdim=True)
        x_normalized = x / (norms + 1e-12)

        # 3. Hitung Cosine
        cosine = F.linear(x_normalized, W)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # Stability

        # 4. Update Statistik Norm (Moving Average)
        with torch.no_grad():
            mean = norms.mean().detach()
            std = norms.std().detach()
            self.batch_mean = self.batch_mean * \
                (1 - self.t_alpha) + mean * self.t_alpha
            self.batch_std = self.batch_std * \
                (1 - self.t_alpha) + std * self.t_alpha

        # 5. Hitung Image Quality Indicator (norm_hat)
        # Mengembalikan nilai antara -1 (jelek) sampai 1 (bagus)
        margin_scaler = (norms - self.batch_mean) / (self.batch_std + 1e-12)
        margin_scaler = margin_scaler * self.h  # scaling
        margin_scaler = torch.clamp(margin_scaler, -1.0, 1.0)

        # 6. Adaptasi Margin
        # G_angle: margin sudut, G_add: margin aditif
        # Jika gambar jelek -> margin dikurangi. Jika bagus -> margin ditambah.
        g_angle = -self.m * margin_scaler
        g_add = self.m + (self.m * margin_scaler)

        # 7. Terapkan Margin
        cos_m = torch.cos(g_angle)
        sin_m = torch.sin(g_angle)
        sin_theta = torch.sqrt(1.0 - cosine.pow(2))
        cos_theta_m = cosine * cos_m - sin_theta * sin_m

        # Khusus AdaFace: Margin ditambahkan juga ke logit (additive)
        cos_theta_m_final = cos_theta_m - g_add

        # Pilih logit yang sesuai label
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * cos_theta_m_final) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
