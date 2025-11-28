# magface.py
# -------------------------------------------------------------
# Loss Function: MagFace (Magnitude-Aware Margin)
# Referensi: CVPR 2021
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MagFaceHead(nn.Module):
    """
    MagFace: Magnitude-Aware Margin
    Menggunakan besaran (magnitude) fitur untuk menentukan kualitas wajah.
    """

    def __init__(self, embedding_size, classnum, s=64.0, l_a=10, u_a=110, l_m=0.45, u_m=0.8, lambda_g=35):
        super().__init__()
        self.s = s
        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        self.lambda_g = lambda_g

        self.weight = nn.Parameter(torch.FloatTensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def _calc_cosine(self, x_norm, w_norm):
        # x_norm: (B, F), w_norm: (C, F)
        # return: (B, C)
        return F.linear(x_norm, w_norm)

    def forward(self, x, label):
        # 1. Hitung Magnitude x
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm_val = x_norm.clamp(min=self.l_a, max=self.u_a)

        # 2. Normalize x & w
        x_feat = x / x_norm
        w_feat = F.normalize(self.weight, dim=1)

        # 3. Hitung Cosine
        cosine = self._calc_cosine(x_feat, w_feat)

        # 4. Hitung Margin Adaptif berdasarkan Magnitude
        # m(a) linear function antara l_m dan u_m
        margin = (self.u_m - self.l_m) / (self.u_a - self.l_a) * \
            (x_norm_val - self.l_a) + self.l_m

        # 5. Terapkan Margin ke Logit Benar
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # cos(theta + m)
        cos_m = torch.cos(margin)
        sin_m = torch.sin(margin)
        sin_theta = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        cos_theta_m = cosine * cos_m - sin_theta * sin_m

        # Ganti logit asli dengan logit bermargin hanya untuk kelas yang benar
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cosine

        # 6. Scale
        output *= self.s

        # 7. Loss Tambahan (g_loss) untuk memaksa magnitude beraturan
        # MagFace punya loss tambahan agar magnitude besar = mudah, kecil = sulit
        g_loss = torch.mean(1.0 / x_norm_val) * self.lambda_g

        return output, g_loss
