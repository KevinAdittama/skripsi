# loss_magface.py
# -------------------------------------------------------------
# Loss Function: MagFace (Magnitude-Aware Margin)
# Referensi: CVPR 2021
# - Versi ini dibuat lebih "tunable" dan aman buat dataset kecil
# - Default param disetel lebih ringan (sering lebih stabil)
# -------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MagFaceHead(nn.Module):
    """
    MagFace: Magnitude-Aware Margin
    - margin m(a) ditentukan oleh magnitude a = ||x||
    - logits = s * cos(theta + m(a)) untuk kelas target, dan s * cos(theta) untuk non-target
    - optional g_loss untuk regularisasi magnitude (bisa dimatikan dengan g_weight=0 di script training)
    """

    def __init__(
        self,
        embedding_size: int,
        classnum: int,
        s: float = 32.0,
        l_a: float = 1.0,
        u_a: float = 20.0,
        l_m: float = 0.2,
        u_m: float = 0.5,
        lambda_g: float = 35.0,
    ):
        super().__init__()
        self.classnum = classnum
        self.s = float(s)

        # magnitude range
        self.l_a = float(l_a)
        self.u_a = float(u_a)

        # margin range
        self.l_m = float(l_m)
        self.u_m = float(u_m)

        self.lambda_g = float(lambda_g)

        self.weight = nn.Parameter(torch.empty(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def _margin_from_norm(self, a: torch.Tensor) -> torch.Tensor:
        """
        m(a) = linear mapping dari [l_a, u_a] -> [l_m, u_m], lalu clamp.
        a shape: (B, 1)
        """
        a = a.clamp(min=self.l_a, max=self.u_a)
        # linear interpolation
        m = (self.u_m - self.l_m) / (self.u_a -
                                     self.l_a + 1e-12) * (a - self.l_a) + self.l_m
        return m

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        """
        return:
          logits: (B, C)
          g_loss: scalar tensor
        """
        # 1) magnitude
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(min=1e-12)  # (B,1)

        # 2) normalize feature & weight untuk cosine
        x_feat = x / x_norm
        w_feat = F.normalize(self.weight, dim=1)

        # 3) cosine
        cosine = F.linear(x_feat, w_feat).clamp(-1.0 +
                                                1e-7, 1.0 - 1e-7)  # (B,C)

        # 4) adaptive margin m(a)
        margin = self._margin_from_norm(x_norm)  # (B,1)

        # 5) apply cos(theta + m) for target
        cos_m = torch.cos(margin)
        sin_m = torch.sin(margin)
        sin_theta = torch.sqrt((1.0 - cosine.pow(2)).clamp(0.0, 1.0))
        cos_theta_m = cosine * cos_m - sin_theta * sin_m

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        output = one_hot * cos_theta_m + (1.0 - one_hot) * cosine
        logits = output * self.s

        # 6) g_loss (regularisasi magnitude) — bisa dimatikan dari script training via g_weight=0
        # bikin magnitude "rapi": magnitude kecil dihukum lebih besar
        x_norm_val = x_norm.clamp(min=self.l_a, max=self.u_a)
        g_loss = torch.mean(1.0 / x_norm_val) * self.lambda_g

        return logits, g_loss
