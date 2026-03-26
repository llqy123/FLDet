import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LGVA(nn.Module):
    def __init__(self):
        super().__init__()

        self.topk = 50
        d_model = 512
        # coarse projection (semantic anchor)
        self.coarse_proj = nn.Linear(d_model, d_model)

        # gate for fine modulation
        self.fine_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        # fusion head
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def augmentation(self, x, concept_feats):
        """
        x: [N, D]
        concept_feats: [M, D]
        """
        sim = x @ concept_feats.T
        val, idx = sim.topk(self.topk, dim=-1)
        w = F.softmax(val, dim=-1)
        concepts = concept_feats[idx]  # [N, K, D]
        vec = (w.unsqueeze(-1) * concepts).sum(dim=1)  # [N, D]
        return vec

    def forward(self, x, fine_concept_feats, coarse_concept_feats):
        dtype = self.coarse_proj.weight.dtype
        x = x.to(dtype)
        coarse_concept_feats = coarse_concept_feats.to(dtype)
        fine_concept_feats = fine_concept_feats.to(dtype)
        # ---------- coarse retrieval ----------
        coarse_vec = self.augmentation(x, coarse_concept_feats)
        coarse_vec = F.normalize(coarse_vec, dim=-1)

        # ---------- coarse-anchored feature ----------
        x_coarse = F.normalize(x + coarse_vec, dim=-1)

        # ---------- fine retrieval (conditioned on coarse) ----------
        fine_vec = self.augmentation(x_coarse, fine_concept_feats)
        fine_vec = F.normalize(fine_vec, dim=-1)

        # ---------- adaptive fine gating ----------
        gate = self.fine_gate(coarse_vec)  # [N, 1]
        fine_residual = gate * fine_vec

        # ---------- coarse projection ----------
        coarse_proj = F.normalize(self.coarse_proj(x), dim=-1)

        # ---------- fusion ----------
        fused = self.fuse(
            torch.cat([coarse_proj, coarse_vec, fine_residual], dim=-1)
        )
        fused = F.normalize(fused + coarse_proj, dim=-1)

        return fused