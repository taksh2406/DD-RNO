"""
DD-RNO: Domain-Decomposed Routed Neural Operator with Learned Canonical Quadrature.

Novel contributions:
1. Physics-adaptive hard domain routing (SDF + Re-dependent BL thickness)
2. Learned Canonical Quadrature (LCQ) for force prediction

Architecture:
  SDF Grid → Fourier Neural Operator (SpectralTrunkFNO) → Z
  AoA/Re   → MappingNetwork       → w
  Query pts → DomainRouter → partition masks
  Decoders (PC-MoS):
    Outer → InviscidMLP → (u,v,p,nut)
    BL    → BLMLP       → (u,v,p,nut)
    Wake  → WakeMLP     → (u,v,p,nut)
  LCQ Head predicts force coefficients [Cl, Cd] from Cp.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from .encoders import SpectralTrunkFNO, MappingNetwork


# ─── Building Blocks ─────────────────────────────────────────────────────────

class FourierEncoding(nn.Module):
    """Fourier positional encoding for query coordinates."""
    def __init__(self, n_freqs=8, input_dim=2):
        super().__init__()
        self.n_freqs = n_freqs
        self.input_dim = input_dim
        # Output: input_dim * (2 * n_freqs) + input_dim
        self.out_dim = input_dim * (2 * n_freqs + 1)
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        # x: (B, N, input_dim)
        proj = x.unsqueeze(-1) * self.freqs  # (B, N, D, F)
        return torch.cat([x, proj.sin().flatten(-2), proj.cos().flatten(-2)], dim=-1)


class FiLMBlock(nn.Module):
    """Residual Block with Feature-wise Linear Modulation."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.film_gen = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x, z):
        # z: (B, cond_dim) -> gamma, beta: (B, 1, dim)
        gamma, beta = self.film_gen(z).unsqueeze(1).chunk(2, dim=-1)
        h = self.net(self.ln(x))
        h = (1 + gamma) * h + beta
        return x + h


class DomainMLP(nn.Module):
    """High-Fidelity Decoder with FiLM injection at every block."""
    def __init__(self, in_dim, cond_dim, hidden_dim, n_blocks, out_dim):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, cond_dim) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_enc, z):
        h = self.in_proj(x_enc)
        for block in self.blocks:
            h = block(h, z)
        return self.head(h)


# ─── Domain Router ────────────────────────────────────────────────────────────

class DomainRouter(nn.Module):
    """
    Physics-adaptive hard domain partition.

    Partitions query points into:
      - Inviscid (outer): SDF > δ_bl AND x <= x_wake
      - Boundary Layer:   SDF <= δ_bl
      - Wake:             x > x_wake

    δ_bl is Reynolds-dependent: δ_bl = C * Re^{-1/5}
    """
    def __init__(self, c_bl=5.0, x_wake=1.05):
        super().__init__()
        self.c_bl = c_bl
        self.x_wake = x_wake

    def forward(self, query_pts, sdf, re_phys):
        """
        query_pts: (B, N, 2) — x, y coordinates
        sdf:       (B, N) — signed distance to surface
        re_phys:   (B, 1) — Reynolds number

        Returns: dict of boolean masks, each (B, N)
        """
        # Physics-adaptive BL thickness
        delta_bl = self.c_bl * re_phys.pow(-0.2)  # (B, 1)

        # Soft transitions
        # width of transition region (scale with delta_bl)
        w_bl = delta_bl * 0.1 
        w_wake = 0.05
        
        # Soft mask for BL (1 inside BL, 0 outside)
        mask_bl = torch.sigmoid((delta_bl - sdf) / w_bl)
        # Soft mask for Wake (1 inside wake, 0 outside)
        mask_wake = torch.sigmoid((query_pts[..., 0] - self.x_wake) / w_wake)
        
        # Soft mask for Inv (1 outside BL and outside Wake)
        mask_inv = (1.0 - mask_bl) * (1.0 - mask_wake)
        
        # We also need to make sure they sum to 1 to preserve total signal energy
        total_mask = mask_bl + mask_wake + mask_inv
        
        return {'inv': mask_inv / total_mask, 'bl': mask_bl / total_mask, 'wake': mask_wake / total_mask}


# ─── Surface Pressure + LCQ Force Head ───────────────────────────────────────

class LCQForceHead(nn.Module):
    """
    Learned Canonical Quadrature: predicts Cl, Cd from Cp via learned
    AoA-dependent integration weights.

    Standard integration: Cl = Σ w_i * Cp_i * n_y,i * ds_i
    LCQ replaces the fixed geometric weights with a learned function
    of the flow condition, capturing neural interpolation error.
    """
    def __init__(self, cond_dim=64, n_canonical=256):
        super().__init__()
        self.n_canonical = n_canonical
        # Condition → per-node weights for Cl and Cd
        self.weight_net = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, n_canonical * 2),  # 2 forces × N weights
        )

    def forward(self, cp, w):
        """
        cp: (B, N_canon) — predicted surface Cp
        w:  (B, cond_dim) — flow condition latent (contains AoA info)
        Returns: (B, 2) — [Cl, Cd]
        """
        weights = self.weight_net(w).view(-1, 2, self.n_canonical)  # (B, 2, N)
        # Learned weighted sum: each force is a dot product with Cp
        forces = (weights * cp.unsqueeze(1)).sum(dim=-1)            # (B, 2)
        return forces


# ─── Top-Level Model ──────────────────────────────────────────────────────────

class DDRNO(nn.Module):
    """
    Domain-Decomposed Routed Neural Operator with Learned Canonical Quadrature.
    """
    def __init__(self, geom_dim=64, w_dim=64, hidden=256,
                 n_fourier=8, n_canonical=1024,
                 inv_layers=4, bl_layers=6, wake_layers=4,
                 use_domain_routing=True, use_lcq=True,
                 use_specinr=True, **kwargs):
        super().__init__()
        
        # Support legacy argument names
        if 'hidden_dim' in kwargs:
            hidden = kwargs['hidden_dim']
        if 'inv_n_layers' in kwargs:
            inv_layers = kwargs['inv_n_layers']
        if 'vis_n_layers' in kwargs:
            bl_layers = kwargs['vis_n_layers']
        if 'wake_n_layers' in kwargs:
            wake_layers = kwargs['wake_n_layers']
            
        self.use_domain_routing = use_domain_routing
        self.use_lcq = use_lcq
        self.use_specinr = use_specinr
        
        # SpecINR: Decoder takes (Local Spectral + Global Flow + Coord Features)
        # cond_dim = geom_dim (local) + w_dim (global)
        cond_dim = geom_dim + w_dim 

        # ── Encoder: Novel Spectral Trunk ──
        # Outputs 64x64x64 feature map
        self.trunk = SpectralTrunkFNO(out_channels=geom_dim)
        self.mapping = MappingNetwork(cond_dim=4, w_dim=w_dim)

        # ── Fourier Encoding ──
        self.fourier = FourierEncoding(n_freqs=n_fourier, input_dim=2)
        f_dim = self.fourier.out_dim 

        # SpecINR: Decoders take concat(Fourier-coord, local-spectral, local-sdf, local-normal) as input token.
        # FiLM conditioning stays global (w_dim only).
        if use_specinr:
            token_dim = self.fourier.out_dim + geom_dim + 1 + 2  # Fourier coord + probed features + SDF + normal
        else:
            token_dim = self.fourier.out_dim + 1 + 2  # Fourier coord + SDF + normal

        # ── Domain Router ──
        self.router = DomainRouter(c_bl=5.0, x_wake=1.05)

        # ── Per-Domain Decoders ──
        # in_dim = token_dim, cond_dim = w_dim (global flow only)
        # All decoders now output the full 4 channels to prevent zero-padding artifacts during soft blending
        self.inv_dec  = DomainMLP(token_dim, w_dim, hidden, inv_layers,  out_dim=4)
        self.bl_dec   = DomainMLP(token_dim, w_dim, hidden, bl_layers,   out_dim=4)
        self.wake_dec = DomainMLP(token_dim, w_dim, hidden, wake_layers, out_dim=4)

        # ── Surface + Force ──
        self.lcq_head = LCQForceHead(cond_dim=w_dim, n_canonical=n_canonical)

        # ── Standardization buffers ──
        self.register_buffer('field_mean', torch.zeros(4))
        self.register_buffer('field_std',  torch.ones(4))
        self.register_buffer('cl_cd_mean', torch.zeros(2))
        self.register_buffer('cl_cd_std',  torch.ones(2))

        # ── Coordinate Scaling (AirfRANS Domain) ──
        self.register_buffer('coord_min', torch.tensor([-2.0, -3.0]))
        self.register_buffer('coord_max', torch.tensor([6.0, 3.0]))

    def set_standardization_stats(self, cl_cd_mean, cl_cd_std, field_mean=None, field_std=None):
        self.cl_cd_mean.data.copy_(torch.tensor(cl_cd_mean))
        self.cl_cd_std.data.copy_(torch.tensor(cl_cd_std))
        if field_mean is not None:
            self.field_mean.data.copy_(torch.tensor(field_mean))
        if field_std is not None:
            self.field_std.data.copy_(torch.tensor(field_std))

    def bilinear_probe(self, feature_map, points):
        """
        Probes a 2D feature map at arbitrary continuous coordinates.
        feature_map: (B, C, H, W)
        points:      (B, N, 2) in world space
        Returns:     (B, N, C) probed features
        """
        B, C, H, W = feature_map.shape
        # Map world points to [-1, 1] relative to the FNO grid
        # Assumes the FNO grid covers the same domain as coord_min/max
        pts_norm = 2.0 * (points - self.coord_min) / (self.coord_max - self.coord_min + 1e-8) - 1.0
        
        # grid_sample expects (B, H_out, W_out, 2)
        grid = pts_norm.unsqueeze(2) # (B, N, 1, 2)
        probed = torch.nn.functional.grid_sample(feature_map, grid, align_corners=True)
        return probed.squeeze(-1).permute(0, 2, 1) # (B, N, C)

    def forward(self, sdf_grid, aoa_enc, re_enc, query_pts, query_sdf, re_phys, surf_pts=None, query_normal=None, surf_normal=None):
        """
        Full SpecINR forward pass with Local Coordinate Injection.
        """
        # 1. Global Conditioning
        w = self.mapping(aoa_enc, re_enc) # (B, w_dim)
        
        # 2. Spectral Feature Map (if SpecINR is enabled)
        if self.use_specinr:
            z_map = self.trunk(sdf_grid) # (B, geom_dim, 64, 64)
            # 3. Local Feature Probing
            f_q = self.bilinear_probe(z_map, query_pts)  # (B, N, geom_dim)
        
        # 4. Fourier encoding + SpecINR token (augmented with SDF and normal)
        pts_norm = 2.0 * (query_pts - self.coord_min) / (self.coord_max - self.coord_min + 1e-8) - 1.0
        x_enc_raw = self.fourier(pts_norm)            # (B, N, f_dim)
        
        B, N, _ = query_pts.shape
        if query_normal is None:
            query_normal = torch.zeros(B, N, 2, device=query_pts.device)
            
        if self.use_specinr:
            x_enc = torch.cat([x_enc_raw, f_q, query_sdf.unsqueeze(-1), query_normal], dim=-1)  # (B, N, token_dim)
        else:
            x_enc = torch.cat([x_enc_raw, query_sdf.unsqueeze(-1), query_normal], dim=-1)  # (B, N, token_dim)

        # 5. Domain Routing
        if self.use_domain_routing:
            masks = self.router(query_pts, query_sdf, re_phys)
        else:
            masks = {'inv': torch.zeros_like(query_sdf, dtype=torch.bool),
                     'bl': torch.ones_like(query_sdf, dtype=torch.bool),
                     'wake': torch.zeros_like(query_sdf, dtype=torch.bool)}
        
        pred_field = torch.zeros(B, N, 4, device=query_pts.device)

        # 6. Decoder Execution (local spectral context fused into token)
        if self.use_domain_routing:
            # Unconditionally evaluate all decoders and blend them using the soft masks.
            inv_out = self.inv_dec(x_enc, w)
            bl_out = self.bl_dec(x_enc, w)
            wake_out = self.wake_dec(x_enc, w)
            
            pred_field = (
                inv_out * masks['inv'].unsqueeze(-1) +
                bl_out * masks['bl'].unsqueeze(-1) +
                wake_out * masks['wake'].unsqueeze(-1)
            )
        else:
            pred_field = self.bl_dec(x_enc, w)

        # 7. Force Prediction (Spectral-Probed Surface)
        if self.use_lcq:
            if surf_pts is not None:
                if self.use_specinr:
                    f_s = self.bilinear_probe(z_map, surf_pts)  # (B, 1024, geom_dim)
                surf_norm = 2.0 * (surf_pts - self.coord_min) / (self.coord_max - self.coord_min + 1e-8) - 1.0
                surf_enc_raw = self.fourier(surf_norm)
                
                # Surface points lie on the wall, so local SDF is 0.0
                surf_sdf = torch.zeros(B, surf_pts.shape[1], 1, device=surf_pts.device)
                if surf_normal is None:
                    surf_normal = torch.zeros(B, surf_pts.shape[1], 2, device=surf_pts.device)
                    
                if self.use_specinr:
                    surf_enc = torch.cat([surf_enc_raw, f_s, surf_sdf, surf_normal], dim=-1)  # (B, 1024, token_dim)
                else:
                    surf_enc = torch.cat([surf_enc_raw, surf_sdf, surf_normal], dim=-1)  # (B, 1024, token_dim)
                surf_out = self.bl_dec(surf_enc, w)
                cp = surf_out[..., 2]
            else:
                cp = torch.zeros(B, self.lcq_head.n_canonical, device=query_pts.device)

            pred_clcd = self.lcq_head(cp, w)
        else:
            pred_clcd = self.lcq_head.weight_net(w)[:, :2]

        return pred_field, pred_clcd
