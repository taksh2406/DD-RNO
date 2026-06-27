"""
Simple Trainer for DD-RNO.
Two losses only: field MSE + force MSE. No GradNorm, no GP, no discriminator.
"""
import os
import time
import torch
import torch.nn.functional as F
import logging


class SimpleTrainer:
    def __init__(self, model, train_loader, val_loader, config, member_id=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.adaptive_loss = bool(config.get('adaptive_loss', True))
        self.lambda_field = float(config.get('lambda_field', 1.0))
        self.lambda_force = float(config.get('lambda_force', 10.0))

        if self.adaptive_loss:
            # Multi-Task Adaptive Loss (Kendal et al. 2018)
            # log_vars init to 0.0 -> initial weights = 1.0
            self.log_vars = torch.nn.Parameter(torch.zeros(2, device=self.device))
            params = list(model.parameters()) + [self.log_vars]
        else:
            params = list(model.parameters())
        
        self.opt = torch.optim.AdamW(
            params,
            lr=float(config.get('lr', 3e-4)),
            weight_decay=float(config.get('weight_decay', 1e-4))
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=int(config.get('epochs', 1000))
        )

        self.field_mean = torch.tensor(config['field_mean'], device=self.device)
        self.field_std  = torch.tensor(config['field_std'],  device=self.device)
        self.cl_cd_mean = torch.tensor(config['cl_cd_mean'], device=self.device)
        self.cl_cd_std  = torch.tensor(config['cl_cd_std'],  device=self.device)

        self.logger = logging.getLogger(f"DDRNO_{member_id}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    def train_epoch(self, epoch):
        self.model.train()
        n_batches = len(self.train_loader)
        metrics = {'loss': 0, 'field_mse': 0, 'force_mse': 0, 'Cl_MAE': 0, 'Cd_MAE': 0}

        for batch in self.train_loader:
            sdf_grid  = batch['sdf_grid'].to(self.device)
            aoa_enc   = batch['aoa_enc'].to(self.device)
            re_enc    = batch['re_enc'].to(self.device)
            query_pts = batch['query_pts'].to(self.device)
            query_sdf = batch['query_phi'].to(self.device).squeeze(-1)  # (B, N)
            query_norm = batch['query_normal'].to(self.device)
            re_phys   = batch['re_phys'].to(self.device)
            target    = batch['target'].to(self.device)

            # Standardize force labels
            cl_cd_raw = batch['cl_cd_raw'].to(self.device)
            cl_cd_std = (cl_cd_raw - self.cl_cd_mean) / self.cl_cd_std

            self.opt.zero_grad()

            pred_field, pred_clcd = self.model(
                sdf_grid, aoa_enc, re_enc, query_pts, query_sdf, re_phys,
                surf_pts=batch['surf_xy'].to(self.device),
                query_normal=query_norm,
                surf_normal=batch['surf_normal'].to(self.device)
            )

            # Loss 1: Field MSE (standardized)
            # Complexity Addition: SDF-based importance weighting
            # Penalize errors near the surface (where forces are calculated) 10x more
            sdf_weight = torch.exp(-15.0 * query_sdf.abs()).unsqueeze(-1) + 0.1 # (B, N, 1)
            l_field = (sdf_weight * (pred_field - target).pow(2)).mean()

            # Loss 2: Force MSE (standardized)
            l_force = F.mse_loss(pred_clcd, cl_cd_std)

            if self.adaptive_loss:
                # Adaptive Multi-Task Loss: L = sum( exp(-log_var) * L + log_var )
                w_field = torch.exp(-self.log_vars[0])
                w_force = torch.exp(-self.log_vars[1])
                loss = (w_field * l_field + self.log_vars[0]) + (w_force * l_force + self.log_vars[1])
            else:
                loss = self.lambda_field * l_field + self.lambda_force * l_force

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.opt.step()
            
            if self.adaptive_loss:
                # Clamp log_vars for optimization stability
                with torch.no_grad():
                    self.log_vars.clamp_(min=-3.0, max=5.0)

            # Metrics (no grad)
            with torch.no_grad():
                metrics['loss']      += loss.item() / n_batches
                metrics['field_mse'] += l_field.item() / n_batches
                metrics['force_mse'] += l_force.item() / n_batches

                # Physical force errors
                pred_phys = pred_clcd * self.cl_cd_std + self.cl_cd_mean
                metrics['Cl_MAE'] += (pred_phys[:, 0] - cl_cd_raw[:, 0]).abs().mean().item() / n_batches
                metrics['Cd_MAE'] += (pred_phys[:, 1] - cl_cd_raw[:, 1]).abs().mean().item() / n_batches

        return metrics

    def validate(self, epoch):
        self.model.eval()
        n_batches = len(self.val_loader)
        u_err, v_err, p_err = 0, 0, 0
        cl_err, cd_err = 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                sdf_grid  = batch['sdf_grid'].to(self.device)
                aoa_enc   = batch['aoa_enc'].to(self.device)
                re_enc    = batch['re_enc'].to(self.device)
                query_pts = batch['query_pts'].to(self.device)
                query_sdf = batch['query_phi'].to(self.device).squeeze(-1)
                query_norm = batch['query_normal'].to(self.device)
                re_phys   = batch['re_phys'].to(self.device)
                target    = batch['target'].to(self.device)
                cl_cd_raw = batch['cl_cd_raw'].to(self.device)
                cl_cd_std = (cl_cd_raw - self.cl_cd_mean) / self.cl_cd_std

                pred_field, pred_clcd = self.model(
                    sdf_grid, aoa_enc, re_enc, query_pts, query_sdf, re_phys,
                    surf_pts=batch['surf_xy'].to(self.device),
                    query_normal=query_norm,
                    surf_normal=batch['surf_normal'].to(self.device)
                )

                err = (pred_field - target).pow(2)
                u_err += err[..., 0].mean().item() / n_batches
                v_err += err[..., 1].mean().item() / n_batches
                p_err += err[..., 2].mean().item() / n_batches

                pred_phys = pred_clcd * self.cl_cd_std + self.cl_cd_mean
                cl_err += (pred_phys[:, 0] - cl_cd_raw[:, 0]).abs().mean().item() / n_batches
                cd_err += (pred_phys[:, 1] - cl_cd_raw[:, 1]).abs().mean().item() / n_batches

        val_loss = u_err + v_err + p_err
        m = {
            'u_mse': u_err, 'v_mse': v_err, 'p_mse': p_err,
            'cl_mae': cl_err, 'cd_mae': cd_err,
        }
        return val_loss, m

    def train(self, epochs=1000):
        best_cl = float('inf')
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints/ddrno')
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(epochs):
            t0 = time.time()
            res = self.train_epoch(epoch)
            dt = time.time() - t0

            _, m = self.validate(epoch)

            self.logger.info(
                f"Ep {epoch:03d} | "
                f"u:{m['u_mse']:.4f} v:{m['v_mse']:.4f} p:{m['p_mse']:.4f} | "
                f"Cl:{m['cl_mae']:.4f} Cd:{m['cd_mae']:.6f} | "
                f"L_f:{res['field_mse']:.4f} L_c:{res['force_mse']:.4f} | "
                f"t:{dt:.1f}s"
            )

            if m['cl_mae'] < best_cl:
                best_cl = m['cl_mae']
                self.save(os.path.join(checkpoint_dir, f"best_cl_ep{epoch:03d}.pt"))
                self.save(os.path.join(checkpoint_dir, "best_cl.pt"))
            self.sched.step()

        self.save(os.path.join(checkpoint_dir, 'last.pt'))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
