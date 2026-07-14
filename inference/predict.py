"""
DD-RNO Inference API: .dat file → (Cl, Cd) and field predictions.

This module provides the end-to-end pipeline for predicting aerodynamic
coefficients and flow fields from raw airfoil coordinate files (.dat format),
without requiring any CFD mesh or HDF5 preprocessing.

Usage:
    from inference.predict import DDRNOPredictor
    
    predictor = DDRNOPredictor("checkpoints/ddrno/best_cl.pt",
                               "configs/ddrno.yaml")
    result = predictor.predict_from_dat("naca0012.dat", aoa_deg=5.0, re=3e6)
    print(f"Cl = {result['Cl']:.4f}, Cd = {result['Cd']:.6f}")
"""
import os
import yaml
import numpy as np
import torch
from scipy.spatial import KDTree


def read_dat_file(dat_path):
    """
    Read an airfoil .dat file (Selig or Lednicer format).
    
    Returns:
        xy: (N, 2) array of airfoil surface coordinates, normalized to
            chord length 1.0 with leading edge at origin.
    """
    lines = open(dat_path, 'r').readlines()
    coords = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
            coords.append([x, y])
        except ValueError:
            continue  # Skip header lines

    xy = np.array(coords, dtype=np.float64)
    
    # Normalize: translate LE to origin, scale chord to 1.0
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    chord = x_max - x_min
    if chord > 0:
        xy[:, 0] = (xy[:, 0] - x_min) / chord
        xy[:, 1] = xy[:, 1] / chord

    return xy.astype(np.float32)


def compute_sdf_grid_from_coords(airfoil_xy, res=64,
                                  domain=((-0.5, 1.5), (-1.0, 1.0))):
    """
    Compute a signed distance field grid from airfoil surface coordinates.
    
    Reproduces the same representation used by the training pipeline:
    nearest-neighbor interpolation of surface SDF (unsigned distance to the
    nearest surface point) on a regular Cartesian grid, with the interior
    of the airfoil zeroed out.
    
    Args:
        airfoil_xy: (N, 2) array of airfoil surface coordinates (chord-normalized)
        res: grid resolution (default 64)
        domain: ((xmin, xmax), (ymin, ymax)) grid bounds
        
    Returns:
        sdf_grid: (res, res) float32 array of distances
    """
    from matplotlib.path import Path
    
    xi = np.linspace(domain[0][0], domain[0][1], res)
    yi = np.linspace(domain[1][0], domain[1][1], res)
    xx, yy = np.meshgrid(xi, yi)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (res*res, 2)
    
    # Build KDTree of airfoil surface points for fast nearest-neighbor lookup
    tree = KDTree(airfoil_xy)
    dists, _ = tree.query(grid_pts)
    
    # Ensure coordinates are ordered to compute a correct inside/outside path
    xy_ord = order_contour(airfoil_xy)
    path = Path(xy_ord)
    inside = path.contains_points(grid_pts)
    dists[inside] = 0.0
    
    sdf_grid = dists.reshape(res, res).astype(np.float32)
    return sdf_grid


def compute_surface_normals(airfoil_xy):
    """
    Compute outward-pointing unit normals at each airfoil surface point.
    Uses finite differences on the ordered contour.
    
    Args:
        airfoil_xy: (N, 2) ordered airfoil surface coordinates
        
    Returns:
        normals: (N, 2) outward unit normals
    """
    N = airfoil_xy.shape[0]
    # Tangent vectors (forward difference, wrapped)
    tangents = np.roll(airfoil_xy, -1, axis=0) - airfoil_xy
    # Normal = rotate tangent by -90 degrees: (dx, dy) → (dy, -dx)
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=-1)
    # Normalize
    norms = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norms
    
    # Ensure outward-pointing: normals should point away from airfoil centroid
    centroid = airfoil_xy.mean(axis=0)
    outward = airfoil_xy - centroid
    # If dot product with outward direction is negative, flip
    dots = (normals * outward).sum(axis=-1)
    flip_mask = dots < 0
    normals[flip_mask] *= -1
    
    return normals.astype(np.float32)


def order_contour(airfoil_xy):
    """
    Ensure coordinates are ordered consecutively along the contour starting from 
    the upper trailing edge and walking forward. If the points are already 
    consecutively ordered (like in standard Selig .dat files), we bypass the 
    nearest-neighbor search to avoid jumping the trailing edge gap.
    """
    dists = np.linalg.norm(np.diff(airfoil_xy, axis=0), axis=1)
    if dists.sum() < 3.0:
        return airfoil_xy

    x_max = airfoil_xy[:, 0].max()
    candidates = np.where((airfoil_xy[:, 0] > 0.98 * x_max) & (airfoil_xy[:, 1] >= 0.0))[0]
    if len(candidates) == 0:
        candidates = [np.argmax(airfoil_xy[:, 0])]
    te_upper_idx = candidates[np.argmax(airfoil_xy[candidates, 1])]

    n_pts = len(airfoil_xy)
    remaining = list(range(n_pts))
    remaining.remove(te_upper_idx)

    # Force the first step to walk forward along the upper surface (decreasing x)
    dists_from_start = np.linalg.norm(airfoil_xy[remaining] - airfoil_xy[te_upper_idx], axis=1)
    closest_idxs = np.argsort(dists_from_start)[:5]
    next_idx = None
    for idx in closest_idxs:
        real_idx = remaining[idx]
        if airfoil_xy[real_idx, 0] < airfoil_xy[te_upper_idx, 0] - 1e-4:
            next_idx = real_idx
            break
    if next_idx is None:
        next_idx = remaining[closest_idxs[0]]

    ordered_idx = [te_upper_idx, next_idx]
    remaining.remove(next_idx)
    curr = next_idx
    while len(remaining) > 0:
        dists_to_curr = np.linalg.norm(airfoil_xy[remaining] - airfoil_xy[curr], axis=1)
        next_idx = remaining[np.argmin(dists_to_curr)]
        ordered_idx.append(next_idx)
        remaining.remove(next_idx)
        curr = next_idx

    return airfoil_xy[ordered_idx]


def resample_curve(xy, n_target=1024):
    """
    Resample airfoil curve to fixed number of points using cosine spacing.
    Concentrates points near leading and trailing edges.
    """
    n_orig = xy.shape[0]
    if n_orig == n_target:
        return xy
    
    # Compute cumulative arc length
    diffs = np.diff(xy, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0], np.cumsum(seg_lengths)])
    arc /= arc[-1]  # normalize to [0, 1]
    
    # Cosine-spaced parameter values (clusters at 0 and 1)
    t_new = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_target)))
    
    # Interpolate x and y
    x_new = np.interp(t_new, arc, xy[:, 0])
    y_new = np.interp(t_new, arc, xy[:, 1])
    
    return np.stack([x_new, y_new], axis=-1).astype(np.float32)


class DDRNOPredictor:
    """
    End-to-end predictor: .dat file → (Cl, Cd) and optional field values.
    
    This class encapsulates the full pipeline:
    1. Read airfoil coordinates from .dat file
    2. Compute SDF grid from coordinates
    3. Prepare canonical surface points for LCQ force integration
    4. Run DD-RNO inference
    5. Denormalize and return physical Cl, Cd values
    """
    
    def __init__(self, checkpoint_path, config_path, device=None):
        """
        Args:
            checkpoint_path: path to model checkpoint (.pt file) OR a dict mapping 
                            split names (e.g. 'full', 'reynolds') to checkpoint paths.
            config_path: path to config YAML file (or dict of config paths matching keys)
            device: torch device (auto-detected if None)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Standardize checkpoint_path input
        if isinstance(checkpoint_path, str):
            self.checkpoints = {'full': checkpoint_path}
        elif isinstance(checkpoint_path, dict):
            self.checkpoints = checkpoint_path
        else:
            raise ValueError("checkpoint_path must be a string or dict of paths")

        # Standardize config_path input
        if isinstance(config_path, str):
            self.config_paths = {k: config_path for k in self.checkpoints.keys()}
        elif isinstance(config_path, dict):
            self.config_paths = config_path
        else:
            raise ValueError("config_path must be a string or dict of paths")

        from model.dd_rno import DDRNO
        self.models = {}
        for key, ckpt_p in self.checkpoints.items():
            cfg_p = self.config_paths.get(key, list(self.config_paths.values())[0])
            with open(cfg_p, 'r') as f:
                config_raw = yaml.safe_load(f)
            config = {}
            for section in ('data', 'model', 'training'):
                if section in config_raw:
                    config.update(config_raw[section])
            
            m = DDRNO(
                geom_dim=config.get('geom_dim', 128),
                w_dim=config.get('w_dim', 64),
                hidden=config.get('hidden', 256),
                n_fourier=config.get('n_fourier', 8),
                n_canonical=config.get('n_canonical', 1024),
                inv_layers=config.get('inv_layers', 4),
                bl_layers=config.get('bl_layers', 6),
                wake_layers=config.get('wake_layers', 4),
            ).to(self.device)

            ckpt = torch.load(ckpt_p, map_location=self.device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                m.load_state_dict(ckpt['model_state_dict'])
            else:
                m.load_state_dict(ckpt)
            m.eval()
            self.models[key] = m

        # Default model reference for single-model behavior
        self.model = next(iter(self.models.values()))
        self.config = config
        
        # Grid domain (must match prepare_airfrans.py)
        self.grid_domain = ((-0.5, 1.5), (-1.0, 1.0))
        self.grid_res = 64
        self.n_surface = self.config.get('n_canonical', 1024)
        
        # Load canonical mapping parameters for LCQ surface resampling
        mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canonical_surface_mapping.npz")
        if os.path.exists(mapping_path):
            data = np.load(mapping_path)
            self.s_canon = data["s_canon"]
            self.n_sign = data["n_sign"]
        else:
            self.s_canon = None
            self.n_sign = None
        
    def _encode_conditions(self, aoa_deg, re):
        """Encode angle of attack and Reynolds number."""
        alpha = np.radians(aoa_deg)
        aoa_enc = np.array([np.sin(alpha), np.cos(alpha), np.sin(2*alpha)],
                           dtype=np.float32)
        re_enc = np.array([np.log10(max(re, 1e4) / 1e6)], dtype=np.float32)
        return aoa_enc, re_enc, float(re)
    
    def predict_from_coords(self, airfoil_xy, aoa_deg, re, 
                             query_pts=None, return_field=False):
        """
        Predict Cl, Cd (and optionally field values) from airfoil coordinates.
        
        Args:
            airfoil_xy: (N, 2) airfoil surface coordinates (chord-normalized)
            aoa_deg: angle of attack in degrees
            re: Reynolds number
            query_pts: (M, 2) optional query coordinates for field prediction
            return_field: if True and query_pts given, return field predictions
            
        Returns:
            dict with keys 'Cl', 'Cd', and optionally 'field' (M, 4) array
        """
        # Ensure coordinates are ordered consecutively along the contour
        airfoil_xy = order_contour(airfoil_xy)

        # Auto-resample sparse inputs to 1000 points to ensure smooth SDF computation
        if len(airfoil_xy) < 500:
            diffs = np.diff(airfoil_xy, axis=0)
            lens = np.linalg.norm(diffs, axis=1)
            arc_native = np.concatenate([[0], np.cumsum(lens)])
            arc_native /= arc_native[-1]

            from scipy.interpolate import interp1d
            f_x = interp1d(arc_native, airfoil_xy[:, 0], kind='linear', fill_value='extrapolate')
            f_y = interp1d(arc_native, airfoil_xy[:, 1], kind='linear', fill_value='extrapolate')
            t_fine = np.linspace(0, 1, 1000)
            airfoil_xy = np.stack([f_x(t_fine), f_y(t_fine)], axis=-1).astype(np.float32)

        # 1. Compute SDF grid
        sdf_grid = compute_sdf_grid_from_coords(airfoil_xy, self.grid_res,
                                                  self.grid_domain)
        
        # 2. Prepare canonical surface points (HDF5-aligned resampling)
        if self.s_canon is not None:
            n_ord = compute_surface_normals(airfoil_xy)

            diffs = np.diff(airfoil_xy, axis=0)
            lens = np.linalg.norm(diffs, axis=1)
            arc_native = np.concatenate([[0], np.cumsum(lens)])
            arc_native /= arc_native[-1]

            from scipy.interpolate import interp1d
            f_x = interp1d(arc_native, airfoil_xy[:, 0], kind='linear', fill_value='extrapolate')
            f_y = interp1d(arc_native, airfoil_xy[:, 1], kind='linear', fill_value='extrapolate')
            surf_xy = np.stack([f_x(self.s_canon), f_y(self.s_canon)], axis=-1).astype(np.float32)

            f_nx = interp1d(arc_native, n_ord[:, 0], kind='linear', fill_value='extrapolate')
            f_ny = interp1d(arc_native, n_ord[:, 1], kind='linear', fill_value='extrapolate')
            surf_normals = np.stack([f_nx(self.s_canon), f_ny(self.s_canon)], axis=-1).astype(np.float32)
            surf_normals /= (np.linalg.norm(surf_normals, axis=-1, keepdims=True) + 1e-8)
            surf_normals *= self.n_sign[:, None]
        else:
            surf_xy = resample_curve(airfoil_xy, self.n_surface)
            surf_normals = compute_surface_normals(surf_xy)
        
        # 3. Encode flow conditions
        aoa_enc, re_enc, re_phys = self._encode_conditions(aoa_deg, re)
        
        # 4. Prepare query points
        if query_pts is None:
            # For force-only prediction, use surface points as dummy queries
            q_pts = surf_xy.copy()
            q_normals = surf_normals.copy()
        else:
            q_pts = np.asarray(query_pts, dtype=np.float32)
            # Compute SDF and normals at query points
            tree = KDTree(airfoil_xy)
            q_dists, q_idx = tree.query(q_pts)
            q_normals = compute_surface_normals(airfoil_xy)[q_idx % airfoil_xy.shape[0]]
        
        # Compute query SDF values
        tree = KDTree(airfoil_xy)
        q_sdf, _ = tree.query(q_pts)
        
        # 5. Convert to tensors and run forward pass
        with torch.no_grad():
            t_sdf_grid = torch.from_numpy(sdf_grid[None, None]).float().to(self.device)
            t_aoa = torch.from_numpy(aoa_enc[None]).float().to(self.device)
            t_re_enc = torch.from_numpy(re_enc[None]).float().to(self.device)
            t_re_phys = torch.tensor([[re_phys]]).float().to(self.device)
            t_q_pts = torch.from_numpy(q_pts[None]).float().to(self.device)
            t_q_sdf = torch.from_numpy(q_sdf[None]).float().to(self.device)
            t_q_norm = torch.from_numpy(q_normals[None]).float().to(self.device)
            t_surf = torch.from_numpy(surf_xy[None]).float().to(self.device)
            t_surf_n = torch.from_numpy(surf_normals[None]).float().to(self.device)
            
            if len(self.models) == 1:
                m = next(iter(self.models.values()))
                pred_field, pred_clcd = m(
                    t_sdf_grid, t_aoa, t_re_enc, t_q_pts, t_q_sdf, t_re_phys,
                    surf_pts=t_surf, query_normal=t_q_norm, surf_normal=t_surf_n
                )
                cl_cd_raw = pred_clcd[0].cpu().numpy()
                cl_cd_mean = m.cl_cd_mean.cpu().numpy()
                cl_cd_std = m.cl_cd_std.cpu().numpy()
                cl = float(cl_cd_raw[0] * cl_cd_std[0] + cl_cd_mean[0])
                cd = float(cl_cd_raw[1] * cl_cd_std[1] + cl_cd_mean[1])
                
                if return_field and query_pts is not None:
                    field_raw = pred_field[0].cpu().numpy()
                    field_mean = m.field_mean.cpu().numpy()
                    field_std = m.field_std.cpu().numpy()
                    field_phys = field_raw * field_std + field_mean
            else:
                # Multi-model adaptive ensembling across splits
                re_val = float(re)
                w_reynolds = 1.0 / (1.0 + np.exp((re_val - 1.8e6) / 2e5)) + 1.0 / (1.0 + np.exp((6.2e6 - re_val) / 2e5))
                w_reynolds = float(np.clip(w_reynolds, 0.0, 1.0))
                w_full = 1.0 - w_reynolds
                
                weights = {}
                if 'full' in self.models and 'reynolds' in self.models:
                    weights['full'] = w_full
                    weights['reynolds'] = w_reynolds
                else:
                    weights = {k: 1.0 / len(self.models) for k in self.models.keys()}
                
                cl_total = 0.0
                cd_total = 0.0
                field_phys_total = 0.0
                
                for k, m in self.models.items():
                    wk = weights.get(k, 1.0 / len(self.models))
                    p_field, p_clcd = m(
                        t_sdf_grid, t_aoa, t_re_enc, t_q_pts, t_q_sdf, t_re_phys,
                        surf_pts=t_surf, query_normal=t_q_norm, surf_normal=t_surf_n
                    )
                    clcd_raw = p_clcd[0].cpu().numpy()
                    clcd_m = m.cl_cd_mean.cpu().numpy()
                    clcd_s = m.cl_cd_std.cpu().numpy()
                    clk = float(clcd_raw[0] * clcd_s[0] + clcd_m[0])
                    cdk = float(clcd_raw[1] * clcd_s[1] + clcd_m[1])
                    
                    cl_total += wk * clk
                    cd_total += wk * cdk
                    
                    if return_field and query_pts is not None:
                        f_raw = p_field[0].cpu().numpy()
                        f_m = m.field_mean.cpu().numpy()
                        f_s = m.field_std.cpu().numpy()
                        fk = f_raw * f_s + f_m
                        field_phys_total = field_phys_total + wk * fk
                
                cl = cl_total
                cd = cd_total
                if return_field and query_pts is not None:
                    field_phys = field_phys_total
        
        result = {'Cl': cl, 'Cd': cd}
        if return_field and query_pts is not None:
            result['field'] = field_phys
        return result
    
    def predict_from_dat(self, dat_path, aoa_deg, re, **kwargs):
        """
        Predict Cl, Cd from a .dat airfoil file.
        
        Args:
            dat_path: path to .dat file (Selig or Lednicer format)
            aoa_deg: angle of attack in degrees
            re: Reynolds number
            
        Returns:
            dict with keys 'Cl', 'Cd'
        """
        airfoil_xy = read_dat_file(dat_path)
        return self.predict_from_coords(airfoil_xy, aoa_deg, re, **kwargs)
