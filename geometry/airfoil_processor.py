import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw
import os

import math

class AirfoilProcessor:
    def __init__(self, grid_res=64, domain=((-0.5, 1.5), (-1.0, 1.0))):
        self.grid_res = grid_res
        self.domain = domain
        self.xmin, self.xmax = domain[0]
        self.ymin, self.ymax = domain[1]

    @staticmethod
    def cst_to_coords(cst_params, num_points=256):
        """
        B5 Fix: Core CST -> coordinates pipeline.
        cst_params: array-like of (N_upper + N_lower + 2).
        For classical 12-param: 5 upper, 5 lower, plus TE thickness.
        Let's assume cst_params = [Au_0..Au_N, Al_0..Al_N, dz_te_u, dz_te_l]
        """
        N = (len(cst_params) - 2) // 2
        Au = cst_params[:N]
        Al = cst_params[N:2*N]
        dz_te_u = cst_params[-2]
        dz_te_l = cst_params[-1]
        
        x = np.linspace(1, 0, num_points//2) # TE to LE (upper)
        x_low = np.linspace(0, 1, num_points//2)[1:] # LE to TE (lower)
        
        def basis(x_val, A):
            # C(x) = sqrt(x) * (1-x)
            C = np.sqrt(x_val) * (1 - x_val)
            S = np.zeros_like(x_val)
            n_order = len(A) - 1
            for i, w in enumerate(A):
                K = math.factorial(n_order) / (math.factorial(i) * math.factorial(n_order - i))
                S += w * K * (x_val**i) * ((1 - x_val)**(n_order - i))
            return C * S
            
        y_u = basis(x, Au) + x * dz_te_u
        y_l = basis(x_low, Al) + x_low * dz_te_l
        
        upper = np.stack([x, y_u], axis=-1)
        lower = np.stack([x_low, y_l], axis=-1)
        return np.concatenate([upper, lower], axis=0)

    def load_dat(self, path):
        """Loads airfoil coordinates from a .dat file (UIUC/Selig format)."""
        coords = []
        with open(path, 'r') as f:
            lines = f.readlines()
            start_idx = 0
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        float(parts[0])
                        start_idx = i
                        break
                    except ValueError:
                        continue
            
            for line in lines[start_idx:]:
                parts = line.split()
                if len(parts) == 2:
                    coords.append([float(parts[0]), float(parts[1])])
        
        pts = np.array(coords)
        
        # B6 Fix: Robust Normalization for the 'Black Box' Surrogate
        # 1. Scaling to chord = 1.0
        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        chord = x_max - x_min
        pts[:, 0] = (pts[:, 0] - x_min) / chord
        pts[:, 1] = pts[:, 1] / chord
        
        # 2. Centering (LE at 0,0) - assuming first or last point is TE, middle is LE
        # We'll just use the actual min-x point as the anchor
        le_idx = np.argmin(pts[:, 0])
        le_y = pts[le_idx, 1]
        pts[:, 1] = pts[:, 1] - le_y
        pts[le_idx, 0] = 0.0
        pts[le_idx, 1] = 0.0
        
        return pts

    def process_geometry(self, coords, n_surface=512):
        """
        Processes raw coordinates into surrogate-ready inputs:
        1. 64x64 SDF Grid
        2. Surface normals
        3. Segment lengths (ds) for integration
        """
        # A. Interpolate and Sort (Trailing Edge to Leading Edge to Trailing Edge)
        # Ensure we have a clean surface for integration
        # For simplicity, we assume the input is already fairly clean. 
        # In a production version, we'd use splines.
        
        # B. Compute Normals and Segment Lengths
        # x_i, y_i
        dx = np.diff(coords[:, 0])
        dy = np.diff(coords[:, 1])
        ds = np.sqrt(dx**2 + dy**2)
        
        # Segment midpoints for query
        q_pts = (coords[:-1] + coords[1:]) / 2.0
        
        # Outward normal (2D cross product with segment vector)
        # For CCW ordering, normal is (dy, -dx) / ds
        nx = dy / (ds + 1e-8)
        ny = -dx / (ds + 1e-8)
        normals = np.stack([nx, ny], axis=-1)
        
        # C. Generate 64x64 SDF Grid
        sdf_grid = self.rasterize_sdf(coords)
        
        return {
            'q_pts': q_pts,
            'normals': normals,
            'ds': ds,
            'sdf_grid': sdf_grid
        }

    def rasterize_sdf(self, coords):
        """Generates 64x64 SDF grid from polygon coordinates."""
        # 1. Map to grid indices
        xs = ((coords[:, 0] - self.xmin) / (self.xmax - self.xmin) * self.grid_res).astype(int)
        ys = ((coords[:, 1] - self.ymin) / (self.ymax - self.ymin) * self.grid_res).astype(int)
        xs = np.clip(xs, 0, self.grid_res - 1)
        ys = np.clip(ys, 0, self.grid_res - 1)
        
        # 2. Draw and Fill Polygon
        img = Image.new('L', (self.grid_res, self.grid_res), 0)
        draw = ImageDraw.Draw(img)
        pts = list(zip(xs.tolist(), ys.tolist()))
        draw.polygon(pts, fill=255)
        mask = np.array(img) > 0
        
        # 3. Distance Transform
        dist_out = distance_transform_edt(~mask)
        dist_in  = distance_transform_edt(mask)
        sdf = (dist_out - dist_in) / (self.grid_res / (self.xmax - self.xmin))
        
        return sdf.astype(np.float32)

if __name__ == "__main__":
    # Test
    proc = AirfoilProcessor()
    # Mock some coordinates
    theta = np.linspace(0, 2*np.pi, 100)
    coords = np.stack([0.5 + 0.5*np.cos(theta), 0.1*np.sin(theta)], axis=1)
    data = proc.process_geometry(coords)
    print(f"SDF Grid Shape: {data['sdf_grid'].shape}")
    print(f"Normals Shape: {data['normals'].shape}")
