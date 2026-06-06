import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


LABEL_KEYS = ("Cl", "Cd", "Cl_p", "Cd_p", "Cl_v", "Cd_v")


def _resample_curve(values, n_fixed=1024):
    if values is None:
        return np.zeros((n_fixed, 2), dtype=np.float32)
    values = np.asarray(values)
    n_orig = values.shape[0]
    if n_orig == n_fixed:
        return values.astype(np.float32)
    x_orig = np.linspace(0.0, 1.0, n_orig)
    x_new = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_fixed)))
    if values.ndim == 1:
        return np.interp(x_new, x_orig, values).astype(np.float32)
    cols = [np.interp(x_new, x_orig, values[:, j]) for j in range(values.shape[1])]
    return np.stack(cols, axis=-1).astype(np.float32)


def _stats_from_cache(data_cache):
    n_total = 0
    sums = np.zeros(4, dtype=np.float64)
    sums_sq = np.zeros(4, dtype=np.float64)
    labels = []
    for d in data_cache:
        fields = np.stack([d["u"], d["v"], d["p"], d["nu_t"]], axis=-1).astype(np.float64)
        n_total += fields.shape[0]
        sums += fields.sum(axis=0)
        sums_sq += np.square(fields).sum(axis=0)
        labels.append(d["cl_cd_raw"])
    field_mean = sums / max(n_total, 1)
    field_var = np.maximum(sums_sq / max(n_total, 1) - field_mean**2, 0.0)
    field_std = np.sqrt(field_var) + 1e-6
    all_cl_cd = np.stack(labels)
    return (
        all_cl_cd.mean(axis=0).astype(np.float32),
        (all_cl_cd.std(axis=0) + 1e-6).astype(np.float32),
        field_mean.astype(np.float32),
        field_std.astype(np.float32),
    )


class AirfRANSDataset(Dataset):
    def __init__(
        self,
        h5_path,
        n_query=2048,
        external_stats=None,
        surface_points=1024,
        deterministic=False,
        seed=12345,
    ):
        self.h5_path = h5_path
        self.n_query = int(n_query)
        self.surface_points = int(surface_points)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.n_surf = int(0.20 * self.n_query)
        self.n_bl = int(0.60 * self.n_query)
        self.n_unif = self.n_query - self.n_surf - self.n_bl
        
        # Performance: Store only sample names, not the data
        with h5py.File(h5_path, "r") as f:
            self.sample_names = sorted(list(f.keys()))
            self.n_samples = len(self.sample_names)

        if external_stats is not None:
            self.cl_cd_mean = np.asarray(external_stats["cl_cd_mean"], dtype=np.float32)
            self.cl_cd_std = np.asarray(external_stats["cl_cd_std"], dtype=np.float32)
            self.field_mean = np.asarray(external_stats["field_mean"], dtype=np.float32)
            self.field_std = np.asarray(external_stats["field_std"], dtype=np.float32)
        else:
            # Memory-Efficient Dry Run for Stats
            print(f"Scanning {h5_path} for stats (Memory-Efficient)...")
            n_total = 0
            field_sums = np.zeros(4, dtype=np.float64)
            field_sums_sq = np.zeros(4, dtype=np.float64)
            label_list = []
            
            with h5py.File(h5_path, "r") as f:
                for name in self.sample_names:
                    grp = f[name]
                    # Read only target fields for stats
                    u = grp["u"][:].reshape(-1).astype(np.float64)
                    v = grp["v"][:].reshape(-1).astype(np.float64)
                    p = grp["p"][:].reshape(-1).astype(np.float64)
                    nut = grp["nu_t"][:].reshape(-1).astype(np.float64)
                    fields = np.stack([u, v, p, nut], axis=-1)
                    
                    n_total += fields.shape[0]
                    field_sums += fields.sum(axis=0)
                    field_sums_sq += np.square(fields).sum(axis=0)
                    
                    cl = float(grp["Cl"][0] if "Cl" in grp else 0.0)
                    cd = float(grp["Cd"][0] if "Cd" in grp else 0.0)
                    label_list.append([cl, cd])
            
            self.field_mean = (field_sums / max(n_total, 1)).astype(np.float32)
            field_var = np.maximum(field_sums_sq / max(n_total, 1) - self.field_mean**2, 0.0)
            self.field_std = (np.sqrt(field_var) + 1e-6).astype(np.float32)
            all_cl_cd = np.stack(label_list)
            self.cl_cd_mean = all_cl_cd.mean(axis=0).astype(np.float32)
            self.cl_cd_std = (all_cl_cd.std(axis=0) + 1e-6).astype(np.float32)

    def __len__(self):
        return self.n_samples

    def _load_sample(self, idx):
        """Lazy loader for sample data."""
        name = self.sample_names[idx]
        with h5py.File(self.h5_path, "r") as f:
            grp = f[name]
            sdf = grp["sdf"][:].reshape(-1)
            xy = grp["xy"][:]
            normal = grp["normal"][:]
            surface = grp["surface"][:].astype(bool) if "surface" in grp else (sdf < 1e-4)
            
            sample = {
                "name": name,
                "xy": xy,
                "surface": surface,
                "u": grp["u"][:].reshape(-1),
                "v": grp["v"][:].reshape(-1),
                "p": grp["p"][:].reshape(-1),
                "nu_t": grp["nu_t"][:].reshape(-1),
                "sdf": sdf,
                "normal": normal,
                "sdf_grid": grp["sdf_grid"][:],
                "Re": float(grp["Re"][0] if "Re" in grp else 1.0e6),
                "aoa_rad": float(grp["AoA_rad"][0] if "AoA_rad" in grp else 0.0),
                "tc": float(grp.attrs.get("tc", 0.12)),
                "camber": float(grp.attrs.get("camber", 0.0)),
                "labels": {k: float(grp[k][0] if k in grp else 0.0) for k in LABEL_KEYS},
            }
            sample["cl_cd_raw"] = np.array([float(grp["Cl"][0]), float(grp["Cd"][0])], dtype=np.float32)
            sample["cl_cd"] = (sample["cl_cd_raw"] - self.cl_cd_mean) / self.cl_cd_std
            
            sample["cl_cd_pressure_raw"] = np.array([float(grp["Cl_p"][0]), float(grp["Cd_p"][0])], dtype=np.float32)
            sample["surf_idx"] = np.where(surface)[0]
            
            # Standardization on-the-fly
            fields = np.stack([sample["u"], sample["v"], sample["p"], sample["nu_t"]], axis=-1)
            sample["target_standardized"] = ((fields - self.field_mean) / self.field_std).astype(np.float32)
            
            # Native Fields
            sample["airfoil_xy_native"] = grp["airfoil_xy"][:] if "airfoil_xy" in grp else None
            sample["airfoil_normal_native"] = grp["airfoil_normal"][:] if "airfoil_normal" in grp else None
            sample["airfoil_line_native"] = grp["airfoil_line"][:] if "airfoil_line" in grp else None
            sample["airfoil_length_native"] = grp["airfoil_length"][:] if "airfoil_length" in grp else None
            sample["airfoil_cell_normal_native"] = grp["airfoil_cell_normal"][:] if "airfoil_cell_normal" in grp else None
            sample["airfoil_p_native"] = grp["airfoil_p"][:].reshape(-1) if "airfoil_p" in grp else None
            sample["airfoil_phi_native"] = grp["airfoil_phi"][:] if "airfoil_phi" in grp else None
            
        return sample

    def __getitem__(self, idx):
        sample = self._load_sample(idx)
        rng = np.random.default_rng(self.seed + idx) if self.deterministic else np.random
        xy, sdf, normal = sample["xy"], sample["sdf"], sample["normal"]
        n = xy.shape[0]

        exact_surface_idx = sample["surf_idx"]
        if len(exact_surface_idx) == 0: exact_surface_idx = np.where(sdf < 1e-6)[0]
        
        bl_idx = np.where(sdf < 0.05)[0]
        if len(bl_idx) == 0: bl_idx = np.arange(n)

        # Sampling with LE bias
        surf_xy = xy[exact_surface_idx]
        if len(surf_xy) > 0:
            x_min = np.min(surf_xy[:, 0])
            weights = np.exp(-20.0 * (surf_xy[:, 0] - x_min))
            weights /= (np.sum(weights) + 1e-8)
            surf_probs = 0.8 * weights + 0.2 * (np.ones(len(surf_xy)) / len(surf_xy))
            surf_probs /= np.sum(surf_probs)
        else:
            surf_probs = None

        i_surf = rng.choice(exact_surface_idx, size=self.n_surf, replace=(len(exact_surface_idx) < self.n_surf), p=surf_probs)
        i_bl = rng.choice(bl_idx, size=self.n_bl, replace=(len(bl_idx) < self.n_bl))
        i_unif = rng.choice(n, size=self.n_unif, replace=(n < self.n_unif))
        idx_q = np.concatenate([i_surf, i_bl, i_unif])

        # Condition encoding
        alpha_rad = sample["aoa_rad"]
        aoa_enc = np.array([np.sin(alpha_rad), np.cos(alpha_rad), np.sin(2.0 * alpha_rad)], dtype=np.float32)
        re = sample["Re"]
        re_enc = np.array([np.log10(max(re, 1e4) / 1e6)], dtype=np.float32)

        # Native Mesh handling (Robust)
        if sample["airfoil_xy_native"] is not None:
            airfoil_xy = _resample_curve(sample["airfoil_xy_native"], self.surface_points)
            airfoil_n = _resample_curve(sample["airfoil_normal_native"], self.surface_points)
            airfoil_n /= (np.linalg.norm(airfoil_n, axis=-1, keepdims=True) + 1e-8)
            airfoil_p = _resample_curve(sample["airfoil_p_native"], self.surface_points)
        else:
            airfoil_idx = exact_surface_idx[:self.surface_points]
            airfoil_xy = xy[airfoil_idx]
            airfoil_n = normal[airfoil_idx]
            airfoil_p = sample["p"][airfoil_idx]

        airfoil_t = np.stack([-airfoil_n[:, 1], airfoil_n[:, 0]], axis=-1).astype(np.float32)
        re_sqrt = float(np.sqrt(max(re, 1e4)))

        return {
            "sdf_grid": torch.from_numpy(sample["sdf_grid"][None]).float(),
            "aoa_enc": torch.from_numpy(aoa_enc),
            "aoa_rad": torch.tensor([alpha_rad]).float(),
            "re_enc": torch.from_numpy(re_enc),
            "re_phys": torch.tensor([re]).float(),
            "tc": torch.tensor([sample["tc"]]).float(),
            "camber": torch.tensor([sample["camber"]]).float(),
            "query_pts": torch.from_numpy(xy[idx_q]).float(),
            "query_phi": torch.from_numpy(sdf[idx_q, None]).float(),
            "query_phi_re": torch.from_numpy(sdf[idx_q, None] * re_sqrt).float(),
            "query_normal": torch.from_numpy(normal[idx_q]).float(),
            "target": torch.from_numpy(sample["target_standardized"][idx_q]).float(),
            "cl_cd_raw": torch.from_numpy(sample["cl_cd_raw"]).float(),
            "cl_cd_pressure_raw": torch.from_numpy(sample["cl_cd_pressure_raw"]).float(),
            "surf_xy": torch.from_numpy(airfoil_xy).float(),
            "surf_phi": torch.from_numpy(np.zeros((len(airfoil_xy), 1), dtype=np.float32)).float(),
            "surf_phi_re": torch.from_numpy(np.zeros((len(airfoil_xy), 1), dtype=np.float32)).float(),
            "surf_normal": torch.from_numpy(airfoil_n).float(),
            "surf_tangent": torch.from_numpy(airfoil_t).float(),
            "surf_p": torch.from_numpy(airfoil_p).float(),
            "airfoil_line_native": torch.from_numpy(sample["airfoil_line_native"]).long() if sample["airfoil_line_native"] is not None else None,
            "airfoil_normal_native": torch.from_numpy(sample["airfoil_cell_normal_native"]).float() if sample["airfoil_cell_normal_native"] is not None else None,
            "airfoil_length_native": torch.from_numpy(sample["airfoil_length_native"]).float() if sample["airfoil_length_native"] is not None else None,
            "airfoil_xy_native": torch.from_numpy(sample["airfoil_xy_native"]).float() if sample["airfoil_xy_native"] is not None else None,
            "airfoil_phi_native": torch.from_numpy(sample["airfoil_phi_native"]).float() if sample["airfoil_phi_native"] is not None else None,
        }


def physics_collate(batch):
    meta_keys = {"airfoil_line_native", "airfoil_normal_native", "airfoil_length_native", "airfoil_xy_native", "airfoil_phi_native"}
    elem = batch[0]
    out = {}
    for key in elem.keys():
        if key in meta_keys:
            out[key] = [d.get(key) for d in batch]
        else:
            from torch.utils.data._utils.collate import default_collate
            try:
                out[key] = default_collate([d[key] for d in batch])
            except Exception:
                out[key] = [d[key] for d in batch]
    return out


def get_loaders(train_h5, test_h5, val_fraction=0.2, batch_size=4, n_query=2048, num_workers=4):
    full_train = AirfRANSDataset(train_h5, n_query=n_query)
    stats = {"cl_cd_mean": full_train.cl_cd_mean, "cl_cd_std": full_train.cl_cd_std, "field_mean": full_train.field_mean, "field_std": full_train.field_std}
    val_full = AirfRANSDataset(train_h5, n_query=n_query, external_stats=stats, deterministic=True)
    test_ds = AirfRANSDataset(test_h5, n_query=n_query, external_stats=stats, deterministic=True)

    n_total = len(full_train)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    n_val = int(n_total * val_fraction)
    idx_val, idx_train = indices[:n_val], indices[n_val:]

    train_ds = Subset(full_train, idx_train.tolist())
    val_ds = Subset(val_full, idx_val.tolist())

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=physics_collate),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=physics_collate),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=physics_collate),
    )
