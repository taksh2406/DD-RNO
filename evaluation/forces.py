import numpy as np
import torch


def integrate_pressure_numpy(p_points, line_indices, cell_normals, cell_lengths, aoa_rad, scaler=1.0):
    """
    Integrate pressure coefficients over native AirfRANS airfoil line cells.
    p_points is kinematic pressure (p/rho). 
    scaler is set to 1.0 to align with AirfRANS HDF5 stored labels (Cl_p, Cd_p).
    """
    p_points = np.asarray(p_points).reshape(-1)
    line_indices = np.asarray(line_indices, dtype=np.int64)
    cell_normals = np.asarray(cell_normals)
    cell_lengths = np.asarray(cell_lengths).reshape(-1)

    p_cell = p_points[line_indices].mean(axis=1)
    force = np.sum(p_cell[:, None] * cell_normals * cell_lengths[:, None], axis=0)

    cos_a = np.cos(float(aoa_rad))
    sin_a = np.sin(float(aoa_rad))
    
    # AirfRANS HDF5 Convention (Verified via Comprehensive Audit):
    # fx = sum(p*nx*dl), fy = sum(p*ny*dl)
    # CL =  fy*cos(a) - fx*sin(a)
    # CD =  fx*cos(a) + fy*sin(a)
    cl_p = ( force[1] * cos_a - force[0] * sin_a) * scaler
    cd_p = ( force[0] * cos_a + force[1] * sin_a) * scaler
    return float(cl_p), float(cd_p)


def integrate_pressure_torch(p_points, line_indices, cell_normals, cell_lengths, aoa_rad, scaler=1.0):
    """
    Batched torch version of native-cell pressure integration.
    """
    if p_points.dim() == 1:
        p_points = p_points.unsqueeze(0)
    if aoa_rad.dim() == 0:
        aoa_rad = aoa_rad.unsqueeze(0)
    elif aoa_rad.dim() == 2:
        aoa_rad = aoa_rad.squeeze(-1)

    if line_indices.max() >= p_points.size(1):
        raise ValueError(f"Integration Index Error: Max index {line_indices.max()} exceeds point count {p_points.size(1)}")

    # V20 Forensic Truth: AirfRANS HDF5 labels are generated from cell-averaged point pressure.
    # Formula: F_cell = p_avg * n_cell * l_cell.
    # Sign: Positive (Net Pressure pushes correctly in AirfRANS frame).
    p_cell = p_points[:, line_indices].mean(dim=-1)
    
    cell_normals = cell_normals.to(p_points.device)
    cell_lengths = cell_lengths.to(p_points.device).view(1, -1, 1)

    # Force vector in Cartesian coordinates
    force = (p_cell.unsqueeze(-1) * cell_normals.unsqueeze(0) * cell_lengths).sum(dim=1)

    cos_a = torch.cos(aoa_rad).view(-1)
    sin_a = torch.sin(aoa_rad).view(-1)
    
    # AirfRANS HDF5 Convention (Cl_p = Fy*cos(a) - Fx*sin(a))
    cl_p = ( force[:, 1] * cos_a - force[:, 0] * sin_a) * scaler
    cd_p = ( force[:, 0] * cos_a + force[:, 1] * sin_a) * scaler
    return cl_p, cd_p


def validate_pressure_labels(dataset, max_samples=10, atol=5e-4):
    """
    Check native-cell pressure integration against stored official pressure labels.

    This should be run after preprocessing and before training. A failure means
    the HDF5 force geometry is not faithful enough for publication metrics.
    """
    rows = []
    n = min(max_samples, len(dataset.data_cache))
    for i in range(n):
        sample = dataset.data_cache[i]
        cl_p, cd_p = integrate_pressure_numpy(
            sample["airfoil_p_native"],
            sample["airfoil_line_native"],
            sample["airfoil_cell_normal_native"],
            sample["airfoil_length_native"],
            sample["aoa_rad"],
        )
        cl_ref = sample["labels"]["Cl_p"]
        cd_ref = sample["labels"]["Cd_p"]
        rows.append(
            {
                "idx": i,
                "name": sample["name"],
                "cl_p": cl_p,
                "cl_p_ref": cl_ref,
                "cd_p": cd_p,
                "cd_p_ref": cd_ref,
                "cl_abs_err": abs(cl_p - cl_ref),
                "cd_abs_err": abs(cd_p - cd_ref),
            }
        )

    worst = max(rows, key=lambda r: max(r["cl_abs_err"], r["cd_abs_err"])) if rows else None
    if worst and max(worst["cl_abs_err"], worst["cd_abs_err"]) > atol:
        raise RuntimeError(
            "Pressure integration validation failed: "
            f"cd_err={worst['cd_abs_err']:.3e}"
        )
    return rows


def integrate_forces_torch(p, line, normal, length, aoa_rad, scaler=1.0):
    """
    Unified force integration. V21: Focused on bit-perfect Pressure (Cp).
    Viscous drag (Cd_v) is handled by the GP Corrector.
    """
    cl_p, cd_p = integrate_pressure_torch(p, line, normal, length, aoa_rad, scaler=scaler)
    return cl_p, cd_p 
