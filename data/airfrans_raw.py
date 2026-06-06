import h5py
import numpy as np

def load_airfrans_sample(h5_path, sample_idx):
    """Load one simulation from AirfRANS HDF5 (group-based format)."""
    with h5py.File(h5_path, 'r') as f:
        grp = f[f'sample_{sample_idx}']
        u      = grp['u'][:]
        v      = grp['v'][:]
        p      = grp['p'][:]
        sdf    = grp['sdf'][:]
        xy     = grp['xy'][:]
        normal = grp['normal'][:]
        re     = float(grp['Re'][0])
        aoa    = float(grp['alpha'][0])
    return dict(u=u, v=v, p=p, sdf=sdf, xy=xy, normal=normal, re=re, aoa=aoa)
