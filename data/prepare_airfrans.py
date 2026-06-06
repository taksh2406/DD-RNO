import json
import os

import airfrans as af
import h5py
import numpy as np
from scipy.interpolate import griddata


def manifest_names(root_dir, task="full", train=True):
    split = "train" if train else "test"
    task_key = "full" if task == "scarce" and not train else task
    manifest_path = os.path.join(root_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    key = f"{task_key}_{split}"
    if key not in manifest:
        raise KeyError(f"{key!r} not found in {manifest_path}")
    return list(manifest[key])


def compute_sdf_grid(sim, res=64, domain=((-0.5, 1.5), (-1.0, 1.0))):
    """Rasterize raw mesh SDF onto the fixed geometry grid used by the encoder."""
    xy = sim.position[:, 0:2]
    sdf = sim.sdf.reshape(-1)
    xi = np.linspace(domain[0][0], domain[0][1], res)
    yi = np.linspace(domain[1][0], domain[1][1], res)
    xx, yy = np.meshgrid(xi, yi)
    return griddata(xy, sdf, (xx, yy), method="nearest", fill_value=1.0).astype(np.float32)


def normalise_fields(sim):
    """
    Return dimensionless fields in the convention verified against raw AirfRANS.

    AirfRANS stores pressure as pressure/rho, so Cp scaling is p/(0.5 U_inf^2).
    Turbulent viscosity is normalized by molecular kinematic viscosity, matching
    the AirfRANS boundary-layer API.
    """
    u_inf = float(sim.inlet_velocity)
    nu = float(sim.NU)
    q_inf = 0.5 * u_inf**2
    u = sim.velocity[:, 0] / u_inf
    v = sim.velocity[:, 1] / u_inf
    p = sim.pressure.reshape(-1) / q_inf
    nu_t = sim.nu_t.reshape(-1) / nu
    return u.astype(np.float32), v.astype(np.float32), p.astype(np.float32), nu_t.astype(np.float32)


def normalise_airfoil_fields(sim):
    """Native ordered airfoil-patch fields, normalized consistently with the volume."""
    u_inf = float(sim.inlet_velocity)
    nu = float(sim.NU)
    q_inf = 0.5 * u_inf**2
    velocity = np.asarray(sim.airfoil.point_data["U"][:, :2], dtype=np.float64)
    pressure = np.asarray(sim.airfoil.point_data["p"], dtype=np.float64).reshape(-1)
    nu_t = np.asarray(sim.airfoil.point_data["nut"], dtype=np.float64).reshape(-1)
    return (
        (velocity[:, 0] / u_inf).astype(np.float32),
        (velocity[:, 1] / u_inf).astype(np.float32),
        (pressure / q_inf).astype(np.float32),
        (nu_t / nu).astype(np.float32),
    )


def force_coefficients(sim):
    """Official AirfRANS total, pressure, and viscous force coefficients."""
    (cd, cd_p, cd_v), (cl, cl_p, cl_v) = sim.force_coefficient(reference=True)
    return {
        "Cd": float(cd),
        "Cd_p": float(cd_p),
        "Cd_v": float(cd_v),
        "Cl": float(cl),
        "Cl_p": float(cl_p),
        "Cl_v": float(cl_v),
    }


def airfoil_cell_geometry(sim):
    """Native airfoil line-cell connectivity, lengths, and cell normals."""
    lines = np.asarray(sim.airfoil.lines).reshape(-1, 3)[:, 1:].astype(np.int64)
    airfoil_cells = sim.airfoil.ptc(pass_point_data=False)
    cell_normals = np.asarray(airfoil_cells.cell_data["Normals"][:, :2], dtype=np.float32)
    lengths = np.asarray(sim.airfoil.cell_data["Length"], dtype=np.float32)
    return lines, lengths, cell_normals


def prepare_h5(root_dir, output_path, task="full", train=True):
    """
    Convert raw AirfRANS simulations into a faithful HDF5 tensor cache.

    The raw simulations remain the source of truth. This cache preserves the
    full internal mesh fields, exact surface mask, native airfoil patch geometry,
    physical metadata, and official AirfRANS force coefficients.
    """
    split = "train" if train else "test"
    names = manifest_names(root_dir, task=task, train=train)
    expected = 800 if task == "full" and train else 200 if task in {"full", "scarce"} else None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Preparing AirfRANS task={task} split={split} from {root_dir}")
    print(f"Writing {len(names)} samples to {output_path}")
    if expected is not None and len(names) != expected:
        print(f" ! Warning: expected {expected} samples, manifest has {len(names)}")

    with h5py.File(output_path, "w") as f:
        f.attrs["source_root"] = root_dir
        f.attrs["task"] = task
        f.attrs["split"] = split
        f.attrs["field_scaling"] = "u,v: /U_inf; p: /(0.5*U_inf^2); nu_t: /NU"

        for i, name in enumerate(names):
            try:
                sim = af.Simulation(root_dir, name)
                u, v, p, nu_t = normalise_fields(sim)
                af_u, af_v, af_p, af_nu_t = normalise_airfoil_fields(sim)
                coeffs = force_coefficients(sim)
                airfoil_lines, airfoil_lengths, airfoil_cell_normals = airfoil_cell_geometry(sim)

                grp = f.create_group(f"sample_{i}")
                grp.attrs["name"] = name

                grp.create_dataset("xy", data=sim.position[:, 0:2].astype(np.float32))
                grp.create_dataset("surface", data=np.asarray(sim.surface, dtype=np.bool_))
                grp.create_dataset("sdf", data=sim.sdf.reshape(-1).astype(np.float32))
                grp.create_dataset("normal", data=sim.normals.astype(np.float32))
                grp.create_dataset("u", data=u)
                grp.create_dataset("v", data=v)
                grp.create_dataset("p", data=p)
                grp.create_dataset("nu_t", data=nu_t)
                grp.create_dataset("sdf_grid", data=compute_sdf_grid(sim))

                grp.create_dataset("airfoil_xy", data=sim.airfoil_position.astype(np.float32))
                grp.create_dataset("airfoil_normal", data=sim.airfoil_normals.astype(np.float32))
                grp.create_dataset("airfoil_line", data=airfoil_lines)
                grp.create_dataset("airfoil_length", data=airfoil_lengths)
                grp.create_dataset("airfoil_cell_normal", data=airfoil_cell_normals)
                grp.create_dataset("airfoil_u", data=af_u)
                grp.create_dataset("airfoil_v", data=af_v)
                grp.create_dataset("airfoil_p", data=af_p)
                grp.create_dataset("airfoil_nu_t", data=af_nu_t)

                grp.create_dataset("Re", data=np.array([float(sim.inlet_velocity) / float(sim.NU)], dtype=np.float32))
                grp.create_dataset("AoA_rad", data=np.array([float(sim.angle_of_attack)], dtype=np.float32))
                grp.create_dataset("AoA_deg", data=np.array([float(np.degrees(sim.angle_of_attack))], dtype=np.float32))
                grp.create_dataset("U_inf", data=np.array([float(sim.inlet_velocity)], dtype=np.float32))
                grp.create_dataset("NU", data=np.array([float(sim.NU)], dtype=np.float32))
                grp.create_dataset("RHO", data=np.array([float(sim.RHO)], dtype=np.float32))

                for key, value in coeffs.items():
                    grp.create_dataset(key, data=np.array([value], dtype=np.float32))

                if i % 20 == 0 or i == len(names) - 1:
                    print(
                        f"  {i:04d}/{len(names)} {name} "
                        f"Re={float(sim.inlet_velocity) / float(sim.NU):.0f} "
                        f"AoA={float(np.degrees(sim.angle_of_attack)):.3f} "
                        f"Cl={coeffs['Cl']:.4f} Cd={coeffs['Cd']:.5f}"
                    )
            except Exception as exc:
                print(f"  !! Error on {i} {name}: {type(exc).__name__}: {exc}")
                raise

    print(f"Saved {len(names)} samples -> {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="airfrans_data/raw/airfrans_dataset")
    parser.add_argument("--train_out", default="airfrans_data/processed/airfrans_full_train.h5")
    parser.add_argument("--test_out", default="airfrans_data/processed/airfrans_full_test.h5")
    parser.add_argument("--task", default="full", choices=["full", "scarce", "reynolds", "aoa"])
    args = parser.parse_args()

    prepare_h5(args.root, args.train_out, task=args.task, train=True)
    prepare_h5(args.root, args.test_out, task=args.task, train=False)
