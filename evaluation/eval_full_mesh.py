import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.airfrans_dataset import AirfRANSDataset, physics_collate
from model.dd_rno import DDRNO
from evaluation.forces import integrate_pressure_torch

def _flatten_config(config):
    if 'data' not in config:
        return dict(config)
    flat = {}
    for section in ('data', 'model', 'training', 'evaluation'):
        flat.update(config.get(section, {}))
    return flat

def rel_l2_error(pred, target):
    # Both inputs expected to be torch tensors of shape (N,) or (N, D)
    num = torch.norm(pred - target)
    den = torch.norm(target) + 1e-8
    return float((num / den).item())

def evaluate_full_mesh(ckpt_path, config_path, split="test", chunk_size=16384, device="cuda", out_path=None):
    with open(config_path, 'r') as f:
        config = _flatten_config(yaml.safe_load(f))

    # Initialize dataset just to extract baseline normalization stats
    train_ds = AirfRANSDataset(config["train_h5"], n_query=config.get("query_points_per_sample", 2048))
    stats = {
        "cl_cd_mean": train_ds.cl_cd_mean,
        "cl_cd_std": train_ds.cl_cd_std,
        "field_mean": train_ds.field_mean,
        "field_std": train_ds.field_std,
    }

    eval_h5 = config["test_h5"] if split == "test" else config["train_h5"]
    print(f"Loading split '{split}' from {eval_h5}...")
    ds = AirfRANSDataset(eval_h5, n_query=config.get("query_points_per_sample", 2048), external_stats=stats, deterministic=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=physics_collate)

    use_domain_routing = config.get('use_domain_routing', True)
    use_lcq = config.get('use_lcq', True)
    use_specinr = config.get('use_specinr', True)

    # Automatically detect ablation overrides from checkpoint path
    if "abl_no_routing" in ckpt_path:
        use_domain_routing = False
    if "abl_no_lcq" in ckpt_path:
        use_lcq = False
    if "abl_no_specinr" in ckpt_path:
        use_specinr = False

    # Initialize model
    model = DDRNO(
        geom_dim=config.get('geom_dim', 128),
        w_dim=config.get('w_dim', 64),
        hidden=config.get('hidden', 256),
        n_fourier=config.get('n_fourier', 8),
        n_canonical=config.get('n_canonical', 1024),
        inv_layers=config.get('inv_layers', 4),
        bl_layers=config.get('bl_layers', 6),
        wake_layers=config.get('wake_layers', 4),
        use_domain_routing=use_domain_routing,
        use_lcq=use_lcq,
        use_specinr=use_specinr,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.set_standardization_stats(
        stats['cl_cd_mean'], stats['cl_cd_std'],
        field_mean=stats['field_mean'], field_std=stats['field_std']
    )
    model.eval()

    field_mean = torch.tensor(stats["field_mean"], device=device).view(1, 1, 4)
    field_std = torch.tensor(stats["field_std"], device=device).view(1, 1, 4)
    cl_cd_mean = torch.tensor(stats["cl_cd_mean"], device=device).view(1, 2)
    cl_cd_std = torch.tensor(stats["cl_cd_std"], device=device).view(1, 2)

    results = []
    times = []

    print(f"Starting evaluation of {len(ds)} samples...")
    with torch.no_grad():
        for i in range(len(ds)):
            t0 = time.time()
            # 1. Load full raw sample properties from dataset
            sample = ds._load_sample(i)
            
            # Batch inputs (all size 1)
            sdf_grid = torch.from_numpy(sample["sdf_grid"][None, None]).float().to(device)
            # Reconstruct aoa and re encodings
            alpha_rad = sample["aoa_rad"]
            aoa_enc = torch.tensor([np.sin(alpha_rad), np.cos(alpha_rad), np.sin(2.0 * alpha_rad)], dtype=torch.float32).unsqueeze(0).to(device)
            re = sample["Re"]
            re_enc = torch.tensor([np.log10(max(re, 1e4) / 1e6)], dtype=torch.float32).unsqueeze(0).to(device)
            re_phys = torch.tensor([re], dtype=torch.float32).unsqueeze(0).to(device)
            aoa_rad_t = torch.tensor([alpha_rad], dtype=torch.float32).unsqueeze(0).to(device)

            # Native airfoil properties for force integration
            airfoil_xy = torch.from_numpy(sample["airfoil_xy_native"][None]).float().to(device)
            airfoil_normal = torch.from_numpy(sample["airfoil_normal_native"][None]).float().to(device)
            airfoil_phi = torch.zeros((1, airfoil_xy.shape[1]), dtype=torch.float32).to(device)
            airfoil_line = torch.from_numpy(sample["airfoil_line_native"]).long().to(device)
            airfoil_cell_normal = torch.from_numpy(sample["airfoil_cell_normal_native"]).float().to(device)
            airfoil_length = torch.from_numpy(sample["airfoil_length_native"]).float().to(device)

            # Predict forces via LCQ head
            # For LCQ surface input, we pass the standard 1024 surface points that the model expects.
            # In AirfRANSDataset, surf_xy is resampled from native coordinates. Let's use the resampled ones.
            # Let's get the standard item to obtain surf_xy and surf_normal
            item = ds[i]
            surf_xy = item["surf_xy"].unsqueeze(0).to(device)
            surf_normal = item["surf_normal"].unsqueeze(0).to(device)
            
            # Forward pass for LCQ prediction
            _, pred_clcd_std = model(
                sdf_grid=sdf_grid,
                aoa_enc=aoa_enc,
                re_enc=re_enc,
                query_pts=surf_xy,
                query_sdf=torch.zeros((1, surf_xy.shape[1]), device=device),
                re_phys=re_phys,
                surf_pts=surf_xy,
                query_normal=surf_normal,
                surf_normal=surf_normal
            )
            cl_cd_pred_lcq = (pred_clcd_std * cl_cd_std + cl_cd_mean).squeeze(0).cpu().numpy()

            # 2. Predict fields on full native mesh using chunking
            xy_full = torch.from_numpy(sample["xy"]).float().to(device)
            sdf_full = torch.from_numpy(sample["sdf"]).float().to(device)
            normal_full = torch.from_numpy(sample["normal"]).float().to(device)
            target_full_std = torch.from_numpy(sample["target_standardized"]).float().to(device)

            num_points = xy_full.shape[0]
            pred_field_std_list = []

            for start_idx in range(0, num_points, chunk_size):
                end_idx = min(start_idx + chunk_size, num_points)
                pts_chunk = xy_full[start_idx:end_idx].unsqueeze(0)
                sdf_chunk = sdf_full[start_idx:end_idx].unsqueeze(0)
                normal_chunk = normal_full[start_idx:end_idx].unsqueeze(0)

                pred_chunk_std, _ = model(
                    sdf_grid=sdf_grid,
                    aoa_enc=aoa_enc,
                    re_enc=re_enc,
                    query_pts=pts_chunk,
                    query_sdf=sdf_chunk,
                    re_phys=re_phys,
                    surf_pts=None,
                    query_normal=normal_chunk,
                    surf_normal=None
                )
                pred_field_std_list.append(pred_chunk_std.squeeze(0))

            pred_field_std = torch.cat(pred_field_std_list, dim=0)

            # Denormalize predicted and target fields to physical units
            pred_field_phys = pred_field_std * field_std.squeeze(0) + field_mean.squeeze(0)
            target_field_phys = target_full_std * field_std.squeeze(0) + field_mean.squeeze(0)

            # 3. Predict field at native airfoil boundary nodes for pressure integration
            # We want to run the model at airfoil_xy with its node normals
            pred_airfoil_std, _ = model(
                sdf_grid=sdf_grid,
                aoa_enc=aoa_enc,
                re_enc=re_enc,
                query_pts=airfoil_xy,
                query_sdf=airfoil_phi,
                re_phys=re_phys,
                surf_pts=None,
                query_normal=airfoil_normal,
                surf_normal=None
            )
            pred_airfoil_phys = pred_airfoil_std * field_std.squeeze(0) + field_mean.squeeze(0)
            p_pred_airfoil = pred_airfoil_phys[0, :, 2] # Kinematic pressure

            # Integrate predicted pressure to find Cl and Cd
            cl_pred_int, cd_pred_int = integrate_pressure_torch(
                p_pred_airfoil.unsqueeze(0),
                airfoil_line,
                airfoil_cell_normal,
                airfoil_length,
                aoa_rad_t
            )
            cl_pred_int = float(cl_pred_int.item())
            cd_pred_int = float(cd_pred_int.item())

            # 4. Compute field evaluation metrics on full mesh
            # Standardized MSE (matches AirfRANS official benchmark metric)
            mse_vol = torch.mean((pred_field_std - target_full_std)**2, dim=0).cpu().numpy()
            
            # Surface-only standardized MSE
            surf_mask = sample["surface"]
            surf_mask_t = torch.from_numpy(surf_mask).to(device)
            if surf_mask_t.any():
                mse_surf = torch.mean((pred_field_std[surf_mask_t] - target_full_std[surf_mask_t])**2, dim=0).cpu().numpy()
            else:
                mse_surf = np.zeros(4)

            # Relative L2 error in physical units (velocity component u, v and pressure p)
            rel_l2_u = rel_l2_error(pred_field_phys[:, 0], target_field_phys[:, 0])
            rel_l2_v = rel_l2_error(pred_field_phys[:, 1], target_field_phys[:, 1])
            rel_l2_p = rel_l2_error(pred_field_phys[:, 2], target_field_phys[:, 2])

            dt = time.time() - t0
            times.append(dt)

            # Store result
            cl_gt = float(sample["cl_cd_raw"][0])
            cd_gt = float(sample["cl_cd_raw"][1])
            cl_p_gt = float(sample["cl_cd_pressure_raw"][0])
            cd_p_gt = float(sample["cl_cd_pressure_raw"][1])

            res = {
                "idx": i,
                "name": sample["name"],
                "Re": float(re),
                "AoA": float(alpha_rad * 180.0 / np.pi),
                # Ground truth
                "Cl_gt": cl_gt,
                "Cd_gt": cd_gt,
                "Cl_p_gt": cl_p_gt,
                "Cd_p_gt": cd_p_gt,
                # LCQ Prediction
                "Cl_lcq": float(cl_cd_pred_lcq[0]),
                "Cd_lcq": float(cl_cd_pred_lcq[1]),
                # Pressure-Integration Prediction
                "Cl_int": cl_pred_int,
                "Cd_int": cd_pred_int,
                # Field metrics
                "mse_u": float(mse_vol[0]),
                "mse_v": float(mse_vol[1]),
                "mse_p": float(mse_vol[2]),
                "mse_nut": float(mse_vol[3]),
                "surf_mse_u": float(mse_surf[0]),
                "surf_mse_v": float(mse_surf[1]),
                "surf_mse_p": float(mse_surf[2]),
                "surf_mse_nut": float(mse_surf[3]),
                "rel_l2_u": rel_l2_u,
                "rel_l2_v": rel_l2_v,
                "rel_l2_p": rel_l2_p,
                "time": dt
            }
            results.append(res)

            if i % 10 == 0:
                print(f"Sample {i:03d}/{len(ds)}: Cl_gt={cl_gt:.4f}, Cl_lcq={res['Cl_lcq']:.4f}, Cl_int={cl_pred_int:.4f}, p_L2={rel_l2_p*100:.2f}%, time={dt*1000:.1f}ms")

    # Aggregate summaries
    n_samples = len(results)
    avg_metrics = {
        "mse_u": float(np.mean([r["mse_u"] for r in results])),
        "mse_v": float(np.mean([r["mse_v"] for r in results])),
        "mse_p": float(np.mean([r["mse_p"] for r in results])),
        "mse_nut": float(np.mean([r["mse_nut"] for r in results])),
        
        "surf_mse_u": float(np.mean([r["surf_mse_u"] for r in results])),
        "surf_mse_v": float(np.mean([r["surf_mse_v"] for r in results])),
        "surf_mse_p": float(np.mean([r["surf_mse_p"] for r in results])),
        "surf_mse_nut": float(np.mean([r["surf_mse_nut"] for r in results])),

        "rel_l2_u": float(np.mean([r["rel_l2_u"] for r in results])),
        "rel_l2_v": float(np.mean([r["rel_l2_v"] for r in results])),
        "rel_l2_p": float(np.mean([r["rel_l2_p"] for r in results])),

        # LCQ errors
        "cl_lcq_mae": float(np.mean([abs(r["Cl_lcq"] - r["Cl_gt"]) for r in results])),
        "cd_lcq_mae": float(np.mean([abs(r["Cd_lcq"] - r["Cd_gt"]) for r in results])),
        "cl_lcq_rel_err": float(np.mean([abs(r["Cl_lcq"] - r["Cl_gt"]) / max(abs(r["Cl_gt"]), 1e-8) for r in results])),
        "cd_lcq_rel_err": float(np.mean([abs(r["Cd_lcq"] - r["Cd_gt"]) / max(abs(r["Cd_gt"]), 1e-8) for r in results])),

        # Pressure-integrated errors (compared against pressure-only true force labels Cl_p_gt, Cd_p_gt)
        "cl_int_mae": float(np.mean([abs(r["Cl_int"] - r["Cl_p_gt"]) for r in results])),
        "cd_int_mae": float(np.mean([abs(r["Cd_int"] - r["Cd_p_gt"]) for r in results])),
        "cl_int_rel_err": float(np.mean([abs(r["Cl_int"] - r["Cl_p_gt"]) / max(abs(r["Cl_p_gt"]), 1e-8) for r in results])),
        "cd_int_rel_err": float(np.mean([abs(r["Cd_int"] - r["Cd_p_gt"]) / max(abs(r["Cd_p_gt"]), 1e-8) for r in results])),

        "mean_time_ms": float(np.mean(times) * 1000.0),
        "n_samples": n_samples,
        "split": split,
        "checkpoint": ckpt_path
    }

    # Spearman rank correlation coefficients
    gt_cl = [r["Cl_gt"] for r in results]
    gt_cd = [r["Cd_gt"] for r in results]
    pred_cl_lcq = [r["Cl_lcq"] for r in results]
    pred_cd_lcq = [r["Cd_lcq"] for r in results]
    gt_cl_p = [r["Cl_p_gt"] for r in results]
    gt_cd_p = [r["Cd_p_gt"] for r in results]
    pred_cl_int = [r["Cl_int"] for r in results]
    pred_cd_int = [r["Cd_int"] for r in results]

    avg_metrics["cl_lcq_spearman"] = float(spearmanr(gt_cl, pred_cl_lcq).correlation)
    avg_metrics["cd_lcq_spearman"] = float(spearmanr(gt_cd, pred_cd_lcq).correlation)
    avg_metrics["cl_int_spearman"] = float(spearmanr(gt_cl_p, pred_cl_int).correlation)
    avg_metrics["cd_int_spearman"] = float(spearmanr(gt_cd_p, pred_cd_int).correlation)

    print("\n" + "="*50)
    print(f" FULL MESH EVALUATION RESULTS FOR {split.upper()}")
    print("="*50)
    print(f" u_REL_L2  : {avg_metrics['rel_l2_u']*100:.4f}%")
    print(f" v_REL_L2  : {avg_metrics['rel_l2_v']*100:.4f}%")
    print(f" p_REL_L2  : {avg_metrics['rel_l2_p']*100:.4f}%")
    print("-" * 50)
    print(f" Cl LCQ MAE: {avg_metrics['cl_lcq_mae']:.6f} (Rel: {avg_metrics['cl_lcq_rel_err']*100:.2f}%)")
    print(f" Cd LCQ MAE: {avg_metrics['cd_lcq_mae']:.6f} (Rel: {avg_metrics['cd_lcq_rel_err']*100:.2f}%)")
    print(f" Cl INT MAE: {avg_metrics['cl_int_mae']:.6f} (Rel: {avg_metrics['cl_int_rel_err']*100:.2f}%)")
    print(f" Cd INT MAE: {avg_metrics['cd_int_mae']:.6f} (Rel: {avg_metrics['cd_int_rel_err']*100:.2f}%)")
    print("-" * 50)
    print(f" Mean Time : {avg_metrics['mean_time_ms']:.2f} ms per sample")
    print("="*50)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        # Save both summary and individual samples for analysis
        output_payload = {
            "summary": avg_metrics,
            "samples": results
        }
        with open(out_path, 'w') as f:
            json.dump(output_payload, f, indent=4)
        print(f"Saved results to {out_path}")

    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description="DD-RNO Full Mesh Evaluation Script")
    parser.add_argument("--ckpt", default="checkpoints/ddrno/last.pt", help="Path to checkpoint")
    parser.add_argument("--config", default="configs/ddrno.yaml", help="Path to yaml config")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split to evaluate")
    parser.add_argument("--chunk_size", type=int, default=16384, help="Chunk size for decoder query batching")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--out", default="evaluation/results_full_mesh.json", help="Path to write JSON output")
    args = parser.parse_args()

    evaluate_full_mesh(
        ckpt_path=args.ckpt,
        config_path=args.config,
        split=args.split,
        chunk_size=args.chunk_size,
        device=args.device,
        out_path=args.out
    )

if __name__ == "__main__":
    main()
