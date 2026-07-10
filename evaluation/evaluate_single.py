import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.airfrans_dataset import AirfRANSDataset, physics_collate
from evaluation.forces import integrate_pressure_numpy
from model.dd_rno import DDRNO


def flatten_config(config):
    if "data" not in config:
        return dict(config)
    flat = {}
    for section in ("data", "model", "training", "evaluation"):
        flat.update(config.get(section, {}))
    return flat


def build_model(config, device):
    model = DDRNO(
        geom_dim=config.get("geom_dim", 128),
        w_dim=config.get("w_dim", 64),
        hidden=config.get("hidden", 256),
        n_fourier=config.get("n_fourier", 8),
        n_canonical=config.get("n_canonical", 1024),
        inv_layers=config.get("inv_layers", 4),
        bl_layers=config.get("bl_layers", 6),
        wake_layers=config.get("wake_layers", 4),
        use_domain_routing=config.get("use_domain_routing", True),
        use_lcq=config.get("use_lcq", True),
        use_specinr=config.get("use_specinr", True),
    ).to(device)
    return model


def field_metrics(pred_std, target_std, field_mean, field_std):
    mean = torch.as_tensor(field_mean, device=pred_std.device).view(1, 1, 4)
    std = torch.as_tensor(field_std, device=pred_std.device).view(1, 1, 4)
    pred = pred_std * std + mean
    target = target_std * std + mean
    err = (pred - target) ** 2
    mse = err.mean(dim=(0, 1)).detach().cpu().numpy()
    rel = []
    for i in range(3):
        rel.append((torch.norm(pred[..., i] - target[..., i]) / (torch.norm(target[..., i]) + 1e-8)).item())
    return {
        "u_mse": float(mse[0]),
        "v_mse": float(mse[1]),
        "p_mse": float(mse[2]),
        "nu_t_mse": float(mse[3]),
        "u_rel_l2": float(rel[0]),
        "v_rel_l2": float(rel[1]),
        "p_rel_l2": float(rel[2]),
    }


def evaluate(ckpt_path, config_path, split="test", out_path="results/publication_full/test_metrics.json"):
    with open(config_path, "r") as f:
        config = flatten_config(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = AirfRANSDataset(config["train_h5"], n_query=config.get("query_points_per_sample", 2048))
    stats = {
        "cl_cd_mean": train_ds.cl_cd_mean,
        "cl_cd_std": train_ds.cl_cd_std,
        "field_mean": train_ds.field_mean,
        "field_std": train_ds.field_std,
    }
    eval_h5 = config["test_h5"] if split == "test" else config["train_h5"]
    ds = AirfRANSDataset(eval_h5, n_query=config.get("query_points_per_sample", 2048), external_stats=stats)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=physics_collate)

    model = build_model(config, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    rows = []
    metric_sums = {}
    field_mean = ds.field_mean
    field_std = ds.field_std

    with torch.no_grad():
        for i, batch in enumerate(loader):
            sdf_grid = batch["sdf_grid"].to(device)
            aoa_enc = batch["aoa_enc"].to(device)
            re_enc = batch["re_enc"].to(device)
            q_pts = batch["query_pts"].to(device)
            q_phi = batch["query_phi"].to(device)
            q_norm = batch["query_normal"].to(device)
            target = batch["target"].to(device)
            re_phys = batch["re_phys"].to(device)

            pred, _ = model(
                sdf_grid=sdf_grid,
                aoa_enc=aoa_enc,
                re_enc=re_enc,
                query_pts=q_pts,
                query_sdf=q_phi.squeeze(-1),
                re_phys=re_phys,
                query_normal=q_norm
            )

            m = field_metrics(pred, target, field_mean, field_std)
            for k, v in m.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v

            sample = ds.data_cache[i]
            airfoil_xy = torch.from_numpy(sample["airfoil_xy_native"][None]).float().to(device)
            airfoil_n = torch.from_numpy(sample["airfoil_normal_native"][None]).float().to(device)
            airfoil_phi = torch.zeros((1, airfoil_xy.shape[1]), device=device)
            
            surf_pred, _ = model(
                sdf_grid=sdf_grid,
                aoa_enc=aoa_enc,
                re_enc=re_enc,
                query_pts=airfoil_xy,
                query_sdf=airfoil_phi,
                re_phys=re_phys,
                query_normal=airfoil_n
            )
            p_cp = (surf_pred[..., 2] * field_std[2] + field_mean[2]).squeeze(0).detach().cpu().numpy()

            cl_p_pred, cd_p_pred = integrate_pressure_numpy(
                p_cp,
                sample["airfoil_line_native"],
                sample["airfoil_cell_normal_native"],
                sample["airfoil_length_native"],
                sample["aoa_rad"],
            )
            labels = sample["labels"]
            row = {
                "idx": i,
                "name": sample["name"],
                "Cl_true": labels["Cl"],
                "Cd_true": labels["Cd"],
                "Cl_p_true": labels["Cl_p"],
                "Cd_p_true": labels["Cd_p"],
                "Cd_v_true": labels["Cd_v"],
                "Cl_p_pred": cl_p_pred,
                "Cd_p_pred": cd_p_pred,
                "Cl_p_rel_err": abs(cl_p_pred - labels["Cl_p"]) / max(abs(labels["Cl_p"]), 1e-8),
                "Cd_p_rel_err": abs(cd_p_pred - labels["Cd_p"]) / max(abs(labels["Cd_p"]), 1e-8),
            }
            rows.append(row)

            if i % 25 == 0:
                print(
                    f"{i:04d} p_MSE={m['p_mse']:.5g} "
                    f"Clp_err={100 * row['Cl_p_rel_err']:.2f}% "
                    f"Cdp_err={100 * row['Cd_p_rel_err']:.2f}%"
                )

    n = max(len(rows), 1)
    summary = {k: v / n for k, v in metric_sums.items()}
    summary["Cl_p_rel_err_mean"] = float(np.mean([r["Cl_p_rel_err"] for r in rows]))
    summary["Cd_p_rel_err_mean"] = float(np.mean([r["Cd_p_rel_err"] for r in rows]))
    summary["Cl_p_spearman"] = float(spearmanr([r["Cl_p_true"] for r in rows], [r["Cl_p_pred"] for r in rows]).correlation)
    summary["Cd_p_spearman"] = float(spearmanr([r["Cd_p_true"] for r in rows], [r["Cd_p_pred"] for r in rows]).correlation)
    summary["n_samples"] = len(rows)
    summary["split"] = split
    summary["checkpoint"] = ckpt_path

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    csv_path = os.path.splitext(out_path)[0] + "_samples.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "sample_csv": csv_path}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/ddrno.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="results/publication_full/test_metrics.json")
    args = parser.parse_args()
    evaluate(args.ckpt, args.config, split=args.split, out_path=args.out)


if __name__ == "__main__":
    main()
