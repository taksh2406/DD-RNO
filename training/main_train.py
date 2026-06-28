"""
DD-RNO Training Entry Point.
"""
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.airfrans_dataset import get_loaders
from model.dd_rno import DDRNO
from training.simple_trainer import SimpleTrainer


def _flatten_config(config):
    if 'data' not in config:
        return dict(config)
    flat = {}
    for section in ('data', 'model', 'training', 'evaluation'):
        flat.update(config.get(section, {}))
    return flat


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DD-RNO Training")
    parser.add_argument('--config', type=str, default='configs/ddrno.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--no_domain_routing', action='store_true')
    parser.add_argument('--no_lcq', action='store_true')
    parser.add_argument('--no_specinr', action='store_true')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)

    with open(config_path, 'r') as f:
        config = _flatten_config(yaml.safe_load(f))

    if args.seed is not None:
        config['seed'] = args.seed
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.checkpoint_dir is not None:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.no_domain_routing:
        config['use_domain_routing'] = False
    if args.no_lcq:
        config['use_lcq'] = False
    if args.no_specinr:
        config['use_specinr'] = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(" DD-RNO Training")
    print(f" Device : {str(device).upper()}")
    print(f" Epochs : {config.get('epochs', 1000)}")
    print(f" Batch  : {config.get('batch_size', 4)}")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_loaders(
        train_h5=config['train_h5'],
        test_h5=config['test_h5'],
        val_fraction=float(config.get('val_fraction', 0.2)),
        batch_size=int(config['batch_size']),
        n_query=int(config.get('query_points_per_sample', 2048)),
        num_workers=4,
    )

    ds = train_loader.dataset
    base_ds = ds.dataset if hasattr(ds, 'dataset') else ds
    config['cl_cd_mean'] = base_ds.cl_cd_mean.tolist()
    config['cl_cd_std']  = base_ds.cl_cd_std.tolist()
    config['field_mean'] = base_ds.field_mean.tolist()
    config['field_std']  = base_ds.field_std.tolist()
    print(f"  Cl Mean: {base_ds.cl_cd_mean[0]:.3f}, Std: {base_ds.cl_cd_std[0]:.3f}")
    print(f"  Field Std: {base_ds.field_std}")

    # Smoke test
    print("\nSmoke test...")
    model = DDRNO(
        geom_dim=config.get('geom_dim', 128),
        w_dim=config.get('w_dim', 64),
        hidden=config.get('hidden', 256),
        n_fourier=config.get('n_fourier', 8),
        n_canonical=config.get('n_canonical', 256),
        inv_layers=config.get('inv_layers', 4),
        bl_layers=config.get('bl_layers', 6),
        wake_layers=config.get('wake_layers', 4),
        use_domain_routing=config.get('use_domain_routing', True),
        use_lcq=config.get('use_lcq', True),
        use_specinr=config.get('use_specinr', True),
    ).to(device)
    model.eval()

    batch = next(iter(train_loader))
    with torch.no_grad():
        pred_f, pred_c = model(
            sdf_grid=batch['sdf_grid'].to(device),
            aoa_enc=batch['aoa_enc'].to(device),
            re_enc=batch['re_enc'].to(device),
            query_pts=batch['query_pts'].to(device),
            query_sdf=batch['query_phi'].to(device).squeeze(-1),
            re_phys=batch['re_phys'].to(device),
            surf_pts=batch['surf_xy'].to(device),
            query_normal=batch['query_normal'].to(device),
            surf_normal=batch['surf_normal'].to(device),
        )
    print(f"  Field: {list(pred_f.shape)}, Forces: {list(pred_c.shape)}")
    assert pred_f.shape[-1] == 4
    assert pred_c.shape[-1] == 2
    print("  Smoke test passed!")

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Set stats
    model.set_standardization_stats(
        config['cl_cd_mean'], config['cl_cd_std'],
        field_mean=config['field_mean'], field_std=config['field_std']
    )

    seed = int(config.get('seed', 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = SimpleTrainer(model, train_loader, val_loader, config)

    if args.resume:
        trainer.load(args.resume)
        print(f"  Resumed from: {args.resume}")

    trainer.train(epochs=config.get('epochs', 1000))


if __name__ == '__main__':
    main()
