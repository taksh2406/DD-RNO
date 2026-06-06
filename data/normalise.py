import numpy as np

NORM_STATS_PATH = 'configs/normalisation_stats.json'

def normalise_sample(sample):
    u_inf = 1.0
    rho   = 1.0
    q_inf = 0.5 * rho * u_inf**2
    return {
        'u':   sample['u'] / u_inf,
        'v':   sample['v'] / u_inf,
        'p':   sample['p'] / q_inf,
        'sdf': sample['sdf'],          
        'xy':  sample['xy'],           
        'aoa_enc': [
            float(np.sin(np.radians(sample['aoa']))),
            float(np.cos(np.radians(sample['aoa']))),
            float(np.sin(np.radians(2 * sample['aoa']))),
        ],
        're_enc': float(np.log10(sample['re'] / 1e6)),
        'turb_model': sample.get('turb_model', 0),
        'ma':  sample.get('ma', 0.3),
    }
