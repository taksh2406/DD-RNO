<div align="center">

# DD-RNO: Domain-Decomposed Routed Neural Operator for Airfoil Flow Prediction

<img src="assets/DD-RNO.png" alt="Neural Surrogate for CFD Simulations" width="560">

Official implementation of **DD-RNO**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.3+](https://img.shields.io/badge/pytorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Official open-source repository for **DD-RNO (Domain-Decomposed Routed Neural Operator)**, a physics-informed deep learning surrogate for high-fidelity airfoil flow field prediction ($u_x, u_y, p, \nu_t$) and direct surface force integration ($C_L, C_D$).

[**Key Features**](#key-features) • [**Architecture**](#architecture) • [**Getting Started**](#getting-started) • [**License**](#license) • [**Citation**](#citation)

---

</div>

## Key Features

- **Mesh-Free .DAT Inference**: Predict complete flow fields and aerodynamic forces directly from raw `.dat` coordinate files in **< 150 ms** without requiring any CFD meshing or preprocessing.
- **Spectral Geometry Trunk (SpecINR)**: Continuous 2D FNO trunk that extracts global shape context from Signed Distance Fields ($64 \times 64$) and enables continuous spatial probing.
- **Domain-Decomposed Specialist Decoders**: Three specialized decoders (Inviscid, Boundary Layer, and Wake) dynamically routed via Reynolds-adaptive wall distance gates ($\delta_{\text{BL}} \propto \text{Re}^{-1/5}$).
- **Learned Canonical Quadrature (LCQ)**: Differentiable surface integration pipeline that maps continuous surface pressure fields $C_p$ and global flow latents $\mathbf{w}$ directly to lift ($C_L$) and drag ($C_D$) coefficients.
- **Feature-wise Linear Modulation (FiLM)**: Deep conditioning injection across all decoder layers driven by a StyleGAN-inspired mapping network.

---

## Architecture

The end-to-end DD-RNO pipeline maps Signed Distance Fields (SDF) and flow conditions $(\alpha, \text{Re})$ to continuous spatial flow fields and aerodynamic force coefficients ($C_L, C_D$):

<p align="center">
  <img src="./assets/master_architecture.png" alt="DD-RNO Master Architecture" width="95%"/>
</p>

---

## Getting Started

### 1. Installation
Clone the repository and install dependencies in a Python 3.10+ environment:
```bash
git clone https://github.com/taksh2406/DD-RNO.git
cd DD-RNO
pip install -r requirements.txt
```

### 2. Standalone Inference Script
Run direct predictions from a raw `.dat` file:
```bash
python example_inference.py
```

### 3. Python API Usage
```python
from inference.predict import DDRNOPredictor

# Initialize predictor with trained checkpoint and config
predictor = DDRNOPredictor("checkpoints/ddrno/best_cl.pt", "configs/ddrno.yaml")

# Predict Cl and Cd directly from a .dat file (NACA 0012 at alpha = 5 deg, Re = 3M)
result = predictor.predict_from_dat("naca0012.dat", aoa_deg=5.0, re=3e6)
print(f"Cl = {result['Cl']:.4f}, Cd = {result['Cd']:.5f}")
# Output: Cl = 0.4933, Cd = 0.01030

# Optionally query continuous flow fields (ux, uy, p, nut) at arbitrary coordinates (x, y)
import numpy as np
query_points = np.array([[0.5, 0.1], [0.5, 0.0], [1.2, 0.0]])
res = predictor.predict_from_dat("naca0012.dat", aoa_deg=5.0, re=3e6, query_pts=query_points, return_field=True)
print("Flow fields shape:", res['field'].shape) # (3, 4) -> [ux, uy, p, nut]
```

### 4. Training & Evaluation
To train DD-RNO on the Full dataset split:
```bash
python training/main_train.py --config configs/ddrno.yaml --seed 42
```

To evaluate a trained checkpoint on the test split:
```bash
python evaluation/eval_full_mesh.py --ckpt checkpoints/ddrno/best_cl.pt --config configs/ddrno.yaml --split test
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you find DD-RNO useful in your research, please cite the authors: Taksh Mehta, Piyush Singh Bhati and Harshal Akolekar.
