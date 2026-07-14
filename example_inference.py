"""
DD-RNO Inference Example
------------------------
Simple script demonstrating how to predict aerodynamic forces (Cl, Cd)
and flow fields from a raw airfoil .dat coordinate file.
"""
from inference.predict import DDRNOPredictor

def main():
    # 1. Initialize predictor with checkpoint and config
    checkpoint_path = "checkpoints/ddrno/best_cl.pt"
    config_path = "configs/ddrno.yaml"
    predictor = DDRNOPredictor(checkpoint_path, config_path)

    # 2. Predict Cl and Cd for NACA 0012 at alpha = 5 deg, Re = 3M
    result = predictor.predict_from_dat("naca0012.dat", aoa_deg=5.0, re=3e6)
    
    print("=" * 45)
    print("DD-RNO Aerodynamic Prediction Result")
    print("=" * 45)
    print(f"Airfoil: NACA 0012")
    print(f"Angle of Attack (alpha): 5.0 deg")
    print(f"Reynolds Number (Re):    3.0e6")
    print(f"Lift Coefficient (Cl):  {result['Cl']:.4f}")
    print(f"Drag Coefficient (Cd):  {result['Cd']:.5f}")
    print("=" * 45)

if __name__ == "__main__":
    main()
