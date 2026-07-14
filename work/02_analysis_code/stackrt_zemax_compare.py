import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

LUMERICAL_PATH = r"D:\Program Files\Lumerical\v241\api\python"
if LUMERICAL_PATH not in sys.path:
    sys.path.append(LUMERICAL_PATH)

try:
    import lumapi
except ImportError:
    print("Error: lumapi not found. Please check Lumerical installation path.")
    sys.exit(1)

def run_simulation():
    # Wavelength range 500nm to 700nm with 0.01nm resolution
    wl_start = 500e-9
    wl_end = 700e-9
    wl_step = 0.01e-9
    
    # Generate wavelength array
    wl = np.arange(wl_start, wl_end + wl_step/2, wl_step)
    f = 3e8 / wl
    
    # Reverse so frequency is strictly increasing (safe practice for stackrt)
    wl = wl[::-1]
    f = f[::-1]
    
    num_freqs = len(f)
    print(f"Total points: {num_freqs}")
    
    # Target Reflectances
    R1 = 0.50
    R2 = 0.06
    R3 = 0.008
    
    # Calculate required refractive indices to match target reflection
    n0 = 1.0
    n1 = n0 * (1 + np.sqrt(R1)) / (1 - np.sqrt(R1))
    n2 = n1 * (1 - np.sqrt(R2)) / (1 + np.sqrt(R2))
    n3 = n2 * (1 - np.sqrt(R3)) / (1 + np.sqrt(R3))
    
    print(f"Calculated virtual refractive indices:")
    print(f"n0 = {n0:.4f} (Air)")
    print(f"n1 = {n1:.4f} (to match R=50%)")
    print(f"n2 = {n2:.4f} (to match R=6%)")
    print(f"n3 = {n3:.4f} (to match R=0.8%)")
    
    # Thicknesses
    d_layer1 = 2e-3 # 2mm
    d_layer2 = 1e-6 # 1um
    
    # Assemble n_matrix and d
    n_matrix = np.zeros((4, num_freqs), dtype=complex)
    n_matrix[0, :] = n0
    n_matrix[1, :] = n1
    n_matrix[2, :] = n2
    n_matrix[3, :] = n3
    
    d = np.array([0, d_layer1, d_layer2, 0])
    
    print("Starting Lumerical session...")
    fdtd = lumapi.FDTD(hide=True)
    
    try:
        print("Running stackrt (Coherent calculation)...")
        res = fdtd.stackrt(n_matrix, d, f)
        
        # Flatten outputs
        R_total = res["Rp"].flatten()
        
        # Ensure results directory exists
        if not os.path.exists("results"):
            os.makedirs("results")
            
        # Save to CSV
        df = pd.DataFrame({
            "Wavelength_nm": wl * 1e9,
            "Reflection": R_total
        })
        df.sort_values("Wavelength_nm", inplace=True)
        csv_path = "results/zemax_compare_stackrt.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Plot 1: Full Range
        plt.figure(figsize=(12, 6))
        plt.plot(df["Wavelength_nm"], df["Reflection"], linewidth=0.5)
        plt.title("Stackrt Reflection (500-700nm, 0.01nm step)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflection")
        plt.grid(True)
        plt.savefig("results/zemax_compare_full.png", dpi=300)
        plt.close()
        
        # Plot 2: Zoom in (0.5nm window) to observe FSR from 2mm cavity
        plt.figure(figsize=(12, 6))
        mask = (df["Wavelength_nm"] >= 500) & (df["Wavelength_nm"] <= 500.5)
        plt.plot(df.loc[mask, "Wavelength_nm"], df.loc[mask, "Reflection"], linewidth=1.5)
        plt.title("Stackrt Reflection (Zoom 500 - 500.5 nm)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflection")
        plt.grid(True)
        plt.savefig("results/zemax_compare_zoom.png", dpi=300)
        plt.close()
        
        print("Plots generated successfully in 'results' folder.")
        
    finally:
        fdtd.close()

if __name__ == "__main__":
    run_simulation()
