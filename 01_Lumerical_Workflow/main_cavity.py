import os
import sys
import time
from datetime import datetime

import numpy as np


# ==========================================
# 1. Path and environment
# ==========================================
LUMERICAL_PATH = r"D:\Program Files\Lumerical\v241\api\python"
if os.path.exists(LUMERICAL_PATH):
    if LUMERICAL_PATH not in sys.path:
        sys.path.append(LUMERICAL_PATH)
    os.environ["PATH"] += os.pathsep + r"D:\Program Files\Lumerical\v241\bin"

try:
    import lumapi
except ImportError:
    print("Error: lumapi was not found. Please check the Lumerical installation path and Python environment.")
    lumapi = None


# ==========================================
# 2. Global configuration
# ==========================================
CONFIG = {
    "MODEL_TYPE": "PSS_TiO2",

    # Wavelength range in um.
    "WAVELENGTH_START": 0.2,
    "WAVELENGTH_STOP": 0.6,
    "SPECTRAL_RESOLUTION_NM": 0.02,

    # Cavity sweep in um. Step 0.2 um equals 200 nm.
    # The existing layer convention in this project stores layer thicknesses in um.
    "CAVITY_START_UM": 1000.0,
    "CAVITY_STOP_UM": 1200.0,
    "CAVITY_STEP_UM": 0.2,

    # Normal incidence, p-polarized reflectance channel, consistent with main_dynamic.py.
    "ANGLE_DEG": 0.0,
    "POLARIZATION": "p",

    "PSS_TIO2_MODEL": {
        "LAYERS": [
            ("RefReflector", 0),
            ("Air", 1000.0),
            ("HSQ", 0.040),
            ("PSS", 0.005),
            ("SOC", 0.050),
            ("TiO2", 0.020),
            ("Cu", 0),
        ]
    },
}


# ==========================================
# 3. Cavity-length sequence simulation
# ==========================================
class CavitySimulator:
    def __init__(self, config):
        self.config = config

        span_nm = (config["WAVELENGTH_STOP"] - config["WAVELENGTH_START"]) * 1000
        num_points = int(round(span_nm / config["SPECTRAL_RESOLUTION_NM"])) + 1
        self.wavelengths = np.linspace(config["WAVELENGTH_START"], config["WAVELENGTH_STOP"], num_points)
        self.freqs = 3e8 / (self.wavelengths * 1e-6)

        start = config["CAVITY_START_UM"]
        stop = config["CAVITY_STOP_UM"]
        step = config["CAVITY_STEP_UM"]
        self.cavity_axis_um = np.round(np.arange(start, stop + step / 2, step), 10)
        self.Ncavity = len(self.cavity_axis_um)

        print(f"[Config] Wavelength points: N_lambda = {num_points}")
        print(f"[Config] Cavity sweep: {start:g} um to {stop:g} um, step = {step:g} um")
        print(f"[Config] Cavity points: N_cavity = {self.Ncavity}")

    def _get_n_matrix(self, model_key):
        layers = self.config[model_key]["LAYERS"]
        n_matrix = np.zeros((len(layers), len(self.freqs)), dtype=complex)
        thicknesses = []

        w_um = self.wavelengths
        cu_n_k = (1.1 + 2.5j) * np.ones_like(w_um)
        air_idx = -1

        for i, (mat, thick_um) in enumerate(layers):
            thicknesses.append(thick_um * 1e-6)

            if mat == "Air":
                air_idx = i

            if isinstance(mat, (int, float, complex)):
                n_matrix[i, :] = mat
            elif mat == "RefReflector":
                n_matrix[i, :] = 5.8284
            elif mat == "Air":
                n_matrix[i, :] = 1.0
            elif mat == "HSQ":
                n_matrix[i, :] = 1.41
            elif mat == "PSS":
                n_matrix[i, :] = 1.50 + 0.05j
            elif mat == "SOC":
                n_matrix[i, :] = 1.55 + 0.005 / (w_um**2)
            elif mat == "TiO2":
                n_matrix[i, :] = 2.4 + 0.02 / (w_um**2)
            elif mat == "Cu":
                n_matrix[i, :] = cu_n_k
            else:
                n_matrix[i, :] = 1.5

        return n_matrix, np.array(thicknesses), air_idx

    def run_cavity_sequence(self):
        if not lumapi:
            raise RuntimeError("lumapi is not available.")

        model_key = self.config["MODEL_TYPE"].upper() + "_MODEL"
        n_matrix, thicknesses_base, air_idx = self._get_n_matrix(model_key)

        if air_idx == -1:
            raise ValueError("Model does not contain an 'Air' layer to scan.")

        spectra = np.zeros((self.Ncavity, len(self.wavelengths)))
        result_key = "Rp" if self.config["POLARIZATION"].lower() == "p" else "Rs"

        print("Starting Lumerical FDTD API session...")
        fdtd = lumapi.FDTD(hide=True)

        start_time = time.time()
        print(f"Running {self.Ncavity} StackRT cavity simulations (matrix shape: {spectra.shape})...")

        try:
            for i, cavity_um in enumerate(self.cavity_axis_um):
                thicknesses = thicknesses_base.copy()
                thicknesses[air_idx] = cavity_um * 1e-6

                res = fdtd.stackrt(n_matrix, thicknesses, self.freqs, float(self.config["ANGLE_DEG"]))
                spectra[i, :] = np.real(np.asarray(res[result_key]).flatten())

                if (i + 1) % max(1, self.Ncavity // 4) == 0:
                    elapsed = time.time() - start_time
                    pct = (i + 1) / self.Ncavity * 100
                    print(f"   [Progress] {i + 1}/{self.Ncavity} ({pct:.0f}%) - elapsed {elapsed:.1f}s")
        finally:
            fdtd.close()

        print(f"Cavity sequence simulation completed. Total time: {time.time() - start_time:.2f}s")

        return {
            "cavity_axis_um": self.cavity_axis_um,
            "cavity_axis_m": self.cavity_axis_um * 1e-6,
            "wavelengths": self.wavelengths,
            "spectra": spectra,
        }


# ==========================================
# 4. Save result
# ==========================================
class CavityAnalyzer:
    @staticmethod
    def save_npz(data, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        npz_path = os.path.join(save_dir, f"cavity_spectra_{timestamp}.npz")
        np.savez_compressed(
            npz_path,
            cavity_axis_um=data["cavity_axis_um"],
            cavity_axis_m=data["cavity_axis_m"],
            wavelengths=data["wavelengths"],
            spectra=data["spectra"],
        )
        print(f"Saved compressed StackRT data: {npz_path}")
        return npz_path


# ==========================================
# 5. Main
# ==========================================
def main():
    print("=== Lumerical StackRT Cavity-Length Sequence Simulation ===")

    sim = CavitySimulator(CONFIG)

    try:
        data = sim.run_cavity_sequence()
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return

    output_dir = os.path.join(os.path.dirname(__file__), "stackrt_result")
    CavityAnalyzer.save_npz(data, output_dir)
    print("All tasks completed.")


if __name__ == "__main__":
    main()
