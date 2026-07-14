import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


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

    # Incident angle sweep in degrees.
    "ANGLE_START_DEG": -10.0,
    "ANGLE_STOP_DEG": 10.0,
    "ANGLE_STEP_DEG": 0.05,

    # Keep the same p-polarized reflectance channel used by main_dynamic.py.
    "POLARIZATION": "p",

    "SIMPLE_MODEL": {
        "LAYERS": [
            ("RefReflector", 0),
            ("Air", 1000.0),
            (1.6488, 1),
            (1.9723, 0),
        ]
    },

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
# 3. Angle sequence simulation
# ==========================================
class AngleSimulator:
    def __init__(self, config):
        self.config = config

        span_nm = (config["WAVELENGTH_STOP"] - config["WAVELENGTH_START"]) * 1000
        num_points = int(round(span_nm / config["SPECTRAL_RESOLUTION_NM"])) + 1
        self.wavelengths = np.linspace(config["WAVELENGTH_START"], config["WAVELENGTH_STOP"], num_points)
        self.freqs = 3e8 / (self.wavelengths * 1e-6)

        start = config["ANGLE_START_DEG"]
        stop = config["ANGLE_STOP_DEG"]
        step = config["ANGLE_STEP_DEG"]
        self.angle_axis = np.round(np.arange(start, stop + step / 2, step), 10)
        self.Ntheta = len(self.angle_axis)

        print(f"[Config] Wavelength points: N_lambda = {num_points}")
        print(f"[Config] Angle sweep: {start:g} deg to {stop:g} deg, step = {step:g} deg")
        print(f"[Config] Angle points: N_angle = {self.Ntheta}")

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

    def run_angle_sequence(self):
        if not lumapi:
            raise RuntimeError("lumapi is not available.")

        model_key = "SIMPLE_MODEL" if self.config["MODEL_TYPE"] == "simple" else "PSS_TIO2_MODEL"
        n_matrix, thicknesses, air_idx = self._get_n_matrix(model_key)

        if air_idx == -1:
            raise ValueError("Model does not contain an 'Air' layer.")

        spectra = np.zeros((self.Ntheta, len(self.wavelengths)))
        result_key = "Rp" if self.config["POLARIZATION"].lower() == "p" else "Rs"

        print("Starting Lumerical FDTD API session...")
        fdtd = lumapi.FDTD(hide=True)

        start_time = time.time()
        print(f"Running {self.Ntheta} StackRT angle simulations (matrix shape: {spectra.shape})...")

        try:
            for i, theta_deg in enumerate(self.angle_axis):
                res = fdtd.stackrt(n_matrix, thicknesses, self.freqs, float(theta_deg))
                spectra[i, :] = np.real(np.asarray(res[result_key]).flatten())

                if (i + 1) % max(1, self.Ntheta // 4) == 0:
                    elapsed = time.time() - start_time
                    pct = (i + 1) / self.Ntheta * 100
                    print(f"   [Progress] {i + 1}/{self.Ntheta} ({pct:.0f}%) - elapsed {elapsed:.1f}s")
        finally:
            fdtd.close()

        print(f"Angle sequence simulation completed. Total time: {time.time() - start_time:.2f}s")

        cavity_length_um = thicknesses[air_idx] * 1e6
        return {
            # Compatibility fields: keep the same names and row-major spectra layout as main_dynamic.py.
            "t_axis": self.angle_axis,
            "wavelengths": self.wavelengths,
            "L_t": np.full(self.Ntheta, cavity_length_um),
            "spectra": spectra,
            # Explicit angle fields for new angle-sweep analysis code.
            "angle_axis": self.angle_axis,
            "theta_axis": self.angle_axis,
        }


# ==========================================
# 4. Save and quick-look plot
# ==========================================
class AngleAnalyzer:
    @staticmethod
    def save_and_plot(data, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        npz_path = os.path.join(save_dir, "angle_dynamic_time.npz")
        np.savez_compressed(
            npz_path,
            t_axis=data["t_axis"],
            wavelengths=data["wavelengths"],
            L_t=data["L_t"],
            spectra=data["spectra"],
            angle_axis=data["angle_axis"],
            theta_axis=data["theta_axis"],
        )
        print(f"Saved compressed StackRT data: {npz_path}")

        angles = data["angle_axis"]
        wavelengths_nm = data["wavelengths"] * 1000
        spectra = data["spectra"]

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        im = axs[0].pcolormesh(wavelengths_nm, angles, spectra, shading="auto", cmap="viridis")
        axs[0].set_title("Reflectance vs Wavelength and Angle")
        axs[0].set_xlabel("Wavelength (nm)")
        axs[0].set_ylabel("Incident angle (deg)")
        fig.colorbar(im, ax=axs[0], label="Reflectance")

        center_idx = int(np.argmin(np.abs(angles)))
        min_idx = 0
        max_idx = len(angles) - 1
        axs[1].plot(wavelengths_nm, spectra[min_idx, :], label=f"{angles[min_idx]:.2f} deg")
        axs[1].plot(wavelengths_nm, spectra[center_idx, :], label=f"{angles[center_idx]:.2f} deg")
        axs[1].plot(wavelengths_nm, spectra[max_idx, :], label=f"{angles[max_idx]:.2f} deg")
        axs[1].set_title("Spectra at Selected Angles")
        axs[1].set_xlabel("Wavelength (nm)")
        axs[1].set_ylabel("Reflectance")
        axs[1].grid(True)
        axs[1].legend()

        png_path = os.path.join(save_dir, "angle_dynamic_time_dashboard.png")
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"Saved quick-look plot: {png_path}")


# ==========================================
# 5. Main
# ==========================================
def main():
    print("=== Lumerical StackRT Angle Sequence Simulation ===")

    sim = AngleSimulator(CONFIG)

    try:
        data = sim.run_angle_sequence()
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return

    output_dir = os.path.join(os.path.dirname(__file__), "stackrt_result")
    AngleAnalyzer.save_and_plot(data, output_dir)
    print("All tasks completed.")


if __name__ == "__main__":
    main()
