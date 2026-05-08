import os
import sys
import numpy as np

# Try to find and import lumapi
# Default path for Lumerical 2024
LUMERICAL_PATH = r"D:\Program Files\Lumerical\v241\api\python"
if LUMERICAL_PATH not in sys.path:
    sys.path.append(LUMERICAL_PATH)

try:
    import lumapi
except ImportError:
    print("Error: lumapi not found. Please check Lumerical installation path.")
    lumapi = None

class SimulationEngine:
    """
    Handles Lumerical STACK simulation execution.
    """
    def __init__(self, material_manager):
        self.mm = material_manager
        self.fdtd = None

    def start_session(self):
        if lumapi:
            self.fdtd = lumapi.FDTD(hide=True)
            self.mm.add_custom_materials(self.fdtd)
        else:
            raise RuntimeError("lumapi not available.")

    def run_stack(self, stack_config):
        """
        Runs stackrt for a given stack configuration.
        """
        if not self.fdtd:
            self.start_session()

        layers = stack_config["layers"]
        material_names = [l[0] for l in layers]
        thicknesses = [l[1] for l in layers]
        
        freqs = self.mm.freqs
        num_freqs = len(freqs)
        num_layers = len(layers)
        
        # Build n matrix (num_layers x num_freqs)
        n_matrix = np.zeros((num_layers, num_freqs), dtype=complex)
        
        for i, mat_name in enumerate(material_names):
            # Check if it's a custom material we added
            if "custom" in mat_name:
                n, k = getattr(self.mm, f"get_{mat_name.split('_')[0].lower()}")()
                n_matrix[i, :] = n + 1j*k
            else:
                # Get from Lumerical database
                try:
                    # Check if material exists
                    if not self.fdtd.materialexists(mat_name):
                        print(f"Warning: Material {mat_name} not in DB. Searching for alternatives...")
                        all_mats = self.fdtd.getmaterial().split("\n")
                        found = False
                        for m in all_mats:
                            if mat_name.split(" ")[0] in m:
                                print(f"Using alternative: {m}")
                                mat_name = m
                                found = True
                                break
                        if not found:
                            raise ValueError(f"Material {mat_name} and alternatives not found.")

                    n_raw = self.fdtd.getindex(mat_name, freqs)
                    if n_raw.ndim > 1:
                        n_matrix[i, :] = n_raw[:, 0]
                    else:
                        n_matrix[i, :] = n_raw
                except Exception as e:
                    print(f"Warning: Could not get index for {mat_name}: {e}. Using vacuum (n=1).")
                    n_matrix[i, :] = 1.0

        # Run stackrt
        # theta = 0 (normal incidence)
        # Note: In Lumerical stackrt, d[0] and d[-1] are ignored as they are semi-infinite
        # But we must provide them.
        res = self.fdtd.stackrt(n_matrix, np.array(thicknesses), freqs)
        
        # res is a struct containing Rp, Rs, Tp, Ts, etc.
        # For normal incidence, Rp = Rs
        return {
            "wavelengths": self.mm.wavelengths,
            "R": res["Rp"],
            "T": res["Tp"]
        }

    def close_session(self):
        if self.fdtd:
            self.fdtd.close()
            self.fdtd = None
