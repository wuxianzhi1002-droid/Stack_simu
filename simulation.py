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

    def generate_fsp(self, stack_config, export_dir="results"):
        """
        Generates an FDTD layout (.fsp) file to visualize the layer stack.
        """
        if not self.fdtd:
            self.start_session()
            
        self.fdtd.switchtolayout()
        self.fdtd.selectall()
        self.fdtd.delete()
        
        layers = stack_config["layers"]
        name = stack_config["name"]
        
        # Build from bottom to top
        current_z = 0.0 # Substrate top surface is at z=0
        
        for mat_name, thickness in reversed(layers):
            actual_thickness = thickness
            if thickness == 0:
                actual_thickness = 2e-6 # 2 microns for semi-infinite substrate
                z_min = current_z - actual_thickness
                z_max = current_z
            else:
                z_min = current_z
                z_max = current_z + actual_thickness
                current_z = z_max
                
            self.fdtd.addrect()
            # Clean up material name for object naming
            obj_name = mat_name.replace(" ", "_").replace("(", "").replace(")", "")
            self.fdtd.set("name", obj_name)
            self.fdtd.set("x span", 5e-6)
            self.fdtd.set("y span", 5e-6)
            self.fdtd.set("z min", z_min)
            self.fdtd.set("z max", z_max)
            
            use_mat = mat_name
            if "custom" not in mat_name:
                try:
                    if not self.fdtd.materialexists(mat_name):
                        all_mats = self.fdtd.getmaterial().split("\n")
                        for m in all_mats:
                            if mat_name.split(" ")[0] in m:
                                use_mat = m
                                break
                except Exception:
                    pass
            self.fdtd.set("material", use_mat)
            
        # Add an FDTD region to encapsulate the interesting part of the stack
        fdtd_z_min = -1.0e-6
        fdtd_z_max = current_z + 1.0e-6
        self.fdtd.addfdtd()
        self.fdtd.set("dimension", "2D") # 2D is enough for 1D stack
        self.fdtd.set("x span", 2e-6)
        self.fdtd.set("z min", fdtd_z_min)
        self.fdtd.set("z max", fdtd_z_max)
        
        # Add Plane Wave Source
        self.fdtd.addplane()
        self.fdtd.set("name", "source")
        self.fdtd.set("injection axis", "z-axis")
        self.fdtd.set("direction", "backward") # Injection from top down
        self.fdtd.set("x span", 2e-6)
        self.fdtd.set("z", current_z + 0.5e-6)
        self.fdtd.set("wavelength start", self.mm.wavelengths[0] * 1e-9)
        self.fdtd.set("wavelength stop", self.mm.wavelengths[-1] * 1e-9)
        
        # Add Monitors
        # Reflection (above source)
        self.fdtd.addpower()
        self.fdtd.set("name", "R_monitor")
        self.fdtd.set("monitor type", "Linear X")
        self.fdtd.set("x span", 2e-6)
        self.fdtd.set("z", current_z + 0.8e-6)
        
        # Transmission (in substrate)
        self.fdtd.addpower()
        self.fdtd.set("name", "T_monitor")
        self.fdtd.set("monitor type", "Linear X")
        self.fdtd.set("x span", 2e-6)
        self.fdtd.set("z", -0.5e-6)

        import os
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        filepath = os.path.join(export_dir, f"{name}.fsp")
        self.fdtd.save(os.path.abspath(filepath))

