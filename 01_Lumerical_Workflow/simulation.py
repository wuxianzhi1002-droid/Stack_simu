import os
import sys
import numpy as np

LUMERICAL_PATH = r"D:\Program Files\Lumerical\v241\api\python"

if os.path.exists(LUMERICAL_PATH):
    if LUMERICAL_PATH not in sys.path:
        sys.path.append(LUMERICAL_PATH)
    os.environ['PATH'] += os.pathsep + r"D:\Program Files\Lumerical\v241\bin"

try:
    import lumapi
except ImportError as e:
    print(f"Error: lumapi not found. {e}")
    lumapi = None

class SimulationEngine:
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
        if not self.fdtd:
            self.start_session()

        layers = stack_config["layers"]
        material_names = [l[0] for l in layers]
        thicknesses = [l[1] for l in layers]
        
        freqs = self.mm.freqs
        num_freqs = len(freqs)
        num_layers = len(layers)
        
        n_matrix = np.zeros((num_layers, num_freqs), dtype=complex)
        
        for i, mat_name in enumerate(material_names):
            if "custom" in mat_name:
                n, k = getattr(self.mm, f"get_{mat_name.split('_')[0].lower()}")()
                n_matrix[i, :] = n + 1j*k
            else:
                n_matrix[i, :] = 1.0

        # Run stackrt
        # Note: In Lumerical stackrt, d[0] and d[-1] are ignored as they are semi-infinite
        res = self.fdtd.stackrt(n_matrix, np.array(thicknesses), freqs)
        
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
        
        current_z = 0.0 # Substrate top surface is at z=0
        
        for mat_name, thickness in reversed(layers):
            actual_thickness = thickness
            
            # SCALE DOWN 2MM GAP FOR FDTD VISUALIZATION (Otherwise impossible)
            if "Virt1" in mat_name and thickness > 1e-4:
                actual_thickness = 2e-6 # 2um
                
            if thickness == 0:
                actual_thickness = 2e-6 # 2 microns for semi-infinite
                z_min = current_z - actual_thickness
                z_max = current_z
            else:
                z_min = current_z
                z_max = current_z + actual_thickness
                current_z = z_max
                
            if mat_name == "Air_custom":
                z_min = current_z - actual_thickness 
                z_max = current_z + 2.0e-6
            
            self.fdtd.addrect()
            obj_name = mat_name.replace(" ", "_")
            self.fdtd.set("name", obj_name)
            self.fdtd.set("x span", 2.0e-6)
            self.fdtd.set("y span", 2.0e-6)
            self.fdtd.set("z min", z_min)
            self.fdtd.set("z max", z_max)
            self.fdtd.set("material", mat_name)
            
        fdtd_z_min = -1.0e-6
        fdtd_z_max = current_z + 1.0e-6 
        
        self.fdtd.addfdtd()
        self.fdtd.set("dimension", "3D")
        self.fdtd.set("x span", 2.0e-6)
        self.fdtd.set("y span", 2.0e-6)
        self.fdtd.set("z min", fdtd_z_min)
        self.fdtd.set("z max", fdtd_z_max)
        
        self.fdtd.set("global monitor use source limits", True)
        self.fdtd.set("global monitor frequency points", 50)
        
        self.fdtd.addplane()
        self.fdtd.set("name", "source")
        self.fdtd.set("injection axis", "z-axis")
        self.fdtd.set("direction", "backward")
        self.fdtd.set("x span", 2.0e-6)
        self.fdtd.set("y span", 2.0e-6)
        self.fdtd.set("z", current_z + 0.5e-6) 
        self.fdtd.set("wavelength start", self.mm.wavelengths[0] * 1e-9)
        self.fdtd.set("wavelength stop", self.mm.wavelengths[-1] * 1e-9)
        
        import os
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        filepath = os.path.join(export_dir, f"{name}.fsp")
        self.fdtd.save(os.path.abspath(filepath))
