import numpy as np

class MaterialManager:
    """
    Manages material properties (n, k) for the STACK simulation.
    Handles custom synthetic models for Zemax comparison.
    """
    def __init__(self):
        # Wavelength range 500nm to 700nm with 0.01nm step (20001 points)
        self.wavelengths = np.linspace(500, 700, 20001)
        self.freqs = 3e8 / (self.wavelengths * 1e-9)

    def get_air(self):
        return np.ones_like(self.wavelengths), np.zeros_like(self.wavelengths)

    def get_virt1(self):
        # Virtual material to match 50% Reflection at interface with Air
        return 5.8284 * np.ones_like(self.wavelengths), np.zeros_like(self.wavelengths)

    def get_virt2(self):
        # Virtual material to match 6% Reflection at interface with virt1
        return 3.5349 * np.ones_like(self.wavelengths), np.zeros_like(self.wavelengths)

    def get_virt3(self):
        # Virtual material to match 0.8% Reflection at interface with virt2
        return 2.9545 * np.ones_like(self.wavelengths), np.zeros_like(self.wavelengths)

    def add_custom_materials(self, fdtd):
        """
        Adds custom materials to the Lumerical session.
        """
        custom_mats = {
            "Air_custom": self.get_air(),
            "Virt1_custom": self.get_virt1(),
            "Virt2_custom": self.get_virt2(),
            "Virt3_custom": self.get_virt3()
        }

        for name, (n, k) in custom_mats.items():
            eps = (n + 1j*k)**2
            try:
                if not fdtd.materialexists(name):
                    new_mat = fdtd.addmaterial("Sampled data")
                    fdtd.setmaterial(new_mat, "name", name)
                
                sort_idx = np.argsort(self.freqs)
                data = np.vstack((self.freqs[sort_idx], eps[sort_idx])).T
                fdtd.setmaterial(name, "sampled data", data)
                fdtd.setmaterial(name, "color", np.random.rand(4))
            except Exception as e:
                print(f"Warning: Could not add material {name}: {e}")
