import numpy as np

class MaterialManager:
    """
    Manages material properties (n, k) for the STACK simulation.
    Handles Lumerical database materials and custom synthetic models.
    """
    def __init__(self):
        # Wavelength range in nm (400-800nm)
        self.wavelengths = np.linspace(400, 800, 401)
        self.freqs = 3e8 / (self.wavelengths * 1e-9)

    def get_hsq(self):
        """
        HSQ (Hydrogen Silsesquioxane).
        Updated Cauchy model: n = 1.39 + 0.003 / (lambda^2 in um)
        Reference: Dow Corning XR-1541 Datasheet / Literature.
        """
        w_um = self.wavelengths / 1000.0
        n = 1.39 + 0.003 / (w_um**2)
        k = np.zeros_like(self.wavelengths)
        return n, k

    def get_soc(self):
        """
        SOC (Spin-On Carbon).
        Approximate Cauchy model: n = 1.55 + 0.005 / (lambda^2 in um)
        """
        w_um = self.wavelengths / 1000.0
        n = 1.55 + 0.005 / (w_um**2)
        k = 0.01 * np.ones_like(self.wavelengths) # Small absorption
        return n, k

    def get_pss(self):
        """
        PSS conductive layer.
        Updated dispersive model: n = 1.48 + 0.004 / (lambda^2 in um), k ~ 0.05.
        Reference: PEDOT:PSS optical properties in visible range.
        """
        w_um = self.wavelengths / 1000.0
        n = 1.48 + 0.004 / (w_um**2)
        k = 0.05 * np.ones_like(self.wavelengths)
        return n, k

    def get_cr(self, fdtd):
        """
        Cr (Chromium) - Thin conductive layer.
        Try to get from Lumerical database.
        """
        # Note: In STACK simulation via lumapi, we often use the material name 
        # as defined in the Lumerical session. 
        return "Cr (Chromium) - CRC"

    def get_tio2(self):
        """
        TiO2 Hardmask.
        Typical n ~ 2.4 - 2.6 (dispersive).
        """
        w_um = self.wavelengths / 1000.0
        n = 2.4 + 0.02 / (w_um**2)
        k = np.zeros_like(self.wavelengths)
        return n, k

    def get_hfo2(self):
        """
        HfO2 Hardmask.
        Typical n ~ 2.0 - 2.1 (dispersive).
        """
        w_um = self.wavelengths / 1000.0
        n = 2.0 + 0.015 / (w_um**2)
        k = np.zeros_like(self.wavelengths)
        return n, k

    def get_cu(self):
        """
        Cu (Copper).
        """
        return "Cu (Copper) - CRC"

    def get_sin(self):
        """
        SiN (Silicon Nitride).
        """
        return "Si3N4 (Silicon Nitride) - Luke"

    def add_custom_materials(self, fdtd):
        """
        Adds custom materials to the Lumerical session.
        """
        custom_mats = {
            "HSQ_custom": self.get_hsq(),
            "SOC_custom": self.get_soc(),
            "PSS_custom": self.get_pss(),
            "TiO2_custom": self.get_tio2(),
            "HfO2_custom": self.get_hfo2()
        }

        for name, (n, k) in custom_mats.items():
            eps = (n + 1j*k)**2
            try:
                # Check if material already exists
                if not fdtd.materialexists(name):
                    new_mat = fdtd.addmaterial("Sampled data")
                    fdtd.setmaterial(new_mat, "name", name)
                
                # Convert wavelengths to frequency
                # stackrt expects freq in descending order if lambda is ascending, 
                # but sampled data needs to be ascending freq.
                # My freq is descending (since lambda is 400->800).
                # Sort them.
                sort_idx = np.argsort(self.freqs)
                data = np.vstack((self.freqs[sort_idx], eps[sort_idx])).T
                fdtd.setmaterial(name, "sampled data", data)
                # Ensure color doesn't conflict
                fdtd.setmaterial(name, "color", np.random.rand(4))
            except Exception as e:
                print(f"Warning: Could not add material {name}: {e}")
