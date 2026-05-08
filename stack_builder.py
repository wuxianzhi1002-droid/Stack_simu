class StackBuilder:
    """
    Builds the layer stack configurations for Lumerical STACK simulation.
    """
    def __init__(self, material_manager):
        self.mm = material_manager

    def get_stacks(self):
        """
        Returns a list of stack configurations.
        Each stack is a list of tuples: (material_name, thickness_nm)
        The first layer is the top layer (Air is implicit in STACK if not added).
        The last layer is the semi-infinite substrate.
        """
        
        # Base layers shared by all
        hsq_thick = 30e-9
        soc_thick = 45e-9
        hm_thick = 20e-9
        cond_thick = 5e-9
        cu_thick = 100e-9 # Assumed thick Cu layer
        
        stacks = []

        # Case 1: PSS + TiO2
        stacks.append({
            "name": "PSS_TiO2",
            "layers": [
                ("HSQ_custom", hsq_thick),
                ("PSS_custom", cond_thick),
                ("SOC_custom", soc_thick),
                ("TiO2_custom", hm_thick),
                (self.mm.get_cu(), cu_thick),
                (self.mm.get_sin(), 0) # 0 means semi-infinite in some conventions, or we handle it in simulation.py
            ]
        })

        # Case 2: Cr + TiO2
        stacks.append({
            "name": "Cr_TiO2",
            "layers": [
                ("HSQ_custom", hsq_thick),
                (self.mm.get_cr(None), cond_thick),
                ("SOC_custom", soc_thick),
                ("TiO2_custom", hm_thick),
                (self.mm.get_cu(), cu_thick),
                (self.mm.get_sin(), 0)
            ]
        })

        # Case 3: PSS + HfO2
        stacks.append({
            "name": "PSS_HfO2",
            "layers": [
                ("HSQ_custom", hsq_thick),
                ("PSS_custom", cond_thick),
                ("SOC_custom", soc_thick),
                ("HfO2_custom", hm_thick),
                (self.mm.get_cu(), cu_thick),
                (self.mm.get_sin(), 0)
            ]
        })

        # Case 4: Cr + HfO2
        stacks.append({
            "name": "Cr_HfO2",
            "layers": [
                ("HSQ_custom", hsq_thick),
                (self.mm.get_cr(None), cond_thick),
                ("SOC_custom", soc_thick),
                ("HfO2_custom", hm_thick),
                (self.mm.get_cu(), cu_thick),
                (self.mm.get_sin(), 0)
            ]
        })

        return stacks
