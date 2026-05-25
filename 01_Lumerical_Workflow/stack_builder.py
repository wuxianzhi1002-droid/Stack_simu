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
        """
        stacks = []

        stacks.append({
            "name": "Zemax_Compare_Model",
            "layers": [
                ("Air_custom", 0),          # Semi-infinite Top
                ("Virt1_custom", 2e-3),     # 2mm cavity (matches R=50% surface)
                ("Virt2_custom", 1e-6),     # 1um cavity (matches R=6% surface)
                ("Virt3_custom", 0)         # Semi-infinite Substrate (matches R=0.8% surface)
            ]
        })

        return stacks
