import os
import numpy as np

class ZemaxMaterialManager:
    """
    Translates material models to Zemax Catalog (.AGF) and Coating (.DAT) files.
    """
    def __init__(self, catalog_name="STACK_MATERIALS"):
        self.catalog_name = catalog_name
        self.materials = {
            "HSQ_custom": {"A": 1.39, "B": 0.003},
            "SOC_custom": {"A": 1.55, "B": 0.005},
            "PSS_custom": {"A": 1.48, "B": 0.004, "k": 0.05},
            "TiO2_custom": {"A": 2.4, "B": 0.02},
            "HfO2_custom": {"A": 2.0, "B": 0.015},
            "Reflector_custom": {"n": 5.8284} # 50% R at air interface
        }

    def generate_agf(self, output_path):
        """
        Generates a Zemax AGF catalog file.
        Zemax Cauchy formula: n = A + B/L^2 + C/L^4 ... (L in um)
        """
        lines = [f"CC {self.catalog_name}"]
        for name, props in self.materials.items():
            # Use the key as the material name in AGF (e.g. HSQ)
            clean_name = name.split("_")[0].upper()
            if "A" in props:
                a, b = props["A"], props["B"]
                lines.append(f"NM {clean_name} 2 0 0 0 0 0 0 0")
                lines.append(f"CD {a} {b} 0 0 0 0 0 0")
            else:
                n = props.get("n", 1.0)
                lines.append(f"NM {clean_name} 1 0 0 0 0 0 0 0")
                lines.append(f"CD {n} 0 0 0 0 0 0 0")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Generated {output_path}")

    def generate_coating_dat(self, output_path, stacks):
        """
        Generates a Zemax Coating.DAT file.
        """
        lines = ["! STACK Simulation Coatings", ""]
        
        # Define 50% Reflector
        lines.append("COAT REFLECT_50")
        lines.append("IDEAL 0.5 0.5")
        lines.append("")

        for stack in stacks:
            name = stack["name"]
            layers = stack["layers"]
            
            lines.append(f"COAT {name}")
            
            stack_layers = []
            for mat, thick in layers:
                if mat in ["Reflector_custom", "etch"]:
                    continue
                stack_layers.append((mat, thick))
            
            for mat, thick in stack_layers:
                thick_um = thick * 1e6
                # Material name should match AGF clean_name
                if "_custom" in mat:
                    clean_mat = mat.split("_")[0].upper()
                elif "Chromium" in mat:
                    clean_mat = "CR"
                elif "Copper" in mat:
                    clean_mat = "CU"
                elif "Silicon Nitride" in mat:
                    clean_mat = "SI3N4"
                else:
                    clean_mat = mat.split(" ")[0].upper()
                
                lines.append(f"MATE {clean_mat} {thick_um:.6f}")
            
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Generated {output_path}")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    # 将父目录添加到路径，以便导入 stack_builder 和 materials
    sys.path.append(str(Path(__file__).parent.parent))
    
    from stack_builder import StackBuilder
    from materials import MaterialManager
    
    mm = MaterialManager()
    sb = StackBuilder(mm)
    stacks = sb.get_stacks()
    
    zmm = ZemaxMaterialManager()
    zmm.generate_agf("stack_in_zemax/STACK_MATERIALS.AGF")
    zmm.generate_coating_dat("stack_in_zemax/COATING.DAT", stacks)
