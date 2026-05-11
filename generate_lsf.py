import numpy as np
import os
from materials import MaterialManager
from stack_builder import StackBuilder

def generate_lsf(output_filename="run_stack_simulation.lsf"):
    mm = MaterialManager()
    sb = StackBuilder(mm)
    stacks = sb.get_stacks()
    
    lsf_content = []
    lsf_content.append("# Lumerical Script to run STACK simulations")
    lsf_content.append("clear;")
    lsf_content.append(f"f = [{','.join([str(float(val)) for val in mm.freqs])}]; # Frequency vector")
    lsf_content.append("")

    for stack in stacks:
        name = stack["name"]
        layers = stack["layers"]
        thicknesses = [l[1] for l in layers]
        
        lsf_content.append(f"# --- Configuration for: {name} ---")
        lsf_content.append(f"d_{name} = [{','.join([str(float(d)) for d in thicknesses])}];")
        
        # Build n matrix for this stack in LSF
        n_matrix_str = "n_matrix_" + name + " = matrix(" + str(len(layers)) + ", " + str(len(mm.freqs)) + ");"
        lsf_content.append(n_matrix_str)
        
        for i, (mat_name, _) in enumerate(layers):
            if "custom" in mat_name:
                # For custom materials, we'll need to manually define the index in LSF
                # mat_name is like 'HSQ_custom', we need mm.get_hsq()
                mat_type = mat_name.split('_')[0].lower()
                n, k = getattr(mm, f"get_{mat_type}")()
                n_complex = n + 1j*k
                # Convert complex array to string for LSF (real + 1i*imag)
                n_str = "[" + ",".join([f"{val.real}+{val.imag}*1i" for val in n_complex]) + "]"
                lsf_content.append(f"n_matrix_{name}({i+1}, :) = {n_str};")
            else:
                lsf_content.append(f"if (materialexists('{mat_name}')) {{")
                lsf_content.append(f"    n_matrix_{name}({i+1}, :) = getindex('{mat_name}', f);")
                lsf_content.append("} else {")
                lsf_content.append(f"    ? 'Warning: {mat_name} not found, using n=1';")
                lsf_content.append(f"    n_matrix_{name}({i+1}, :) = 1;")
                lsf_content.append("}")
        
        lsf_content.append(f"res_{name} = stackrt(n_matrix_{name}, d_{name}, f);")
        lsf_content.append(f"R_{name} = res_{name}.Rp;")
        lsf_content.append(f"plot(3e8/f*1e9, R_{name}, 'Wavelength (nm)', 'Reflection', 'Reflection of {name}');")
        lsf_content.append("")

    with open(output_filename, "w") as f:
        f.write("\n".join(lsf_content))
    
    print(f"LSF script generated: {output_filename}")

if __name__ == "__main__":
    generate_lsf()
