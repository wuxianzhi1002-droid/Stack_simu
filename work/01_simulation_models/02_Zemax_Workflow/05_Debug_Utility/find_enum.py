import clr, os
import numpy as np

def find_enum():
    bin_path = r"D:\Program Files\Ansys Zemax OpticStudio 2024 R1.00"
    clr.AddReference(os.path.join(bin_path, "ZOSAPI.dll"))
    import ZOSAPI
    import System
    
    # Load all types from the ZOSAPI_Interfaces assembly
    assembly = System.Reflection.Assembly.LoadFile(os.path.join(bin_path, "ZOSAPI_Interfaces.dll"))
    types = assembly.GetTypes()
    
    print("Found types containing 'DetectorDataType':")
    for t in types:
        if "DetectorDataType" in t.Name:
            print(f"{t.FullName}")
            # List values if it's an enum
            if t.IsEnum:
                for name in System.Enum.GetNames(t):
                    value = System.Enum.Parse(t, name)
                    print(f"  {name} = {int(value)}")

if __name__ == "__main__":
    find_enum()
