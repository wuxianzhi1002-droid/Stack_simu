import clr, os
import numpy as np

def list_interfaces():
    bin_path = r"D:\Program Files\Ansys Zemax OpticStudio 2024 R1.00"
    net_helper_path = os.path.join(bin_path, 'ZOSAPI_NetHelper.dll')
    clr.AddReference(net_helper_path)
    import ZOSAPI_NetHelper
    ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(bin_path)
    dir_zos = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()
    clr.AddReference(os.path.join(dir_zos, "ZOSAPI.dll"))
    clr.AddReference(os.path.join(dir_zos, "ZOSAPI_Interfaces.dll"))
    import ZOSAPI
    
    print("Interfaces in ZOSAPI.Editors.NCE starting with IObject:")
    for attr in dir(ZOSAPI.Editors.NCE):
        if attr.startswith("IObject"):
            print(attr)

if __name__ == "__main__":
    list_interfaces()
