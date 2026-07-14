import clr, os
import numpy as np

def debug_nsc_face():
    bin_path = r"D:\Program Files\Ansys Zemax OpticStudio 2024 R1.00"
    net_helper_path = os.path.join(bin_path, 'ZOSAPI_NetHelper.dll')
    clr.AddReference(net_helper_path)
    import ZOSAPI_NetHelper
    ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(bin_path)
    dir_zos = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()
    clr.AddReference(os.path.join(dir_zos, "ZOSAPI.dll"))
    clr.AddReference(os.path.join(dir_zos, "ZOSAPI_Interfaces.dll"))
    import ZOSAPI
    
    app = ZOSAPI.ZOSAPI_Connection().CreateNewApplication()
    sys = app.PrimarySystem
    sys.MakeNonSequential()
    nce = sys.NCE
    obj = nce.AddObject()
    
    face_data = obj.CoatScatterData.GetFaceData(1)
    print(f"FaceData type: {type(face_data)}")
    print("Available methods/attributes in FaceData:")
    for attr in dir(face_data):
        if not attr.startswith("_"):
            print(attr)
    
    app.CloseApplication()

if __name__ == "__main__":
    try:
        debug_nsc_face()
    except Exception as e:
        print(f"Debug failed: {e}")
