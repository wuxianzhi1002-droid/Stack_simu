import clr, os, winreg
import numpy as np

# =============================================================================
# Zemax 官方样板类封装
# =============================================================================
class ZemaxStandalone:
    def __init__(self, bin_path):
        net_helper_path = os.path.join(bin_path, 'ZOSAPI_NetHelper.dll')
        clr.AddReference(net_helper_path)
        import ZOSAPI_NetHelper
        ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(bin_path)
        dir_zos = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()
        clr.AddReference(os.path.join(dir_zos, "ZOSAPI.dll"))
        clr.AddReference(os.path.join(dir_zos, "ZOSAPI_Interfaces.dll"))
        import ZOSAPI
        self.ZOSAPI = ZOSAPI
        self.TheApplication = ZOSAPI.ZOSAPI_Connection().CreateNewApplication()
        self.TheSystem = self.TheApplication.PrimarySystem

    def __del__(self):
        if hasattr(self, 'TheApplication') and self.TheApplication:
            self.TheApplication.CloseApplication()

def build_only():
    bin_path = r"D:\Program Files\Ansys Zemax OpticStudio 2024 R1.00"
    print("正在构建 Zemax 系统模型 (SI3N4 衬底 + 1e-5 阈值)...")
    zos = ZemaxStandalone(bin_path)
    ZOSAPI = zos.ZOSAPI
    TheSystem = zos.TheSystem
    TheSystem.New(False)
    NCE = ZOSAPI.Editors.NCE
    TheSystem.MakeNonSequential()
    TheNCE = TheSystem.NCE

    # 1. 设置单位 (Millimeters)
    try: TheSystem.SystemData.Units.LensUnits = type(TheSystem.SystemData.Units.LensUnits)(1)
    except: pass

    # 2. 设置追迹阈值
    try: TheSystem.SystemData.NonSequentialData.MinimumRelativeRayIntensity = 1e-5
    except: pass

    # 辅助函数
    def add_nsc_object(editor, obj_type):
        new_obj = editor.AddObject()
        settings = new_obj.GetObjectTypeSettings(obj_type)
        new_obj.ChangeType(settings)
        return new_obj

    # 1. Source Ray
    src = add_nsc_object(TheNCE, NCE.ObjectType.SourceRay)
    src.ZPosition = 0.0
    src.GetCellAt(11).IntegerValue = 100
    src.GetCellAt(12).IntegerValue = 1000
    src.GetCellAt(13).DoubleValue = 1.0

    # 2. Reference Reflector
    ref = add_nsc_object(TheNCE, NCE.ObjectType.RectangularVolume)
    ref.ZPosition = 10.0
    ref.GetCellAt(11).DoubleValue = 2.0
    ref.GetCellAt(12).DoubleValue = 2.0
    ref.GetCellAt(13).DoubleValue = 0.01
    ref.GetCellAt(14).DoubleValue = 2.0
    ref.GetCellAt(15).DoubleValue = 2.0
    try: ref.ReflectBy = type(ref.ReflectBy)(1)
    except: pass
    ref.CoatScatterData.GetFaceData(1).Coating = "REFLECT_50"

    # 3. Film Stack Substrate (SI3N4)
    stack = add_nsc_object(TheNCE, NCE.ObjectType.RectangularVolume)
    stack.ZPosition = 11.01
    stack.Material = "SI3N4"
    stack.GetCellAt(11).DoubleValue = 2.0
    stack.GetCellAt(12).DoubleValue = 2.0
    stack.GetCellAt(13).DoubleValue = 1.0
    stack.GetCellAt(14).DoubleValue = 2.0
    stack.GetCellAt(15).DoubleValue = 2.0
    try: stack.ReflectBy = type(stack.ReflectBy)(1)
    except: pass
    # 确保名称与 COATING.DAT 完全一致
    stack.CoatScatterData.GetFaceData(1).Coating = "PSS_TiO2"

    # 4. Detector
    det = add_nsc_object(TheNCE, NCE.ObjectType.DetectorRectangle)
    det.ZPosition = -5.0
    det.GetCellAt(11).DoubleValue = 5.0
    det.GetCellAt(12).DoubleValue = 5.0
    det.GetCellAt(13).IntegerValue = 100
    det.GetCellAt(14).IntegerValue = 100
    try: det.TypeData.NormalizeCoherentPower = False
    except: pass

    # 保存
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zmx_path = os.path.join(base_dir, "Stack_Final.zmx")
    TheSystem.SaveAs(zmx_path)
    print(f"\n[完成] 模型已保存至: {zmx_path}")

if __name__ == "__main__":
    build_only()
