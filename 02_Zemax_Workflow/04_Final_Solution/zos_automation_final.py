import clr, os, winreg
import numpy as np
import pandas as pd

# =============================================================================
# Zemax 官方样板类封装 (基于用户提供的 refs)
# =============================================================================
class ZemaxStandalone:
    def __init__(self, bin_path):
        # 1. 加载 NetHelper
        net_helper_path = os.path.join(bin_path, 'ZOSAPI_NetHelper.dll')
        if not os.path.exists(net_helper_path):
            net_helper_path = os.path.join(bin_path, r'ZOS-API\Libraries\ZOSAPI_NetHelper.dll')
            
        clr.AddReference(net_helper_path)
        import ZOSAPI_NetHelper
        
        # 2. 初始化 API
        if not ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(bin_path):
            raise Exception("Unable to initialize ZOSAPI with path: " + bin_path)
        
        dir_zos = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()
        
        # 3. 加载核心 DLL
        clr.AddReference(os.path.join(dir_zos, "ZOSAPI.dll"))
        clr.AddReference(os.path.join(dir_zos, "ZOSAPI_Interfaces.dll"))
        import ZOSAPI
        
        self.ZOSAPI = ZOSAPI
        self.TheConnection = ZOSAPI.ZOSAPI_Connection()
        self.TheApplication = self.TheConnection.CreateNewApplication()
        
        if not self.TheApplication.IsValidLicenseForAPI:
            raise Exception("License is not valid for ZOSAPI use")
            
        self.TheSystem = self.TheApplication.PrimarySystem

    def __del__(self):
        if hasattr(self, 'TheApplication') and self.TheApplication:
            self.TheApplication.CloseApplication()

# =============================================================================
# 仿真逻辑实现
# =============================================================================
def run_simulation():
    # --- 1. 环境准备 ---
    bin_path = r"D:\Program Files\Ansys Zemax OpticStudio 2024 R1.00"
    
    print(f"Connecting to OpticStudio at {bin_path}...")
    zos = ZemaxStandalone(bin_path)
    ZOSAPI = zos.ZOSAPI
    TheSystem = zos.TheSystem
    TheSystem.New(False)
    
    # 获取快捷引用
    NCE = ZOSAPI.Editors.NCE
    
    # --- 2. 系统设置 ---
    TheSystem.MakeNonSequential()
    TheNCE = TheSystem.NCE
    
    # 设置单位 - 动态获取枚举类型并转换
    try:
        # 获取 LensUnits 属性当前的类型（枚举类），然后用 1 (Millimeters) 实例化它
        unit_enum_type = type(TheSystem.SystemData.Units.LensUnits)
        TheSystem.SystemData.Units.LensUnits = unit_enum_type(1)
        print(f"System units set to Millimeters.")
    except Exception as e:
        print(f"Warning: Could not set LensUnits dynamically: {e}")

    # 1. 提高非序列光线追迹相对阈值强度到 1e-5 (用户要求)
    try:
        TheSystem.SystemData.NonSequentialData.MinimumRelativeRayIntensity = 1e-5
        print("NSC Minimum Relative Ray Intensity set to 1e-5.")
    except Exception as e:
        print(f"Warning: Could not set MinimumRelativeRayIntensity: {e}")

    # 辅助函数：创建并转换对象类型
    def add_nsc_object(editor, obj_type):
        new_obj = editor.AddObject() # 先添加一个空对象 (Null Object)
        # 获取目标类型的设置对象
        settings = new_obj.GetObjectTypeSettings(obj_type)
        # 更改对象类型
        new_obj.ChangeType(settings)
        return new_obj

    # 1. Source Ray
    src = add_nsc_object(TheNCE, NCE.ObjectType.SourceRay)
    src.ZPosition = 0.0
    # 显式设置光线数和功率 (Cell 11: Layout, 12: Analysis, 13: Power)
    src.GetCellAt(11).IntegerValue = 100   
    src.GetCellAt(12).IntegerValue = 1000  
    src.GetCellAt(13).DoubleValue = 1.0    
    
    # 2. Reference Reflector (使用 RectangularVolume)
    ref = add_nsc_object(TheNCE, NCE.ObjectType.RectangularVolume)
    ref.ZPosition = 10.0
    ref.Material = "" # 设为空代表空气界面
    ref.GetCellAt(11).DoubleValue = 2.0 # X1 Half Width
    ref.GetCellAt(12).DoubleValue = 2.0 # Y1 Half Width
    ref.GetCellAt(13).DoubleValue = 0.01 # Z Length (薄片)
    ref.GetCellAt(14).DoubleValue = 2.0 # X2
    ref.GetCellAt(15).DoubleValue = 2.0 # Y2
    
    try:
        ref.ReflectBy = type(ref.ReflectBy)(1) # 1 为 Coatings
    except:
        pass
    ref.CoatScatterData.GetFaceData(1).Coating = "REFLECT_50"
    
    # 3. Film Stack Substrate (改为 SI3N4 衬底，用户要求)
    stack = add_nsc_object(TheNCE, NCE.ObjectType.RectangularVolume)
    stack.ZPosition = 11.01 # 1mm Air Gap (10 + 0.01 + 1.0)
    stack.Material = "SI3N4" 
    stack.GetCellAt(11).DoubleValue = 2.0
    stack.GetCellAt(12).DoubleValue = 2.0
    stack.GetCellAt(13).DoubleValue = 1.0 # 衬底厚度
    stack.GetCellAt(14).DoubleValue = 2.0
    stack.GetCellAt(15).DoubleValue = 2.0
    
    try:
        stack.ReflectBy = type(stack.ReflectBy)(1)
    except:
        pass
    
    stack_names = ["PSS_TiO2", "Cr_TiO2", "PSS_HfO2", "Cr_HfO2"]
    
    # 4. Detector
    det = add_nsc_object(TheNCE, NCE.ObjectType.DetectorRectangle)
    det.ZPosition = -5.0
    det.GetCellAt(11).DoubleValue = 5.0 
    det.GetCellAt(12).DoubleValue = 5.0 
    det.GetCellAt(13).IntegerValue = 1   
    det.GetCellAt(14).IntegerValue = 1   
    
    # 5. 探测器设置：不要勾选归一化相干功率 (用户要求)
    try:
        det.TypeData.NormalizeCoherentPower = False
        print("Detector 'Normalize Coherent Power' disabled.")
    except Exception as e:
        print(f"Warning: Could not disable NormalizeCoherentPower: {e}")
    
    # --- 3. 运行波长扫描 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    zmx_path = os.path.join(base_dir, "Stack_Final.zmx")
    TheSystem.SaveAs(zmx_path)
    print(f"Project saved to {zmx_path}")
    
    # 高分辨率采样 (0.01nm 步长)
    wavelengths = np.linspace(0.4, 0.8, 40001) 
    results_dir = os.path.join(base_dir, "results_final")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    
    print(f"Starting High-Resolution wavelength sweep ({len(wavelengths)} points per stack)...")
    
    for sn in stack_names:
        print(f"Processing Stack: {sn}")
        stack.CoatScatterData.GetFaceData(1).Coating = sn
        data_list = []
        
        for i, wav in enumerate(wavelengths):
            if i % 1000 == 0:
                print(f"  Lambda: {wav:.4f} um ({i}/{len(wavelengths)})")
            
            TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = wav
            
            NSCRayTrace = TheSystem.Tools.OpenNSCRayTrace()
            NSCRayTrace.ClearDetectors(0)
            NSCRayTrace.SplitNSCRays = True
            NSCRayTrace.UsePolarization = True
            NSCRayTrace.RunAndWaitForCompletion()
            NSCRayTrace.Close()
            
            # 读取探测器的总功率
            # Power = 3. ByRef 返回 (bool, value)
            success, power = TheNCE.GetDetectorData(det.ObjectNumber, 0, 3, 0.0)
            data_list.append((wav, power if success else 0.0))
            
        df = pd.DataFrame(data_list, columns=['Wavelength', 'Reflection'])
        df.to_csv(os.path.join(results_dir, f"reflection_{sn}.csv"), index=False)
        print(f"Saved results for {sn}")

    print("\nSimulation successfully completed!")

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"\nSimulation Failed: {e}")
        import traceback
        traceback.print_exc()
