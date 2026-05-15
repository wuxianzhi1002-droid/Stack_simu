import clr, os, winreg
import numpy as np
import pandas as pd

# =============================================================================
# Zemax 官方样板类封装 (基于用户提供的 refs)
# =============================================================================
class ZemaxStandalone:
    def __init__(self, bin_path):
        # 1. 加载 NetHelper
        net_helper_path = os.path.join(bin_path, r'ZOS-API\Libraries\ZOSAPI_NetHelper.dll')
        if not os.path.exists(net_helper_path):
            # 尝试直接在根目录找
            net_helper_path = os.path.join(bin_path, 'ZOSAPI_NetHelper.dll')
            
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
    data_path = r"D:\Users\wuxianzhi\Documents\Zemax"
    
    print(f"Connecting to OpticStudio at {bin_path}...")
    zos = ZemaxStandalone(bin_path)
    ZOSAPI = zos.ZOSAPI
    TheSystem = zos.TheSystem
    TheSystem.New(False)
    
    # 获取快捷引用
    NCE = ZOSAPI.Editors.NCE
    SystemData = ZOSAPI.SystemData
    
    # --- 2. 系统设置 ---
    TheSystem.MakeNonSequential()
    TheNCE = TheSystem.NCE
    
    # 设置单位 - 动态获取枚举类型并转换
    try:
        # 获取 LensUnits 属性当前的类型（枚举类），然后用 1 (Millimeters) 实例化它
        unit_enum_type = type(TheSystem.SystemData.Units.LensUnits)
        TheSystem.SystemData.Units.LensUnits = unit_enum_type(1)
        print(f"System units set to Millimeters using dynamic enum: {unit_enum_type}")
    except Exception as e:
        print(f"Warning: Could not set LensUnits dynamically: {e}")

    # 辅助函数：创建并转换对象类型
    def add_nsc_object(editor, obj_type):
        new_obj = editor.AddObject() # 先添加一个空对象 (Null Object)
        # 获取目标类型的设置对象 (修正方法名为 GetObjectTypeSettings)
        settings = new_obj.GetObjectTypeSettings(obj_type)
        # 更改对象类型
        new_obj.ChangeType(settings)
        return new_obj

    # 1. Source Ray
    src = add_nsc_object(TheNCE, NCE.ObjectType.SourceRay)
    src.Z = 0
    # 获取设置以修改属性
    src_settings = src.GetObjectTypeSettings(NCE.ObjectType.SourceRay)
    src_settings.NumberOfRays = 1
    src_settings.Power = 1.0
    
    # 2. Reference Reflector
    ref = add_nsc_object(TheNCE, NCE.ObjectType.Rectangle)
    ref.Z = 10.0
    ref.Material = ""
    # 动态获取 ReflectBy 枚举类型并设置
    try:
        reflect_enum_type = type(ref.ReflectBy)
        ref.ReflectBy = reflect_enum_type(1) # 1 为 Coatings
    except:
        pass
    # 修正：通过 CoatScatterData 访问 Coatings
    ref.CoatScatterData.GetFaceData(1).Coating = "REFLECT_50"
    
    # 3. Film Stack Substrate
    stack = add_nsc_object(TheNCE, NCE.ObjectType.Rectangle)
    stack.Z = 11.0
    stack.Material = "SILICON"
    try:
        stack.ReflectBy = type(stack.ReflectBy)(1)
    except:
        pass
    # 修正：后续循环中设置 Coating
    stack_names = ["PSS_TiO2", "Cr_TiO2", "PSS_HfO2", "Cr_HfO2"]
    
    # 4. Detector
    # 修正：使用正确的类型名 DetectorRectangle
    det = add_nsc_object(TheNCE, NCE.ObjectType.DetectorRectangle)
    det.Z = -5.0
    # 修正：通过 GetCellAt 稳健地设置探测器属性
    det.GetCellAt(11).DoubleValue = 2.0 # X Half Width
    det.GetCellAt(12).DoubleValue = 2.0 # Y Half Width
    det.GetCellAt(13).IntegerValue = 1   # X Pixels
    det.GetCellAt(14).IntegerValue = 1   # Y Pixels
    
    # --- 3. 运行波长扫描 ---
    zmx_path = os.path.join(os.getcwd(), "Stack_Final.zmx")
    TheSystem.SaveAs(zmx_path)
    print(f"Project saved to {zmx_path}")
    
    # 恢复高分辨率采样 (0.1nm 步长)
    wavelengths = np.linspace(0.4, 0.8, 4001)
    # wavelengths = np.linspace(0.4, 0.8, 401) # 测试用
    results_dir = os.path.join(os.getcwd(), "results_final")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    
    print(f"Starting wavelength sweep ({len(wavelengths)} points per stack)...")
    
    for sn in stack_names:
        print(f"Processing Stack: {sn}")
        # 修正：设置 Coating
        stack.CoatScatterData.GetFaceData(1).Coating = sn
        data_list = []
        
        for i, wav in enumerate(wavelengths):
            if i % 100 == 0:
                print(f"  Lambda: {wav:.4f} um ({i}/{len(wavelengths)})")
            
            # 更新波长
            TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = wav
            
            # 运行追迹
            NSCRayTrace = TheSystem.Tools.OpenNSCRayTrace()
            NSCRayTrace.ClearDetectors(0)
            NSCRayTrace.SplitNSCRays = True
            NSCRayTrace.UsePolarization = True
            NSCRayTrace.RunAndWaitForCompletion()
            NSCRayTrace.Close()
            
            # 读取探测器的总功率
            # 根据报错信息，签名是 GetDetectorData(Int32, Int32, Int32, Double ByRef)
            # Power = 3
            # 在 pythonnet 中，ByRef 通常返回一个元组 (bool, value)
            success, power = TheNCE.GetDetectorData(det.ObjectNumber, 0, 3, 0.0)
            if success:
                data_list.append((wav, power))
            else:
                data_list.append((wav, 0.0))
            
        # 保存 CSV
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
