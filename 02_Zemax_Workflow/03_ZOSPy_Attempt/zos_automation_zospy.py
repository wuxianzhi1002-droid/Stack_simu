import os
import numpy as np
import pandas as pd
import zospy as zp

def run_nsc_simulation_with_zospy(zemax_path=None):
    # --- 1. 连接 Zemax ---
    # 手动指定 Zemax 安装目录
    zos = zp.ZOS(opticstudio_directory=zemax_path)
    # wakeup() 会尝试连接到运行中的 Zemax，如果没运行则新建
    zos.wakeup()
    
    # 检查连接
    if not zos.connected:
        print("Could not connect to Zemax. Please make sure OpticStudio is installed.")
        return

    # 获取 API 接口
    oss = zos.ZOSAPI # TheApplication
    TheSystem = oss.PrimarySystem
    TheSystem.New(False)
    
    # 获取枚举常量 (ZOSPy 预先加载了它们)
    # 这里的 ZOSAPI 是 .NET 命名空间，ZOSPy 已经映射好了
    constants = zp.zp_constants(oss)
    
    # --- 2. 设置系统 ---
    TheSystem.MakeNonSequential()
    TheNCE = TheSystem.NCE
    
    # 设置单位
    TheSystem.SystemData.Units.LensUnits = constants.SystemData.ZemaxUnit.Millimeters
    
    # 辅助函数：创建并转换对象类型 (ZOSPy 下依然使用标准 API)
    def add_nsc_object(editor, obj_type):
        new_obj = editor.AddObject()
        settings = new_obj.GetObjectSettings(obj_type)
        new_obj.ChangeType(settings)
        return new_obj

    # 1. Source Ray
    src = add_nsc_object(TheNCE, constants.Editors.NCE.ObjectType.SourceRay)
    src.Z = 0
    src_settings = src.GetObjectSettings(constants.Editors.NCE.ObjectType.SourceRay)
    src_settings.NumberOfRays = 1
    src_settings.Power = 1.0
    
    # 2. Reference Reflector
    ref = add_nsc_object(TheNCE, constants.Editors.NCE.ObjectType.Rectangle)
    ref.Z = 10.0
    ref.Material = ""
    ref.ReflectBy = constants.Editors.NCE.ObjectReflectBy.Coatings
    ref.Coatings.Coat = "REFLECT_50"
    
    # 3. Film Stack Substrate
    stack = add_nsc_object(TheNCE, constants.Editors.NCE.ObjectType.Rectangle)
    stack.Z = 11.0
    stack.Material = "SILICON"
    stack.ReflectBy = constants.Editors.NCE.ObjectReflectBy.Coatings
    stack_names = ["PSS_TiO2", "Cr_TiO2", "PSS_HfO2", "Cr_HfO2"]
    
    # 4. Detector
    det = add_nsc_object(TheNCE, constants.Editors.NCE.ObjectType.DetectorRect)
    det.Z = -5.0
    det.X_HalfWidth = 2.0
    det.Y_HalfWidth = 2.0
    det.NumberPixelsX = 1
    det.NumberPixelsY = 1
    
    # --- 3. 运行仿真 ---
    zmx_path = os.path.join(os.getcwd(), "Stack_ZOSPy.zmx")
    TheSystem.SaveAs(zmx_path)
    print(f"System saved to {zmx_path}")
    
    wavelengths = np.linspace(0.4, 0.8, 4001)
    results_dir = os.path.join(os.getcwd(), "results_zospy")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    
    for sn in stack_names:
        print(f"Running simulation for {sn}...")
        stack.Coatings.Coat = sn
        results = []
        
        for wav in wavelengths:
            # 修改波长
            TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = wav
            
            # 清理探测器并追迹
            NSCRayTrace = TheSystem.Tools.OpenNSCRayTrace()
            NSCRayTrace.ClearDetectors(0)
            NSCRayTrace.SplitNSCRays = True
            NSCRayTrace.UsePolarization = True
            NSCRayTrace.RunAndWaitForCompletion()
            NSCRayTrace.Close()
            
            # 读取数据
            power = det.GetDetectorData(0, 0, constants.Analysis.DetectorDataType.Power)
            results.append((wav, power))
        
        # 保存结果
        df = pd.DataFrame(results, columns=['Wavelength', 'Reflection'])
        df.to_csv(os.path.join(results_dir, f"reflection_{sn}.csv"), index=False)

    print("ZOSPy Simulation Finished.")

if __name__ == "__main__":
    zemax_path = r"D:\Program Files\Ansys Zemax OpticStudio 2024 R1.00"
    run_nsc_simulation_with_zospy(zemax_path=zemax_path)
