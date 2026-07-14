import os
import sys
import numpy as np

# This script is intended to be run in an environment with Zemax OpticStudio and ZOS-API.
# It uses the 'Standalone' mode of ZOS-API.

def run_nsc_stack_simulation(zemax_data_path=None, zemax_bin_path=None):
    # --- Boilerplate ZOS-API Connection ---
    import clr, os, winreg
    
    # 1. 查找数据路径 (用于存放 AGF/DAT)
    zemaxData = zemax_data_path
    if not zemaxData:
        try:
            aKey = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Zemax", 0, winreg.KEY_READ)
            zemaxData, _ = winreg.QueryValueEx(aKey, "ZemaxNativeData")
            winreg.CloseKey(aKey)
        except:
            zemaxData = os.path.join(os.environ['Documents'], 'Zemax')

    # 2. 查找二进制库路径 (DLL 所在位置)
    # 核心 DLL 通常在 Program Files 下，不在 Documents 下
    lib_path = zemax_bin_path
    if not lib_path:
        search_dirs = [
            r"C:\Program Files\Ansys Zemax OpticStudio 2024 R2",
            r"C:\Program Files\Ansys Zemax OpticStudio 2024 R1",
            r"C:\Program Files\Ansys Zemax OpticStudio 2023 R2",
            r"C:\Program Files\Zemax OpticStudio",
        ]
        # 尝试从注册表找安装目录 (如果存在)
        try:
            aKey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Zemax\OpticStudio", 0, winreg.KEY_READ)
            installDir, _ = winreg.QueryValueEx(aKey, "InstallDir")
            winreg.CloseKey(aKey)
            if installDir: search_dirs.insert(0, installDir)
        except:
            pass

        for d in search_dirs:
            test_path = os.path.join(d, "ZOS-API", "Libraries")
            if os.path.exists(os.path.join(d, "ZOSAPI.dll")): # 有些版本直接在根目录
                lib_path = d
                break
            elif os.path.exists(test_path):
                lib_path = test_path
                break

    if not lib_path or not os.path.exists(os.path.join(lib_path, "ZOSAPI.dll")):
        print(f"Error: Could not find ZOSAPI.dll.")
        print(f"Please specify zemax_bin_path in the script pointing to your OpticStudio installation folder.")
        return

    print(f"Found ZOS-API Libraries at: {lib_path}")
    sys.path.append(lib_path)
    
    # 使用绝对路径引用以防万一
    clr.AddReference(os.path.join(lib_path, 'ZOSAPI_NetHelper.dll'))
    clr.AddReference(os.path.join(lib_path, 'ZOSAPI.dll'))
    clr.AddReference('ZOSAPI_NetHelper')
    clr.AddReference('ZOSAPI')
    
    # --- Initialize Connection ---
    import ZOSAPI
    
    TheApplication = None
    Connection = ZOSAPI.ZOSAPI_Connection()
    
    print("Attempting to connect to ZOS-API...")
    
    # 优先尝试连接到已经打开的 Zemax (交互模式)
    try:
        TheApplication = Connection.ConnectToApplication()
        if TheApplication:
            print("Successfully connected to a running OpticStudio instance.")
    except:
        pass

    # 如果没找到运行中的，则尝试新建 (后台模式)
    if not TheApplication:
        try:
            TheApplication = Connection.CreateNewApplication()
            if TheApplication:
                print("Successfully started a new OpticStudio instance.")
        except Exception as e:
            print(f"Failed to create new application: {e}")

    if not TheApplication:
        print("\nCRITICAL ERROR: Could not establish ZOS-API connection.")
        return
    
    TheSystem = TheApplication.PrimarySystem
    # 如果是新建的，通常需要 New()；如果是连接现有的，通常不需要
    # 我们这里总是尝试创建一个干净的状态
    TheSystem.New(False)
    
    # --- Setup NSC System ---
    TheSystem.MakeNonSequential()
    TheNCE = TheSystem.NCE
    
    # 设置单位 - 显式转换枚举 (Python.NET 3.0+ 要求)
    try:
        # 使用枚举构造函数将整数 1 转换为 ZemaxUnit.Millimeters
        TheSystem.SystemData.Units.LensUnits = ZOSAPI.SystemData.ZemaxUnit(1)
        print("Set LensUnits to Millimeters (1).")
    except Exception as e:
        print(f"Warning: Could not set LensUnits: {e}")
    
    # 辅助函数：创建并转换对象类型
    def add_nsc_object(editor, obj_type):
        new_obj = editor.AddObject() # 先添加一个空对象 (Null Object)
        # 获取目标类型的设置对象
        settings = new_obj.GetObjectSettings(obj_type)
        # 更改对象类型
        new_obj.ChangeType(settings)
        return new_obj

    # 1. Source Ray (Plane Wave)
    src = add_nsc_object(TheNCE, ZOSAPI.Editors.NCE.ObjectType.SourceRay)
    src.Z = 0
    # 获取设置以修改属性
    src_settings = src.GetObjectSettings(ZOSAPI.Editors.NCE.ObjectType.SourceRay)
    src_settings.NumberOfRays = 1
    src_settings.Power = 1.0
    
    # 2. Reference Reflector (Rectangle)
    ref = add_nsc_object(TheNCE, ZOSAPI.Editors.NCE.ObjectType.Rectangle)
    ref.Z = 10.0 # Positioned at 10mm
    ref.Material = "" # Mirror surface
    # 显式转换枚举
    ref.ReflectBy = ZOSAPI.Editors.NCE.ObjectReflectBy(1) # 1 通常代表 Coatings
    # Note: REFLECT_50 must be in the COATING.DAT
    ref.Coatings.Coat = "REFLECT_50"
    
    # 3. Film Stack Substrate (Rectangle)
    stack = add_nsc_object(TheNCE, ZOSAPI.Editors.NCE.ObjectType.Rectangle)
    stack.Z = 11.0 # 1mm Air Gap
    stack.Material = "SILICON" # Substrate material
    stack.ReflectBy = ZOSAPI.Editors.NCE.ObjectReflectBy(1)
    # We will loop through stack names: PSS_TiO2, Cr_TiO2, etc.
    stack_names = ["PSS_TiO2", "Cr_TiO2", "PSS_HfO2", "Cr_HfO2"]
    
    # 4. Detector Rect (To catch reflection)
    det = add_nsc_object(TheNCE, ZOSAPI.Editors.NCE.ObjectType.DetectorRect)
    det.Z = -5.0 # Placed behind the source to catch return rays
    # 设置探测器尺寸
    det.X_HalfWidth = 2.0
    det.Y_HalfWidth = 2.0
    det.NumberPixelsX = 1
    det.NumberPixelsY = 1
    
    # --- Save System ---
    zmx_path = os.path.join(os.getcwd(), "StackInterferometer.zmx")
    TheSystem.SaveAs(zmx_path)
    print(f"System saved to {zmx_path}")
    
    # --- Simulation Parameters ---
    wavelengths = np.linspace(0.4, 0.8, 4001) # 0.1nm step
    results_dir = os.path.join(os.getcwd(), "results_zemax")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    
    for sn in stack_names:
        print(f"Running simulation for {sn}...")
        stack.Coatings.Coat = sn
        results = []
        
        for wav in wavelengths:
            # Set system wavelength
            TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = wav
            
            # Clear Detector
            NSCRayTrace = TheSystem.Tools.OpenNSCRayTrace()
            NSCRayTrace.ClearDetectors(0)
            
            # Run Ray Trace
            NSCRayTrace.SplitNSCRays = True
            NSCRayTrace.UsePolarization = True
            NSCRayTrace.RunAndWaitForCompletion()
            NSCRayTrace.Close()
            
            # Get Detector Result
            power = det.GetDetectorData(0, 0, ZOSAPI.Analysis.DetectorDataType.Power)
            results.append((wav, power))
        
        # Save results to CSV
        csv_path = os.path.join(results_dir, f"reflection_{sn}.csv")
        try:
            df = pd.DataFrame(results, columns=['Wavelength', 'Reflection'])
            df.to_csv(csv_path, index=False)
        except:
            np.savetxt(csv_path, np.array(results), delimiter=",", header="Wavelength,Reflection")
        
        print(f"Saved {csv_path}")

    # --- Generate Report ---
    report_path = os.path.join(results_dir, "Zemax_Simulation_Report.md")
    with open(report_path, "w") as f:
        f.write("# Zemax NSC Simulation Report\n\n")
        f.write("## System Configuration\n")
        f.write("- **Reference Reflector**: 50% R (Ideal Coating)\n")
        f.write("- **Air Gap**: 1.0 mm\n")
        f.write("- **Film Stack**: Multi-layer coating on Silicon\n")
        f.write(f"- **Wavelength Range**: 400nm - 800nm\n")
        f.write("- **Mode**: NSC with Split Rays & Polarization\n\n")
        f.write("## Results\n")
        f.write("Simulation completed for all stacks. See CSV files for detailed data.\n")
    
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    # Import pandas only when needed as it might not be in Zemax environment
    try:
        import pandas as pd
    except ImportError:
        print("Pandas not found. Results will be saved using numpy.")
    
    # 使用用户提供的路径
    run_nsc_stack_simulation(
        zemax_data_path=r"D:\Users\wuxianzhi\Documents\Zemax",
        zemax_bin_path=r"D:\Program Files\Ansys Zemax OpticStudio 2024 R1.00"
    )
