import numpy as np
import os
import PythonZOSConnection as ZOS
from PythonZOSConnection import ZOSAPI
from datetime import datetime
today = datetime.now().strftime("%m%d")

def collect_sdi_data_with_resume(appid, detector_id):
    # 1. 基础参数设置
    wave_start_global = 0.5  # 500nm
    wave_end_global = 0.7  # 700nm
    delta_w = 0.00005  # 0.05nm
    backup_file = f"sdi_data_{today}_backup_{appid}.npz"

    # 2. 检查是否存在断点备份
    if os.path.exists(backup_file):
        print(f"检测到备份文件: {backup_file}，正在恢复进度...")
        data = np.load(backup_file)
        waves_done = list(data['waves'])
        intensities_done = list(data['intensities'])
        start_index = len(waves_done)
        print(f"已跳过前 {start_index} 个已完成的点。")
    else:
        print("未检测到备份，从头开始仿真...")
        waves_done = []
        intensities_done = []
        start_index = 0

    # 生成完整的波长序列
    all_waves = np.arange(wave_start_global, wave_end_global, delta_w)

    # 如果已经全部跑完了，直接返回
    if start_index >= len(all_waves):
        print("所有数据已完成，无需继续。")
        return np.array(waves_done), np.array(intensities_done)

    # 3. 连接 Zemax
    App, Sys = ZOS.connect_to_instance(appid)
    the_nsc = Sys.NCE
    wave_set = Sys.SystemData.Wavelengths
    nsc_run = Sys.Tools.OpenNSCRayTrace()

    try:
        # 只循环剩余的部分
        for i in range(start_index, len(all_waves)):
            w = all_waves[i]

            # 修改波长并追迹
            wave_set.GetWavelength(1).Wavelength = float(w)
            nsc_run.ClearDetectors(0)
            nsc_run.RunAndWaitForCompletion()

            # 获取相干数据
            success, val = the_nsc.GetCoherentData(detector_id, 0, ZOSAPI.Editors.NCE.DetectorDataType.Power, 0.0)

            current_val = val if success else 0.0
            waves_done.append(w)
            intensities_done.append(current_val)

            # 4. 进度显示与定期备份
            if i % 100 == 0:
                progress = (i + 1) / len(all_waves) * 100
                print(f"\r总进度: {progress:.2f}% | 当前波长: {w:.5f} um", end="")

            # 每 500 个点强制存盘一次
            if i > 0 and i % 500 == 0:
                np.savez(backup_file, waves=np.array(waves_done), intensities=np.array(intensities_done))

    except Exception as e:
        print(f"\n[错误] 仿真中断: {e}")
    finally:
        # 即使报错也尝试保存最后的进度
        np.savez(backup_file, waves=np.array(waves_done), intensities=np.array(intensities_done))
        nsc_run.Close()
        print(f"\n[系统] 当前进度已保存至 {backup_file}")

    # 保存最终的 CSV 结果
    final_res = np.vstack((waves_done, intensities_done)).T
    np.savetxt(f"sdi_final_{today}_results_id{appid}.csv", final_res, delimiter=",", header="Wavelength(um),Intensity",
               comments="")

    return np.array(waves_done), np.array(intensities_done)


# 主程序逻辑
if __name__ == "__main__":
    # 调用带续传功能的函数
    waves, intensities = collect_sdi_data_with_resume(appid=1, detector_id=4)
