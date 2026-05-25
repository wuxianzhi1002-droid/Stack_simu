import numpy as np

# 加载提取出来的 npy 文件 (请根据你提取出的实际文件名修改)
saved_waves = np.load('waves.npy')
saved_intensities = np.load('intensities.npy')

print(f"已恢复波长点数: {len(saved_waves)}")
print(f"已恢复强度点数: {len(saved_intensities)}")

if len(saved_waves) == len(saved_intensities):
    print("✅ 数据对齐，可以续传！")
    last_w = saved_waves[-1]
    print(f"最后一点波长: {last_w:.5f} um")

    # 重新打包成 npz
    backup_file = "sdi_data_backup_1.npz"
    np.savez(backup_file, waves=saved_waves, intensities=saved_intensities)
    print(f"重组完成，{backup_file} 已修复。")

else:
    # 如果长度不一，取较短的那个，防止后续处理报错
    min_len = min(len(saved_waves), len(saved_intensities))
    saved_waves = saved_waves[:min_len]
    saved_intensities = saved_intensities[:min_len]
    print(f"⚠️ 数据长度不一，已自动对齐为 {min_len} 点")
