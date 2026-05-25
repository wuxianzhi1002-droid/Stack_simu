import numpy as np


def merge_npz_to_csv(file_list, output_csv="merged_sdi.csv"):
    waves_all = []
    intensities_all = []

    # 1. 读取所有数据
    for f in file_list:
        data = np.load(f)
        waves_all.append(data["waves"])
        intensities_all.append(data["intensities"])

    # 2. 拼接
    waves_all = np.concatenate(waves_all)
    intensities_all = np.concatenate(intensities_all)

    # 3. 排序
    idx = np.argsort(waves_all)
    waves_sorted = waves_all[idx]
    intensities_sorted = intensities_all[idx]

    # ===== 🔥 4. 按波长去重（保留第一个）=====
    unique_waves = []
    unique_intensities = []

    last_w = None
    tol = 1e-12  # 浮点容差（非常关键）

    for w, i in zip(waves_sorted, intensities_sorted):
        if last_w is None or abs(w - last_w) > tol:
            unique_waves.append(w)
            unique_intensities.append(i)
            last_w = w
        # 如果重复波长 → 自动跳过

    unique_waves = np.array(unique_waves)
    unique_intensities = np.array(unique_intensities)

    # 5. 保存 CSV
    result = np.vstack((unique_waves, unique_intensities)).T
    np.savetxt(
        output_csv,
        result,
        delimiter=",",
        header="Wavelength(um),Intensity",
        comments=""
    )

    print(f"完成：{output_csv}")
    print(f"原始点数: {len(waves_all)} → 去重后: {len(unique_waves)}")

    return unique_waves, unique_intensities


# ===== main =====
if __name__ == "__main__":
    files = [
        "data1.npz",
        "data2.npz"
    ]

    merge_npz_to_csv(files, "merged_sdi.csv")