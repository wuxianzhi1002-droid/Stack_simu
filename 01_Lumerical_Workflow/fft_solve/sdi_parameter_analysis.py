"""
SDI (Spectral Domain Interferometry) 系统参数分析工具
==================================================

本脚本用于分析低相干干涉系统中“轴向分辨率”和“最大无模糊测量范围”与硬件参数的关系。

【物理背景解释】
1. 轴向分辨率 (Axial Resolution, δL) 为何与带宽相关？
   轴向分辨率本质上由光源的相干长度决定。根据傅里叶变换性质，光谱越宽（带宽 Δλ 越大），
   其在空间域对应的相干函数越窄。对于高斯谱光源，δL 与带宽成反比。

2. 最大测量范围 (Max Measurement Range, Lmax) 为何受光谱分辨率限制？
   干涉信号在波数 k 域表现为周期性振荡，腔长 L 越大，振荡越快。
   根据奈奎斯特采样定律，光谱仪的采样间隔（光谱分辨率 δλ_spec）必须足够细，
   才能捕捉到高频振荡。采样越密，能还原的振荡频率越高，即测量范围越大。

3. 两者之间的物理 Trade-off：
   在光谱仪像素总数固定的工程限制下，若想获得极高的分辨率（大带宽），
   必然导致采样间隔变大（光谱分辨率变差），从而缩小了测量范围。

【公式推导】
1. 轴向分辨率 (物理空气腔长):
   δL = (2*ln2 / π) * (λ0² / Δλ) ≈ 0.44 * (λ0² / Δλ)
   
2. 最大测量范围 (物理空气腔长):
   采样频率 f_s = 1 / Δk
   波数 k = 2π / λ, 采样间隔 Δk ≈ |dk/dλ| * δλ_spec = (2π / λ0²) * δλ_spec
   由 Nyquist 定律，最大光程差 OPD_max = π / Δk = λ0² / (2 * δλ_spec)
   物理距离 Lmax = OPD_max / 2 = λ0² / (4 * δλ_spec)

作者: Gemini CLI
日期: 2024-05-29
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ================= 配置参数 =================
# 波长单位均为 nm
L_MIN_RANGE = (200, 1000)
DELTA_L_RANGE = (100, 800)
SPEC_RES_RANGE = (0.01, 0.2)  # nm

# 绘图点数
GRID_SIZE = 200

def calc_axial_resolution(lambda0, delta_lambda):
    """计算轴向分辨率 (nm)"""
    return (2 * np.log(2) / np.pi) * (lambda0**2 / delta_lambda)

def calc_max_range(lambda0, spec_res):
    """计算最大测量范围 (um)"""
    # 公式: Lmax = lambda0^2 / (4 * spec_res)
    # 输入 nm, 输出 nm, 转换成 um
    return (lambda0**2 / (4 * spec_res)) / 1000.0

def run_analysis():
    # 1. 准备数据：轴向分辨率 vs 带宽 (固定中心波长)
    plt.figure(figsize=(15, 12))
    
    # 子图 1: 轴向分辨率 vs 带宽
    plt.subplot(2, 2, 1)
    delta_lambdas = np.linspace(DELTA_L_RANGE[0], DELTA_L_RANGE[1], 100)
    l0_points = np.arange(300, 951, 50)
    for l0 in l0_points:
        # 验证合法性: l0 - dl/2 >= 200 and l0 + dl/2 <= 1000
        mask = (l0 - delta_lambdas/2 >= 200) & (l0 + delta_lambdas/2 <= 1000)
        if not np.any(mask): continue
        dl_valid = delta_lambdas[mask]
        res = calc_axial_resolution(l0, dl_valid)
        plt.plot(dl_valid, res / 1000.0, label=f'λ0 = {l0} nm')
    plt.xlabel('Bandwidth Δλ (nm)')
    plt.ylabel('Axial Resolution δL (μm)')
    plt.title('Axial Resolution vs Bandwidth')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)

    # 子图 2: 最大测量范围 vs 光谱分辨率
    plt.subplot(2, 2, 2)
    spec_res_list = np.linspace(SPEC_RES_RANGE[0], SPEC_RES_RANGE[1], 100)
    for l0 in l0_points:
        # 验证 λ0 是否在系统允许范围内 (考虑最小带宽)
        if l0 - DELTA_L_RANGE[0]/2 < 200 or l0 + DELTA_L_RANGE[0]/2 > 1000:
            continue
        max_ranges = calc_max_range(l0, spec_res_list)
        plt.plot(spec_res_list, max_ranges, label=f'λ0 = {l0} nm')
    plt.xlabel('Spectral Resolution δλ_spec (nm)')
    plt.ylabel('Max Range Lmax (μm)')
    plt.title('Max Range vs Spectral Resolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 准备 2D 热力图数据 (Subplot 3: x=Δλ, y=λ0)
    dl_vec = np.linspace(DELTA_L_RANGE[0], DELTA_L_RANGE[1], GRID_SIZE)
    l0_vec = np.linspace(200, 1000, GRID_SIZE)
    DL, L0_GRID3 = np.meshgrid(dl_vec, l0_vec)
    
    # 计算 λ_min, λ_max 并建立掩模
    L_MIN_CALC = L0_GRID3 - DL / 2.0
    L_MAX_CALC = L0_GRID3 + DL / 2.0
    MASK3 = (L_MIN_CALC >= 200) & (L_MAX_CALC <= 1000)
    
    # 轴向分辨率热力图
    RES_MAP = np.full(L0_GRID3.shape, np.nan)
    RES_MAP[MASK3] = calc_axial_resolution(L0_GRID3[MASK3], DL[MASK3]) / 1000.0 # um

    # 子图 3: 轴向分辨率热力图
    plt.subplot(2, 2, 3)
    im3 = plt.imshow(RES_MAP, origin='lower', extent=[DELTA_L_RANGE[0], DELTA_L_RANGE[1], 200, 1000],
               aspect='auto', cmap='viridis_r', norm=LogNorm())
    plt.colorbar(im3, label='δL (μm)')
    plt.xlabel('Bandwidth Δλ (nm)')
    plt.ylabel('Center Wavelength λ0 (nm)')
    plt.title('Axial Resolution Heatmap')

    # 子图 4: 最大测量范围热力图 (x=δλ_spec, y=λ0)
    plt.subplot(2, 2, 4)
    spec_vec = np.linspace(SPEC_RES_RANGE[0], SPEC_RES_RANGE[1], GRID_SIZE)
    l0_vec_4 = np.linspace(250, 950, GRID_SIZE) # 稍微收缩范围以获得更好视觉效果
    SPEC, L0_GRID4 = np.meshgrid(spec_vec, l0_vec_4)
    MAX_RANGE_MAP = calc_max_range(L0_GRID4, SPEC)
    
    im4 = plt.imshow(MAX_RANGE_MAP, origin='lower', extent=[SPEC_RES_RANGE[0], SPEC_RES_RANGE[1], 250, 950],
               aspect='auto', cmap='plasma', norm=LogNorm())
    plt.colorbar(im4, label='Lmax (μm)')
    plt.xlabel('Spectral Resolution δλ_spec (nm)')
    plt.ylabel('Center Wavelength λ0 (nm)')
    plt.title('Max Range Heatmap')

    plt.tight_layout()
    plt.savefig('sdi_analysis_results.png', dpi=150)
    print("✅ 分析图像已保存为 sdi_analysis_results.png")
    plt.show()

    # --- 打印工程建议 ---
    print("\n" + "="*40)
    print("【工程实施建议】")
    print("1. 如何获得高分辨率 + 大测量范围？")
    print("   - 选择较短的中心波长 (如 400nm) 可以显著提升分辨率 (δL ∝ λ0²)，但会牺牲测量范围 (Lmax ∝ λ0²)。")
    print("   - 关键在于使用 高线数光栅 以获得极小的 δλ_spec，并配合 宽光谱光源。")
    print("2. 参数敏感度：")
    print("   - 中心波长 λ0 是最敏感参数，它对分辨率和范围都是平方级影响。")
    print("   - 光谱分辨率 δλ_spec 直接决定了能否看清深层结构，是系统的硬门槛。")
    print("3. 限制因素：")
    print("   - 探测器像素数: N = Δλ / δλ_spec。目前的 CMOS 线阵相机通常为 2048 或 4096 像素。")
    print("   - 若固定 N=2048，带宽 Δλ=200nm，则 δλ_spec ≈ 0.1nm。")
    print("     在 800nm 下，Lmax ≈ 800^2 / (4 * 0.1) = 1.6 mm。")
    print("="*40)

if __name__ == "__main__":
    run_analysis()
