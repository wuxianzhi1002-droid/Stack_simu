import numpy as np

def calc_axial_resolution(l0, dl):
    # 空气中的轴向分辨率 (FWHM)
    return (2 * np.log(2) / np.pi) * (l0**2 / dl)

def calc_max_range(l0, spec_res):
    # 物理距离轴 = OPD / 2
    return (l0**2 / (4 * spec_res)) / 1000.0 # um

def run_exhaustive_search():
    # 系统物理限制
    WAVELENGTH_LIMITS = (200.0, 1000.0)
    
    # 目标
    target_range = 1000.0 # um (1mm)
    
    # 搜索网格
    # 为了找到极致，我们搜索 λ0 从很短开始
    l0_range = np.linspace(200, 900, 141)
    
    best_res_at_1mm = float('inf')
    best_config = None
    
    print(f"{'λ0 (nm)':<10} | {'Δλ (nm)':<10} | {'δλ_spec (nm)':<15} | {'δL (nm)':<10} | {'Lmax (um)':<10} | {'Pixels'}")
    print("-" * 85)

    for l0 in l0_range:
        # 在 λ0 下，能取的最大带宽 Δλ 是受限于 [200, 1000] 的
        # λ_min = l0 - dl/2 >= 200 => dl <= 2*(l0 - 200)
        # λ_max = l0 + dl/2 <= 1000 => dl <= 2*(1000 - l0)
        dl_max = min(2 * (l0 - 200), 2 * (1000 - l0))
        
        if dl_max <= 0: continue
        
        # 计算该 λ0 下的最佳理论分辨率
        res = calc_axial_resolution(l0, dl_max)
        
        # 为了达到 1mm 范围，需要的光谱分辨率 sr
        # Lmax = l0^2 / (4*sr) = 1000 um => sr = l0^2 / 4000000 (单位 nm)
        sr_needed = (l0**2) / 4000000.0
        
        lmax_actual = calc_max_range(l0, sr_needed)
        pixels = dl_max / sr_needed
        
        # 记录并打印一些关键点
        if res < best_res_at_1mm:
            best_res_at_1mm = res
            best_config = (l0, dl_max, sr_needed, res, lmax_actual, pixels)
        
        # 每隔 100nm 打印一次进度
        if int(l0) % 100 == 0:
            print(f"{l0:<10.0f} | {dl_max:<10.1f} | {sr_needed:<15.4f} | {res:<10.1f} | {lmax_actual:<10.0f} | {pixels:<10.0f}")

    print("\n" + "="*50)
    print("【自动仿真搜索结果】")
    if best_config:
        l0, dl, sr, res, lmax, pix = best_config
        print(f"在系统限制 (200nm-1000nm) 内：")
        print(f"1. 能够达到的最窄轴向分辨率: {res:.1f} nm")
        print(f"2. 对应的参数配置 (满足 1mm 量程):")
        print(f"   - 中心波长 λ0: {l0:.1f} nm")
        print(f"   - 光源带宽 Δλ: {dl:.1f} nm (覆盖范围 {l0-dl/2:.1f} - {l0+dl/2:.1f} nm)")
        print(f"   - 光谱分辨率 δλ: {sr:.4f} nm")
        print(f"   - 探测器像素数: {pix:.0f} (约 {pix/1000:.1f}k 像素)")
        
        print(f"\n3. 结论:")
        if res > 100:
            print(f"   ❌ 无法在 1mm 量程下达到 100nm 以内的轴向分辨率。")
            print(f"   核心瓶颈：波长与带宽的物理限制。")
            print(f"   即使使用全光谱 (200-1000nm)，中心波长在 600nm 时最佳分辨率仅为 {res:.1f} nm。")
            print(f"   若要达到 100nm，中心波长必须远低于 200nm (进入极紫外区)，或量程大幅缩小。")
        else:
            print(f"   ✅ 成功找到符合条件的参数范围！")
    print("="*50)

if __name__ == "__main__":
    run_exhaustive_search()
