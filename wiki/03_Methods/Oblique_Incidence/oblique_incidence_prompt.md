你现在在我的 Lumerical/StackRT 项目目录中工作。请基于现有 01_Lumerical_Workflow 工作流，新增一套“掠入射三角测量 + 多层膜 ASD + 多通灵敏度”的完整仿真验证代码。

项目已有背景：
1. 当前项目中已经有 work/01_simulation_models/01_Lumerical_Workflow/main_angle.py、main_cavity.py、solve_npz_fft.py 等脚本。
2. 已有工作流使用 lumapi.FDTD(hide=True) 创建会话，并通过 fdtd.stackrt(n_matrix, thicknesses, freqs, theta) 做多层膜反射光谱仿真。
3. 已有 _get_n_matrix() 和 PSS_TIO2_MODEL 之类的膜层定义，请优先复用项目中已有的膜层模型，不要重写一套完全不兼容的模型。
4. 如果项目里存在 PSS_TIO2_MODEL、HSQ、BARC、Air、SiO2、Si 等层名，请自动识别并记录。Air 层代表外部空气腔或空气间隙；HSQ/光刻胶顶面是需要测量的表面。
5. 项目之前的光谱范围通常是 0.2 um 到 0.6 um，厚度单位在配置里可能是 um，但传入 stackrt 时通常需要 m。请严格检查单位，避免 um/m 混用。

本次新增目标：
建立一套适用于光刻调平调焦 level sensor 的“掠入射三角测量仿真验证流程”。它不是完整 3D 光线追迹，而是分成两层：
A. 使用 lumapi stackrt 计算多层膜在掠入射角下的反射光谱、p/s 偏振反射率、尽可能获取复反射系数相位。
B. 使用 Python 解析模型计算三角测量中的光栅像横向位移、多通几何增益、检测光栅相位读数、噪声下高度恢复误差。

请新增以下文件：

1. work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/main_grazing_stackrt.py
2. work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/solve_grazing_asd.py
3. work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/simulate_grazing_triangulation.py
4. work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/run_grazing_validation.py
5. work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/grazing_config.py

总体要求：
- 所有脚本都必须支持直接运行。
- 保持高度模块化。
- 使用中文注释。
- 使用 typing 类型提示。
- 所有输出统一保存到：
  - work/01_simulation_models/01_Lumerical_Workflow/oblique incidence//grazing/
  - work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/img/grazing/
  - work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/linear_fit/grazing/
- 所有 npz 文件名必须带时间戳。
- 所有图像必须保存 png。
- 最后生成 summary.md 和 summary.csv。

第一部分：grazing_config.py

请在 grazing_config.py 中定义 CONFIG 字典，至少包含：

CONFIG = {
    "wavelength_min_um": 0.2,
    "wavelength_max_um": 0.6,
    "wavelength_step_nm": 0.1,

    "theta_min_deg": 60.0,
    "theta_max_deg": 85.0,
    "theta_step_deg": 0.25,
    
    "polarizations": ["p", "s"],
    
    "height_scan_nm": {
        "min": -100.0,
        "max": 100.0,
        "step": 1.0
    },
    
    "multipass_list": [1, 2, 4, 6],
    
    "grating_pitch_um": 20.0,
    "imaging_magnification": 1.0,
    
    "source_type": "flat",
    "detector_noise_std": 0.002,
    "shot_noise_enable": true,
    
    "film_uncertainty_nm": 10.0,
    "film_uncertainty_mode": "uniform",
    "num_monte_carlo": 100,
    
    "exclude_perturb_layers_keywords": ["Air", "air", "Vacuum", "vacuum", "Substrate", "Si substrate"],
    
    "mirror_reflectivity": 0.98,
    "extra_mirror_count_per_wafer_pass": 2,
    
    "theta_error_scan_deg": [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
}

注意：theta 是相对于晶圆表面法线的入射角。60-85 deg 是掠入射。请在代码中明确标注。

第二部分：main_grazing_stackrt.py

功能：
1. 自动导入项目已有的膜层模型和 _get_n_matrix()。
2. 如果无法直接导入，请给出清晰错误提示，告诉我需要手动指定模型来源。
3. 使用 lumapi.FDTD(hide=True) 调用 stackrt。
4. 对 theta_axis_deg、wavelengths、polarization 进行扫描。
5. 输出 nominal stack 的 Rp/Rs 光谱。
6. 对膜层厚度做 Monte Carlo 扰动，扰动范围 ±10 nm，默认 uniform 分布。
7. Air 层、入射介质、衬底层默认不扰动；HSQ、TiO2、PSS、BARC、SiO2 等有限厚度层可扰动。
8. 保存数据到 grazing_stackrt_YYYYMMDD_HHMMSS.npz。

输出 npz 字段至少包括：
- wavelengths_m
- wavelengths_um
- freqs_Hz
- theta_axis_deg
- theta_axis_rad
- polarizations
- layer_names
- thicknesses_nominal_m
- thicknesses_mc_m
- perturbation_nm
- R_nominal_p
- R_nominal_s
- R_mc_p
- R_mc_s
- stackrt_result_keys
- config_json

如果 stackrt 返回复振幅系数，例如 rp、rs 或 r_p、r_s，请也保存：
- r_nominal_p
- r_nominal_s
- r_mc_p
- r_mc_s

如果 stackrt 只返回 Rp/Rs 强度，没有复反射系数，也不要报错；请保存现有结果，并在 summary 中说明“lumapi stackrt 未返回复相位，将在 solve_grazing_asd.py 中使用内部 TMM 计算相位”。

第三部分：solve_grazing_asd.py

功能：
从 main_grazing_stackrt.py 输出的 npz 中读取数据，计算 ASD。

核心公式：
反射系数写为：
r(theta, lambda) = |r| exp(-i psi(theta, lambda))

ASD 近似：
Z_ASD(lambda, theta) = lambda / (4*pi*sin(theta)) * dpsi/dtheta

注意：
1. theta 必须用 rad。
2. dpsi/dtheta 用 np.gradient 沿 theta 方向求导。
3. psi 必须先沿 theta 方向 unwrap。
4. 输出单位同时保存 m 和 nm。
5. 分别计算 p/s 偏振。
6. 计算宽带平均 ASD：
   ASD_bb(theta, pol) = sum(W * R * ASD) / sum(W * R)
   其中 W 默认是 flat source；后续可扩展真实光源谱。
7. 计算 Monte Carlo 膜厚扰动带来的 ASD 偏置：
   ASD_error_mc = ASD_bb_mc - ASD_bb_nominal
8. 输出每个 theta、pol 下的：
   - ASD mean
   - ASD std
   - ASD p95
   - ASD max_abs
   - 推荐最优 theta：优先选择 ASD std 小、反射率高、几何灵敏度 2sin(theta) 高的角度。

如果 npz 中已经有 complex r，则直接用它计算相位。
如果没有 complex r，请在 solve_grazing_asd.py 中实现一个内部 TMM 函数，用已有 n_matrix、thicknesses、theta、lambda 计算复反射系数 r_p/r_s。要求：
- 支持复折射率。
- 支持 p/s 偏振。
- 支持任意多层。
- 入射介质和出射衬底可以看作半无限厚。
- 内部层使用有限厚度。
- 用 TMM 计算出的 R=abs(r)^2 要和 stackrt 输出 Rp/Rs 做对比，保存误差图。如果误差很大，在 summary 中报警。

输出 npz 字段：
- wavelengths_m
- theta_axis_deg
- theta_axis_rad
- ASD_nominal_p_nm
- ASD_nominal_s_nm
- ASD_mc_p_nm
- ASD_mc_s_nm
- ASD_bb_nominal_p_nm
- ASD_bb_nominal_s_nm
- ASD_bb_mc_p_nm
- ASD_bb_mc_s_nm
- ASD_error_mc_p_nm
- ASD_error_mc_s_nm
- R_mean_p
- R_mean_s
- recommended_theta_table
- config_json

输出图像：
1. ASD_vs_wavelength_theta_p.png
2. ASD_vs_wavelength_theta_s.png
3. ASD_bb_vs_theta.png
4. ASD_mc_std_vs_theta.png
5. reflectance_vs_theta.png
6. tmm_stackrt_reflectance_compare.png，如果适用。

第四部分：simulate_grazing_triangulation.py

功能：
在 ASD 结果基础上，模拟掠入射三角测量读数。

物理模型：
单次反射几何位移：
s = 2 * sin(theta) * z

N 次同向多通：
s_N = 2 * N * M * sin(theta) * z_meas

其中：
z_meas = z_true + ASD_error_stack

检测光栅相位：
phi_g = 2*pi*s_N / p_g + phi0

四相读数：
I0   = I_bg + I_amp*cos(phi_g) + noise
I90  = I_bg + I_amp*sin(phi_g) + noise
I180 = I_bg - I_amp*cos(phi_g) + noise
I270 = I_bg - I_amp*sin(phi_g) + noise

相位解算：
phi_rec = atan2(I90 - I270, I0 - I180)

局部线性扫描时，高度恢复：
z_rec = unwrap(phi_rec - phi0) * p_g / (4*pi*N*M*sin(theta))

要求：
1. height_scan_nm 从 -100 nm 到 +100 nm。
2. 对 N = 1, 2, 4, 6 分别模拟。
3. 对 theta 选择：
   - 用户指定 theta，例如 70, 75, 80 deg；
   - 自动推荐 theta；
   - 全 theta 扫描。
4. 加入光强损失：
   P_N = P0 * R_wafer^N * mirror_reflectivity^(extra_mirror_count_per_wafer_pass * N)
   I_amp_N 按 sqrt 或线性模型可配置，默认 shot noise 下 SNR ~ sqrt(P_N)。
5. 对每个 N 计算：
   - 理想几何灵敏度 gain = 2*N*M*sin(theta)
   - 光强剩余比例
   - 随机高度噪声 std
   - 膜层 ASD 偏置 std
   - 总误差 RMS
6. 注意区分：
   - 多通会提高几何相位灵敏度；
   - 但膜层 ASD 偏置在换算成高度后不一定下降；
   - 如果光强损失严重，多通 N 过大反而会恶化随机噪声。
7. 输出 summary 表，判断最优 N 和 theta。

输出 npz 字段：
- height_true_nm
- theta_axis_deg
- multipass_list
- z_rec_nm
- z_error_nm
- random_noise_std_nm
- asd_bias_std_nm
- total_rms_nm
- optical_power_ratio
- geometric_gain
- best_theta_deg
- best_multipass_N
- config_json

输出图像：
1. height_reconstruction_N_compare.png
2. error_vs_height_N_compare.png
3. geometric_gain_vs_N.png
4. optical_power_vs_N.png
5. total_error_heatmap_theta_N.png
6. random_vs_asd_error_tradeoff.png

第五部分：run_grazing_validation.py

功能：
一键运行完整流程：
1. 调用 main_grazing_stackrt.py
2. 调用 solve_grazing_asd.py
3. 调用 simulate_grazing_triangulation.py
4. 汇总结果，生成 summary.md 和 summary.csv

summary.md 必须包含：
1. 本次膜层结构 layer_names 和 nominal thickness。
2. 掠入射角范围。
3. 波长范围。
4. p/s 偏振对比。
5. ASD 随 theta 的统计结果。
6. Monte Carlo 膜层 ±10 nm 扰动下的 ASD 偏置。
7. 多通 N=1/2/4/6 的几何增益、光强损失、随机噪声、总误差。
8. 推荐方案：
   - 推荐 theta
   - 推荐 pol
   - 推荐 N
   - 是否值得多通
9. 明确写出结论：
   - 多通是否提高随机精度；
   - 多通是否改善绝对准确度；
   - 膜层 ASD 是否成为主导误差；
   - 下一步是否需要 Zemax 做 OAP/平面镜真实光路角度容差验证。

运行方式：
在项目根目录执行：
python work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/run_grazing_validation.py

也支持分别执行：
python work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/main_grazing_stackrt.py
python work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/solve_grazing_asd.py --input 最新的 grazing_stackrt npz
python work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/simulate_grazing_triangulation.py --input 最新的 grazing_asd npz

实现细节要求：
1. 如果 lumapi import 失败，给出清晰提示，不要静默失败。
2. 不要打开 GUI，使用 hide=True。
3. 所有路径用 pathlib.Path。
4. 所有 npz 保存时包含 config_json。
5. 所有图像标题和坐标轴用英文，代码注释用中文。
6. 所有核心函数加 docstring。
7. 大数组 Monte Carlo 循环要有 tqdm 进度条；如果没有 tqdm，也要能正常运行。
8. 不要破坏现有 main_angle.py、main_cavity.py、solve_npz_fft.py。
9. 不要删除现有文件。
10. 先扫描现有项目结构，确认 PSS_TIO2_MODEL 和 _get_n_matrix() 的来源，再开始写代码。
11. 写完后运行至少一次静态检查：
    python -m py_compile work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/grazing_config.py
    python -m py_compile work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/main_grazing_stackrt.py
    python -m py_compile work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/solve_grazing_asd.py
    python -m py_compile work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/simulate_grazing_triangulation.py
    python -m py_compile work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/run_grazing_validation.py

最后请在终端输出：
- 新增文件列表
- 运行命令
- 如果 lumapi 不可用，说明代码已经完成但没有执行 stackrt；如果 lumapi 可用，则给出输出 npz 和 summary.md 路径。