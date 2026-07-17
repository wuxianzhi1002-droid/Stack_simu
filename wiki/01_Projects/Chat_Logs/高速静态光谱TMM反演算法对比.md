请在当前STACK_simu仓库中完成一个“高速静态光谱TMM反演算法对比”项目。

一、总体目标

建立一套完全独立、可以复制运行的静态光谱生成、TMM反演、全局优化算法比较和延迟基准测试代码。

该项目用于评估从“光谱采集完成、光谱数组已经进入内存”开始，到“输出膜厚拟合结果”为止的算法延迟。

当前只做速度和效果评估，理想目标为100 Hz以上，即单条光谱处理时间小于10 ms；不要求本轮所有全局算法都达到100 Hz，但必须报告它们距离100 Hz目标有多远，并分析后续优化空间。

二、开始前的检查

请先检查仓库结构和以下相关文件：

1.  work\01_simulation_models\01_Lumerical_Workflow\main_dynamic_v4.py 及其相关StackRT动态光谱生成代码；
2. 当前 work\02_analysis_code\tmm_joint_inversion_lockin_v3.py 相关反演代码；
3. 当前  work\04_results_and_datasets 目录结构；
4. 当前StackRT和TMM采用的材料、膜层、厚度、频率轴与复折射率约定；
5. 当前NPZ文件的数据字段和维度。

使用rg或rg --files查找文件，不要假设路径。

不要修改、覆盖或移动现有文件。所有新代码和输出必须放进一个全新的独立文件夹。

三、新文件夹

在以下位置创建：

work/04_results_and_datasets/high_speed_static_spectral_inversion_v1/

建议结构如下：

high_speed_static_spectral_inversion_v1/
├── README.md
├── requirements.txt
├── config_default.json
├── src/
│   ├── model_config.py
│   ├── tmm_stackrt_matched.py
│   ├── main_static_stackrt.py
│   ├── generate_static_dataset.py
│   ├── spectrum_preprocess.py
│   ├── objective_functions.py
│   ├── optimizer_local.py
│   ├── optimizer_sobol.py
│   ├── optimizer_de.py
│   ├── optimizer_cmaes.py
│   ├── optimizer_direct.py
│   ├── optimizer_fft_hybrid.py
│   ├── benchmark_latency.py
│   ├── evaluate_accuracy.py
│   └── run_all_benchmarks.py
├── datasets/
├── benchmark_runs/
└── report/

可以根据实际实现适当调整文件名，但必须保持模块清晰。

整个新文件夹必须自包含，不得通过以下方式依赖旧代码：

from compare_single_spectrum import ...
from tmm_joint_... import ...

应将必要的材料模型、TMM函数、配置和公共函数复制并整理到新文件夹中，使其复制到其他目录后仍能独立运行。

四、物理模型

膜层结构保持为：

RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu

当前预期真值为：

Air = 1000.0 um
HSQ = 30.0 nm
PSS = 10.0 nm
SOC = 40.0 nm
TiO2 = 40.0 nm

当前预期搜索边界为：

Air: 998.0–1002.0 um
HSQ: 20.0–40.0 nm
PSS: 1.0–20.0 nm
SOC: 30.0–50.0 nm
TiO2: 30.0–50.0 nm

请先与现有代码核对。如果现有权威配置不同，在README和报告中说明差异，并统一使用实际数据生成代码对应的配置。

TMM必须保持已经验证的StackRT匹配约定：

1. frequency = 3e8 / lambda_nominal；
2. phase_wavelength = 299792458 / frequency；
3. 材料折射率数组仍在lambda_nominal上求值；
4. 复折射率使用n+i*k；
5. 特征矩阵非对角项使用-i；
6. 正入射时使用q=n；
7. 第一层和最后一层作为半无限介质；
8. 使用Rp反射率；
9. 正入射。

首先编写随机空气腔长度下的StackRT–TMM一致性验证。只有一致性误差足够小后，才能继续进行反演算法比较。

五、静态光谱生成

以main_dynamic.py为参考，新建main_static_stackrt.py，但必须完全取消调制：

1. 不生成时间轴；
2. 不进行空气腔正弦调制；
3. 不计算锁相信号；
4. 不计算lockin_1f_X、Y、R；
5. 每组参数只生成一条静态StackRT反射光谱；
6. 支持批量随机生成不同空气腔和膜厚组合；
7. 生成过程不计入反演延迟。

默认拟合波段：

450–580 nm

默认采样间隔：

0.02 nm

预期波长点数约为6501点。请使用实际数组检查，不要直接假设点数。

不要使用stride=10、global_stride=50或其他直接抽点方式破坏干涉条纹。默认拟合使用完整450–580 nm光谱，即stride=1。

生成的NPZ至少包含：

wavelengths_um
spectra
air_cavity_um
film_thicknesses_nm
noise_sigma
wavelength_offset_nm
wavelength_scale_ppm
source_scale
source_offset
generation_parameters_json

spectra.shape应为：

(N_samples, N_wavelengths)

真值字段只能由评估程序读取，任何优化器、初值生成器、先验、损失函数和候选排序都不得读取真值。

数据集分为：

A. 理想数据：
- StackRT生成；
- 无噪声；
- 无强度漂移；
- 无波长漂移。

B. 噪声数据：
- 加性反射率噪声；
- 光源比例变化；
- 强度偏置；
- 波长零点漂移；
- 波长比例误差。

配置应支持：

- smoke：10条；
- development：30条；
- final：至少100条。

所有数据集使用固定随机种子，并保存完整配置。

如果当前运行环境没有lumapi，仍需完成StackRT数据生成代码，并提供准确的Windows运行命令。可以生成明确标注为“TMM smoke test”的临时测试数据，但绝对不能把TMM fallback数据标注成StackRT数据。

六、取消调制后的目标函数

只使用静态光谱I(lambda)进行拟合，不考虑任何调制或dI/dL信息。

统一目标函数形式：

J(theta) = sum rho((I_model-I_measured)/sigma)

所有算法必须共用同一个：

1. TMM前向模型；
2. 光谱预处理；
3. 残差定义；
4. 参数上下界；
5. 强度校正方式；
6. 最终局部精修器。

对以下强度模型：

I_measured ≈ a * I_TMM + b

a和b应优先在每个厚度候选下通过解析线性最小二乘求解，而不是加入全局优化参数。请实现变量投影，降低优化维数。

同时支持：

- linear loss；
- soft_l1 loss。

理想无噪声数据优先使用linear；噪声数据比较linear和soft_l1。

七、需要比较的算法

不要实现或比较PSO。

必须比较以下六类方法：

1. 单起点局部最小二乘

边界中心或名义值
→ scipy.optimize.least_squares

用途：
- 建立最快速度下限；
- 评估对初值的依赖；
- 用于连续跟踪模式。

2. Sobol多起点＋局部最小二乘

Sobol低差异序列生成全域候选
→ 每个候选运行least_squares
→ 选择目标函数最小结果

必须保证不插入真值。

3. 差分进化DE＋局部精修

Sobol初始种群
→ scipy.optimize.differential_evolution
→ 对候选按空气腔长度聚类
→ 每个不同腔长簇保留代表候选
→ least_squares精修

至少比较：

best1bin
rand1bin

记录种群数量、迭代数和目标函数调用次数。

4. Restart CMA-ES＋局部精修

使用CMA-ES的IPOP或BIPOP重启策略
→ 保留不同空气腔簇候选
→ least_squares精修

将必要依赖写入requirements.txt。若使用第三方cma库，应提供安装和运行说明。

5. DIRECT＋局部精修

scipy.optimize.direct
→ 获取全局候选
→ least_squares精修

作为确定性、不依赖随机种子的全局算法对照。

6. FFT/匹配滤波物理混合方法

光谱重采样到均匀波数
→ FFT或TMM空气腔匹配滤波
→ 提取前K个空气腔候选
→ 每个候选下优化薄膜厚度
→ 使用完整450–580 nm光谱进行least_squares联合精修

FFT零填充只能用于峰值插值，不能把零填充解释成真实分辨率提高。

FFT方法应保留多个候选阶次，不得只使用最高峰。

八、两种运行模式

所有适用算法分别评估：

A. 绝对盲反演模式

每条光谱独立反演，不使用上一帧结果。

用于评估：
- 首帧启动；
- 失锁重捕获；
- 全局优化能力。

B. 连续跟踪模式

第一帧使用全局方法；
后续帧使用上一帧结果作为初值；
空气腔搜索范围根据可配置的最大帧间位移缩小；
使用局部least_squares快速跟踪。

用于评估实际高速测量性能。

九、公平比较

实现两类对比。

对比A：相同目标函数调用预算

各算法设置相同的最大TMM前向调用次数或尽可能接近的预算，比较：

- 正确阶次命中率；
- 误差；
- 延迟；
- 超时率。

对比B：算法合理调优后的最佳表现

允许不同算法使用适合自己的参数，使其尽量达到目标精度，再比较达到目标精度所需时间。

随机算法至少使用多个独立随机种子，不能只报告一个种子的最好结果。

所有失败、超时或错误阶次样本必须保留在统计中，不能删除。

十、时间测量边界

主延迟严格定义为：

光谱数组已在内存
→ 预处理
→ 全局或粗搜索
→ 局部精修
→ 生成拟合结果对象和质量标志

使用：

time.perf_counter_ns()

主计时中不得包含：

- np.load；
- StackRT生成数据；
- Python import；
- 绘图；
- Markdown报告；
- 批量CSV输出；
- 大型文件写入。

每条光谱记录：

preprocess_ms
coarse_search_ms
global_search_ms
local_refine_ms
result_pack_ms
total_online_ms
disk_output_ms
n_forward_evaluations

另外单独记录：

1. 冷启动时间；
2. NPZ读取时间；
3. 结果写盘时间；
4. 热运行单帧延迟；
5. 批量吞吐率。

延迟统计必须包括：

mean
std
min
max
P50
P90
P95
P99
spectra_per_second

100 Hz可行性判据：

单条光谱处理时间小于10 ms。

报告中分别给出：

- P50是否小于10 ms；
- P95是否小于10 ms；
- P99是否小于10 ms；
- 首帧绝对反演是否达到100 Hz；
- 连续跟踪是否达到100 Hz。

目前只评估，不要求通过牺牲精度强行达到100 Hz。

十一、速度优化

在得到未优化基线后，进行以下速度优化，并分别保存优化前后的结果：

1. 预计算材料折射率数组；
2. 预计算frequency、phase_wavelength和k0；
3. 缓存与待拟合参数无关的数组；
4. 避免目标函数内重复创建大数组；
5. 使用工作数组复用；
6. 对种群候选进行批量矢量化；
7. 比较单核和多核；
8. 比较SciPy workers与NumPy矢量化；
9. 在线计时区内不绘图、不写大型文件；
10. 变量投影解析消除强度scale和offset；
11. 对连续帧使用上一帧结果；
12. 粗定位只使用能够解析条纹的有效波段；
13. 局部精修使用完整450–580 nm光谱；
14. 不允许通过未经抗混叠处理的直接抽点来加速。

如果考虑Numba、JAX或其他加速，请作为可选后端，不得使基础NumPy/SciPy版本无法独立运行。

十二、结果指标

每条光谱记录：

- 算法名称；
- 运行模式；
- 随机种子；
- success；
- timeout；
- Air拟合值与误差；
- HSQ拟合值与误差；
- PSS拟合值与误差；
- SOC拟合值与误差；
- TiO2拟合值与误差；
- 光谱RMSE；
- 最终目标函数；
- 正确空气腔阶次标志；
- 目标函数调用次数；
- 各阶段延迟；
- 总延迟。

汇总指标：

- Air MAE、RMSE、P95Abs、MaxAbs；
- 各膜层MAE、RMSE、P95Abs、MaxAbs；
- 正确阶次命中率；
- 成功率；
- 超时率；
- 光谱RMSE；
- P50/P95/P99延迟；
- 每秒处理光谱数；
- 达到100 Hz的比例；
- 不同随机种子结果离散性。

正确阶次判断阈值必须在配置和报告中明确说明，不能隐含。

十三、输出文件

每次运行创建时间戳目录：

benchmark_runs/YYYYMMDD_HHMMSS/

至少输出：

per_spectrum_results.csv
latency_breakdown.csv
algorithm_summary.csv
fitted_spectra.npz
benchmark_metadata.json
config_used.json
plots/

最终报告：

report/global_optimizer_comparison.md

报告必须完整包含：

1. 项目目的；
2. 计时边界；
3. 100 Hz定义；
4. StackRT和TMM约定；
5. 静态数据生成方法；
6. 光谱采样与混叠分析；
7. 各优化算法原理和流程；
8. 算法参数；
9. 公平预算方法；
10. 初值与真值隔离审计；
11. 反演精度；
12. 正确阶次命中率；
13. P50/P95/P99延迟；
14. 首帧绝对反演时间；
15. 连续跟踪时间；
16. 单核与多核对比；
17. 速度优化前后对比；
18. 100 Hz可行性判断；
19. 推荐的最终高速部署方案；
20. 已知限制和下一步优化建议；
21. 完整运行命令。

十四、代码质量要求

1. 使用类型标注；
2. 使用argparse提供命令行入口；
3. 配置可由JSON和命令行覆盖；
4. 固定随机种子；
5. 关键数组进行形状和单位检查；
6. 参数单位明确：Air使用um，膜厚使用nm；
7. 所有在线算法统一返回结果数据结构；
8. 不允许真值进入初值、残差、先验或候选排序；
9. 生成结果后输出真值使用审计；
10. 错误信息必须明确；
11. README给出从生成NPZ到运行基准和生成报告的完整步骤；
12. 使用matplotlib的Agg后端；
13. 保存完整硬件和软件环境信息；
14. 不修改现有文件；
15. 使用apply_patch创建和编辑代码。

十五、验证要求

至少完成：

1. Python语法编译检查；
2. TMM基本单元测试；
3. StackRT–TMM随机空气腔一致性测试；
4. 真值隔离测试；
5. 同一随机种子可重复性测试；
6. 时间边界测试，确认np.load和绘图不在主计时区；
7. 小型smoke数据集完整运行；
8. 如果环境允许，再运行development或final数据集；
9. 检查生成的CSV、JSON、NPZ和Markdown；
10. 检查报告中的结果与CSV一致。

如果当前环境不能运行Lumerical：

- 不要伪造StackRT结果；
- 完成全部代码；
- 使用明确标记的TMM smoke数据验证流程；
- 在README中给出Windows下使用lumapi生成真实StackRT数据的准确命令；
- 清楚列出尚未完成的真实StackRT测试。

十六、执行方式

先检查现有代码和仓库状态，确认权威参数与NPZ结构，然后直接实施，不要只输出计划。

实施过程中保留现有用户修改，不处理无关文件。

完成后报告：

1. 新建文件夹的完整路径；
2. 创建的关键文件；
3. 实际运行过的测试；
4. 实际生成的数据与报告；
5. 各算法目前获得的精度和时间；
6. 是否达到100 Hz；
7. 尚未完成或受环境限制的内容；
8. 最推荐的算法路线。