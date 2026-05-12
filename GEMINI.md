# Lumerical STACK 多层光刻胶膜层反射率建模项目

本项目旨在建立一个可扩展的曝光前 film stack 1D 无限平面膜层光学模型，用于调焦调平相关研究。利用 Ansys Lumerical 2024 的 STACK 模块和 Python API 实现自动化仿真与分析。

## 1. 核心架构

项目采用模块化设计，便于后续扩展至 FDTD 或角度扫描等复杂模型：

- **`materials.py`**: 定义材料属性。包含 HSQ、SOC、PSS 等自定义材料的色散模型（Cauchy 近似），并负责将采样数据注入 Lumerical。
- **`stack_builder.py`**: 膜层结构定义。配置对比实验所需的四种 Stack（PSS vs Cr, TiO2 vs HfO2）。
- **`simulation.py`**: 仿真执行引擎。通过 `lumapi` 调用 `stackrt` 函数进行传输矩阵法（TMM）计算，并能自动生成 `.fsp` 文件供 FDTD 可视化查看。
- **`analysis.py`**: 结果分析模块。检查物理合法性，量化干涉振荡（振幅、峰值数），并提供对比洞察。
- **`export.py`**: 数据与报告导出。生成 CSV 数据、对比图表及最终的 Markdown 仿真报告。
- **`main.py`**: 项目主入口。编排整个仿真生命周期。

## 2. 环境配置

- **软件**: Ansys Lumerical 2024
- **Python**: 推荐使用 **Anaconda** 环境（如 `my_ML`）以获得更好的包管理稳定性。
  - **环境要求**: Python 3.9 - 3.11
  - **必要操作**: 请在你的 Anaconda 环境中安装以下依赖：
    ```powershell
    pip install numpy pandas matplotlib
    ```
- **Lumerical API 链接**: 脚本会自动寻找 `D:\Program Files\Lumerical\v241\api\python` 路径。

## 3. 使用方法

激活你的 Anaconda 环境后，在项目根目录下运行：

```powershell
# 激活环境 (示例)
conda activate my_ML

# 运行仿真
python main.py
```

执行后，可以在 `results/` 目录下找到各膜层结构的 `.fsp` 文件，直接双击使用 Lumerical 软件打开即可查看 3D 建模结果。

## 4. 关键成果 (2026-05-11)

- **FDTD 可视化集成**: 现在已支持为每个膜层配置自动生成基于真实物理厚度的 `.fsp` 工程文件，可以在软件中直观地看到建模结果。
- **反射率分析**: 发现导电层（PSS vs Cr）对反射率有决定性影响（平均值差异达 ~52%）。
- **干涉信号**: PSS 方案保留了更强的薄膜干涉振荡特征，有利于调焦调平信号的提取。
- **自动化流**: 实现了从材料定义到分析报告生成的全自动闭环。

## 5. 后续扩展建议

- **角度扫描**: 引入非垂直入射的角度扫描（Angle sweep）。
- **FDTD 集成**: 在 `simulation.py` 中扩展 FDTD 区域，处理 3D 结构或粗糙度。
- **灵敏度分析**: 针对膜层厚度公差进行蒙特卡洛仿真。

## 6. 编码规范

- **文件编码**: 请使用 **UTF-8 with BOM** 编码读取和保存文档，以防止中文出现乱码。
