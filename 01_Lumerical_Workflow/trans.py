import pandas as pd
import os

# 1. 配置文件路径（请确保 data1.txt 放在该脚本同级目录下）
input_file = './stackrt_result/PSS_TiO2_reflection.csv'
output_file = 'stackrt_result/raw_stack_length1mm.csv'

if not os.path.exists(input_file):
    print(f"错误：找不到输入文件 {input_file}，请检查路径。")
    exit()

try:
    print(f"正在读取 {input_file} ...")

    # 核心改进：sep=None + engine='python' 允许 pandas 自动检测是逗号、Tab还是空格分隔
    # skiprows=0，如果你的 txt 文件最上面有“Multilayer Spectrum...”等闲聊空行，请改为 skiprows=2
    df = pd.read_csv(input_file, sep=None, engine='python', skipinitialspace=True)

    # 打印读取到的原始列名，方便你调试查阅
    print("读取到的原始表头为:", list(df.columns))

    # 2. 智能化列名清洗与波长单位换算
    # 不管原始列名叫什么，我们直接强制重命名为标准格式
    # 假设第 0 列是波长，第 1 列是强度
    old_cols = df.columns
    df.rename(columns={old_cols[0]: 'Wavelength(um)', old_cols[1]: 'Intensity'}, inplace=True)

    # 【核心避坑】检查数据是否需要从 nm 转为 um
    # 如果检测到第一行的波长数值大于 10 (例如 500), 说明它是 nm 单位，自动除以 1000
    first_wavelength = df['Wavelength(um)'].iloc[0]
    if first_wavelength > 10:
        print(f"检测到检测值 {first_wavelength} 应该是 nm 单位，正在自动转换为 um...")
        df['Wavelength(um)'] = df['Wavelength(um)'] / 1000.0
    else:
        print(f"检测到检测值 {first_wavelength} 已经是 um 单位，保持原样。")

    # 3. 导出为标准的无序号 CSV 文件
    df.to_csv(output_file, index=False)
    print(f"【转换成功】已生成标准格式 CSV: {output_file}")
    print(df.head())  # 打印前几行预览一下

except Exception as e:
    print(f"转换失败，错误信息: {e}")