"""
导出 PCA 特征数据的前 100 行为单个 CSV 文件
"""

import numpy as np
import pandas as pd
from pathlib import Path


def export_pca_features_to_csv(npz_path: str, output_csv_path: str, max_rows: int = 100):
    """
    将 PCA 特征数据导出为单个 CSV 文件
    
    参数:
        npz_path: 包含 pca_scores 的 npz 文件路径
        output_csv_path: 输出 CSV 文件路径
        max_rows: 导出的最大行数
    """
    print(f"📂 加载数据: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    if 'pca_scores' not in data.files:
        raise ValueError(f"npz 文件中未找到 'pca_scores' 字段。可用字段: {data.files}")
    
    pca_scores = data['pca_scores']
    print(f"\n📊 PCA 特征信息:")
    print(f"  Shape: {pca_scores.shape}")
    print(f"  Dtype: {pca_scores.dtype}")
    
    if len(pca_scores.shape) != 2:
        raise ValueError(f"pca_scores 必须是二维数组，当前 shape={pca_scores.shape}")
    
    n_rows = min(max_rows, pca_scores.shape[0])
    features_data = pca_scores[:n_rows]
    
    col_names = [f"PC_{i+1}" for i in range(features_data.shape[1])]
    df = pd.DataFrame(features_data, columns=col_names)
    
    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ 已导出 PCA 特征到 CSV:")
    print(f"  文件路径: {csv_path}")
    print(f"  行数: {df.shape[0]}")
    print(f"  列数: {df.shape[1]} (主成分数量)")
    print(f"  列名示例: {col_names[:5]} ... {col_names[-5:]}")
    
    summary_path = csv_path.with_suffix('.summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("PCA 特征数据总结\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"源文件: {npz_path}\n")
        f.write(f"导出行数: {n_rows}\n")
        f.write(f"主成分数量: {features_data.shape[1]}\n\n")
        
        f.write("统计信息:\n")
        f.write("-" * 70 + "\n")
        for i in range(min(10, features_data.shape[1])):
            col = features_data[:, i]
            f.write(f"PC_{i+1}: Min={col.min():.6f}, Max={col.max():.6f}, ")
            f.write(f"Mean={col.mean():.6f}, Std={col.std():.6f}\n")
        
        if features_data.shape[1] > 10:
            f.write(f"... (共 {features_data.shape[1]} 个主成分)\n")
    
    print(f"\n📄 统计摘要已保存至: {summary_path}")


def main():
    npz_path = r"D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\01_Lumerical_Workflow\ML try\Residual MLP\dataset\pca_features\nn_cavity_pca_features_100_20260623_120625.npz"
    
    output_dir = Path(npz_path).parent
    output_csv_path = output_dir / "pca_scores_first_100.csv"
    
    print(f"📤 开始导出 PCA 特征...")
    export_pca_features_to_csv(npz_path, str(output_csv_path), max_rows=100)
    
    print("\n✨ 完成！")


if __name__ == '__main__':
    main()
