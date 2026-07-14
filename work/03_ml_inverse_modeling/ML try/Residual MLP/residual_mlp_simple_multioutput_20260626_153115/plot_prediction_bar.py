"""
从 metrics.json 读取训练结果并绘制方法对比柱状图
无需重新训练模型,直接可视化已有结果
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics(json_path: str) -> dict:
    """加载 metrics.json 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_method_comparison(metrics: dict, output_dir: str):
    """
    绘制两种方法的性能对比柱状图
    
    参数:
        metrics: 从 metrics.json 加载的字典
        output_dir: 输出图片的目录
    """
    results = metrics['results']
    
    methods = {
        'base scalar': results['base_scalar']['metrics']['test'],
        'more feature': results['more_feature']['metrics']['test']
    }
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    method_names = list(methods.keys())
    x = np.arange(len(method_names))
    width = 0.35
    
    # 从 aggregate 中提取 cavity 指标
    mae_values = [methods[method]['aggregate']['cavity_MAE_nm_equiv'] for method in method_names]
    rmse_values = [methods[method]['aggregate']['cavity_RMSE_nm_equiv'] for method in method_names]
    
    bars1 = ax.bar(x - width/2, mae_values, width, label='MAE', color='#4C72B0', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, rmse_values, width, label='RMSE', color='#DD4444', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Cavity Error (nm)', fontsize=14, fontweight='bold')
    ax.set_title('Method Comparison - Test Set Performance', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=13)
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(rmse_values) * 1.1)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'method_comparison_bar_from_json.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存至: {output_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("📊 测试集性能对比详情")
    print("="*70)
    
    for method_name in method_names:
        print(f"\n{method_name}:")
        agg = methods[method_name]['aggregate']
        print(f"  Cavity MAE:  {agg['cavity_MAE_nm_equiv']:.2f} nm")
        print(f"  Cavity RMSE: {agg['cavity_RMSE_nm_equiv']:.2f} nm")
        print(f"  Film Mean MAE:  {agg['film_mean_MAE_nm']:.2f} nm")
        print(f"  Film Mean RMSE: {agg['film_mean_RMSE_nm']:.2f} nm")
        print(f"  Mean R²: {agg['mean_R2']:.6f}")
    
    improvement_mae = methods['base scalar']['aggregate']['cavity_MAE_nm_equiv'] - methods['more feature']['aggregate']['cavity_MAE_nm_equiv']
    improvement_rmse = methods['base scalar']['aggregate']['cavity_RMSE_nm_equiv'] - methods['more feature']['aggregate']['cavity_RMSE_nm_equiv']
    improvement_mae_pct = (improvement_mae / methods['base scalar']['aggregate']['cavity_MAE_nm_equiv']) * 100
    improvement_rmse_pct = (improvement_rmse / methods['base scalar']['aggregate']['cavity_RMSE_nm_equiv']) * 100
    
    print(f"\nMore Feature 相比 Base Scalar:")
    print(f"  Cavity MAE 改善:  {improvement_mae:.2f} nm ({improvement_mae_pct:+.2f}%)")
    print(f"  Cavity RMSE 改善: {improvement_rmse:.2f} nm ({improvement_rmse_pct:+.2f}%)")
    
    film_improvement_mae = methods['base scalar']['aggregate']['film_mean_MAE_nm'] - methods['more feature']['aggregate']['film_mean_MAE_nm']
    film_improvement_rmse = methods['base scalar']['aggregate']['film_mean_RMSE_nm'] - methods['more feature']['aggregate']['film_mean_RMSE_nm']
    film_improvement_mae_pct = (film_improvement_mae / methods['base scalar']['aggregate']['film_mean_MAE_nm']) * 100
    film_improvement_rmse_pct = (film_improvement_rmse / methods['base scalar']['aggregate']['film_mean_RMSE_nm']) * 100
    
    print(f"  Film Mean MAE 改善:  {film_improvement_mae:.2f} nm ({film_improvement_mae_pct:+.2f}%)")
    print(f"  Film Mean RMSE 改善: {film_improvement_rmse:.2f} nm ({film_improvement_rmse_pct:+.2f}%)")
    
    print("\n" + "="*70)


def main():
    metrics_path = r"D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\01_Lumerical_Workflow\ML try\Residual MLP\residual_mlp_simple_multioutput_20260626_153115\metrics.json"
    
    output_dir = Path(metrics_path).parent
    
    print(f"📂 加载数据: {metrics_path}")
    metrics = load_metrics(metrics_path)
    
    print(f"📈 绘制对比柱状图...")
    plot_method_comparison(metrics, str(output_dir))
    
    print("\n✨ 完成！")


if __name__ == '__main__':
    main()
