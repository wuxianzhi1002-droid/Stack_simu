"""
从 metrics.json 读取训练结果并绘制方法对比柱状图
无需重新训练模型，直接可视化已有结果
展示每个方法的 Cavity MAE 和 Film Mean MAE 对比
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
    展示每个方法的 Cavity MAE 和 Film Mean MAE
    
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
    
    # 从 aggregate 中提取 cavity 和 film 的 MAE 指标
    cavity_mae_values = [methods[method]['aggregate']['cavity_MAE_nm_equiv'] for method in method_names]
    film_mae_values = [methods[method]['aggregate']['film_mean_MAE_nm'] for method in method_names]
    
    bars1 = ax.bar(x - width/2, cavity_mae_values, width, 
                   label='Cavity MAE', color='#4C72B0', alpha=0.85, 
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, film_mae_values, width, 
                   label='Film Mean MAE', color='#DD4444', alpha=0.85, 
                   edgecolor='black', linewidth=0.5)
    
    # 在柱子顶部添加数值标注（遵循柱状图数值标注规范）
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Error (nm)', fontsize=14, fontweight='bold')
    ax.set_title('Method Comparison - Cavity vs Film MAE (Test Set)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=13)
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 设置 y 轴范围，确保数值标注有足够空间
    max_value = max(max(cavity_mae_values), max(film_mae_values))
    ax.set_ylim(0, max_value * 1.15)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'method_comparison_cavity_vs_film.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存至: {output_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("📊 测试集性能对比详情")
    print("="*70)
    
    for method_name in method_names:
        agg = methods[method_name]['aggregate']
        print(f"\n{method_name}:")
        print(f"  Cavity MAE:      {agg['cavity_MAE_nm_equiv']:.2f} nm")
        print(f"  Film Mean MAE:   {agg['film_mean_MAE_nm']:.2f} nm")
        print(f"  Film Max MAE:    {agg['film_max_MAE_nm']:.2f} nm")
        print(f"  Cavity RMSE:     {agg['cavity_RMSE_nm_equiv']:.2f} nm")
        print(f"  Film Mean RMSE:  {agg['film_mean_RMSE_nm']:.2f} nm")
        print(f"  Mean R²:         {agg['mean_R2']:.6f}")
        
        # 计算 cavity 和 film 的差异
        diff = agg['cavity_MAE_nm_equiv'] - agg['film_mean_MAE_nm']
        diff_pct = (diff / agg['film_mean_MAE_nm']) * 100
        print(f"  → Cavity 比 Film 高: {diff:.2f} nm ({diff_pct:+.2f}%)")
    
    # 计算两种方法之间的改善
    base_agg = methods['base scalar']['aggregate']
    more_agg = methods['more feature']['aggregate']
    
    cavity_improvement = base_agg['cavity_MAE_nm_equiv'] - more_agg['cavity_MAE_nm_equiv']
    cavity_improvement_pct = (cavity_improvement / base_agg['cavity_MAE_nm_equiv']) * 100
    
    film_improvement = base_agg['film_mean_MAE_nm'] - more_agg['film_mean_MAE_nm']
    film_improvement_pct = (film_improvement / base_agg['film_mean_MAE_nm']) * 100
    
    print(f"\n{'='*70}")
    print("📈 More Feature 相比 Base Scalar 的改善:")
    print(f"  Cavity MAE 改善:  {cavity_improvement:.2f} nm ({cavity_improvement_pct:+.2f}%)")
    print(f"  Film Mean MAE 改善: {film_improvement:.2f} nm ({film_improvement_pct:+.2f}%)")
    print("="*70)


def main():
    metrics_path = r"D:\激光干涉仪\simulation\Lumerical_simulation\STACK_simu\01_Lumerical_Workflow\ML try\Residual MLP\residual_mlp_simple_multioutput_20260626_154611\metrics.json"

    output_dir = Path(metrics_path).parent
    
    print(f"📂 加载数据: {metrics_path}")
    metrics = load_metrics(metrics_path)
    
    print(f"📈 绘制 Cavity vs Film MAE 对比柱状图...")
    plot_method_comparison(metrics, str(output_dir))
    
    print("\n✨ 完成！")


if __name__ == '__main__':
    main()
