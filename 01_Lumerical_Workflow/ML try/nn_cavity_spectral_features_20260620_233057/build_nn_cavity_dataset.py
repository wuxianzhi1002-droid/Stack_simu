"""正式数据集生成入口：当前默认执行版本 2。

旧版 scalar-only 实现已保存在 build_nn_cavity_dataset_v1.py。
版本 2 完整实现在 build_nn_cavity_dataset_v2.py。
"""

from build_nn_cavity_dataset_v2 import main


if __name__ == "__main__":
    main()
