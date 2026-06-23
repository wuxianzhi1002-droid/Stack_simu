"""版本 2 数据集续跑入口。

示例：
python resume_nn_cavity_dataset_v2.py "nn_cavity_spectral_features_20260620_120000"

续跑时 build_nn_cavity_dataset_v2.py 会强制读取 run directory 中的
00_config.json，不会采用新的命令行光谱参数，从而保证 checkpoint 一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import build_nn_cavity_dataset_v2 as builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume a version-2 spectral-feature dataset run.")
    parser.add_argument("run_dir", type=Path, help="已有的 nn_cavity_spectral_features_* 目录。")
    parser.add_argument(
        "--skip-final-merge",
        action="store_true",
        help="续跑完 checkpoint 后暂不合并最终 NPZ/CSV。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forwarded = [
        str(Path(builder.__file__).resolve()),
        "--resume-run-dir",
        str(args.run_dir.resolve()),
        "--skip-final-merge",
        "true" if args.skip_final_merge else "false",
    ]
    sys.argv = forwarded
    builder.main()


if __name__ == "__main__":
    main()
