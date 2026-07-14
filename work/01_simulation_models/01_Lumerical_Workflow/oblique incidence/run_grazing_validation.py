from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from grazing_config import CONFIG, DATA_DIR, ensure_output_dirs
from main_grazing_stackrt import run_stackrt
from simulate_grazing_triangulation import simulate_triangulation
from solve_grazing_asd import solve_asd


NEW_FILES = [
    "01_Lumerical_Workflow/oblique incidence/grazing_config.py",
    "01_Lumerical_Workflow/oblique incidence/main_grazing_stackrt.py",
    "01_Lumerical_Workflow/oblique incidence/solve_grazing_asd.py",
    "01_Lumerical_Workflow/oblique incidence/simulate_grazing_triangulation.py",
    "01_Lumerical_Workflow/oblique incidence/run_grazing_validation.py",
]

RUN_COMMANDS = [
    'python "01_Lumerical_Workflow/oblique incidence/run_grazing_validation.py"',
    'python "01_Lumerical_Workflow/oblique incidence/main_grazing_stackrt.py"',
    'python "01_Lumerical_Workflow/oblique incidence/solve_grazing_asd.py" --input "<grazing_stackrt_npz>"',
    'python "01_Lumerical_Workflow/oblique incidence/simulate_grazing_triangulation.py" --input "<grazing_asd_npz>"',
]


def _npz_text(value: Any) -> str:
    """读取 npz 中保存的 0 维字符串。"""
    return str(np.asarray(value).item())


def _load_npz(path: Path, keys: List[str] | None = None) -> Dict[str, Any]:
    """按需把 npz 内容复制到普通 dict，避免大数组被无谓重复载入。"""
    with np.load(path) as data:
        selected = data.files if keys is None else [key for key in keys if key in data.files]
        return {key: data[key].copy() for key in selected}


def _load_json_text(value: Any) -> Any:
    """解析 npz 中保存的 JSON 字符串。"""
    return json.loads(_npz_text(value))


def _top_recommendations(asd_data: Dict[str, Any], count: int = 6) -> List[Dict[str, Any]]:
    """读取 ASD 推荐表的前若干行。"""
    table = _load_json_text(asd_data["recommended_theta_table"])
    return table[:count]


def _best_by_multipass(tri_data: Dict[str, Any]) -> List[Tuple[int, float, float]]:
    """返回每个 N 的最佳 theta 和 total RMS。"""
    theta_axis = np.asarray(tri_data["theta_axis_deg"], dtype=float)
    multipass_list = np.asarray(tri_data["multipass_list"], dtype=int)
    total = np.asarray(tri_data["total_rms_nm"], dtype=float)
    rows: List[Tuple[int, float, float]] = []
    for n_idx, multipass_n in enumerate(multipass_list):
        theta_idx = int(np.nanargmin(total[:, n_idx]))
        rows.append((int(multipass_n), float(theta_axis[theta_idx]), float(total[theta_idx, n_idx])))
    return rows


def write_unavailable_summary(message: str) -> Tuple[Path, Path]:
    """在 lumapi 不可用或 StackRT 未执行时写入说明性 summary。"""
    ensure_output_dirs()
    summary_md = DATA_DIR / "summary.md"
    summary_csv = DATA_DIR / "summary.csv"
    summary_md.write_text(
        "# Grazing Validation Summary\n\n"
        "StackRT simulation was not executed.\n\n"
        f"Reason: {message}\n\n"
        "The new workflow code has been generated. Run the commands below after lumapi is available.\n\n"
        "```powershell\n"
        + "\n".join(RUN_COMMANDS)
        + "\n```\n",
        encoding="utf-8",
    )
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["status", "stackrt_not_executed"])
        writer.writerow(["reason", message])
    return summary_md, summary_csv


def generate_summary(stackrt_path: Path, asd_path: Path, tri_path: Path) -> Tuple[Path, Path]:
    """汇总完整仿真结果，生成 summary.md 和 summary.csv。"""
    ensure_output_dirs()
    stack_data = _load_npz(
        stackrt_path,
        [
            "layer_names",
            "thicknesses_nominal_m",
            "theta_axis_deg",
            "wavelengths_um",
            "model_source",
        ],
    )
    asd_data = _load_npz(
        asd_path,
        [
            "R_mean_p",
            "R_mean_s",
            "phase_source",
            "recommended_theta_table",
            "tmm_stackrt_diff_max_p",
            "tmm_stackrt_diff_max_s",
        ],
    )
    tri_data = _load_npz(
        tri_path,
        [
            "selected_pol",
            "best_theta_deg",
            "best_multipass_N",
            "total_rms_nm",
            "random_noise_std_nm",
            "asd_bias_std_nm",
            "optical_power_ratio",
            "geometric_gain",
            "theta_axis_deg",
            "multipass_list",
        ],
    )

    layer_names = np.asarray(stack_data["layer_names"], dtype=str)
    thickness_nm = np.asarray(stack_data["thicknesses_nominal_m"], dtype=float) * 1e9
    theta_axis = np.asarray(stack_data["theta_axis_deg"], dtype=float)
    wavelengths_um = np.asarray(stack_data["wavelengths_um"], dtype=float)
    R_mean_p = np.asarray(asd_data["R_mean_p"], dtype=float)
    R_mean_s = np.asarray(asd_data["R_mean_s"], dtype=float)
    phase_source = _npz_text(asd_data["phase_source"])
    top_rows = _top_recommendations(asd_data)
    selected_pol = _npz_text(tri_data["selected_pol"])
    best_theta = float(np.asarray(tri_data["best_theta_deg"]).item())
    best_n = int(np.asarray(tri_data["best_multipass_N"]).item())
    total = np.asarray(tri_data["total_rms_nm"], dtype=float)
    random_noise = np.asarray(tri_data["random_noise_std_nm"], dtype=float)
    asd_bias_std = np.asarray(tri_data["asd_bias_std_nm"], dtype=float)
    optical_power = np.asarray(tri_data["optical_power_ratio"], dtype=float)
    geometry = np.asarray(tri_data["geometric_gain"], dtype=float)
    theta_tri = np.asarray(tri_data["theta_axis_deg"], dtype=float)
    multipass = np.asarray(tri_data["multipass_list"], dtype=int)
    best_theta_idx = int(np.argmin(np.abs(theta_tri - best_theta)))
    best_n_idx = int(np.argmin(np.abs(multipass - best_n)))
    best_total = float(total[best_theta_idx, best_n_idx])
    n1_idx = int(np.argmin(np.abs(multipass - 1)))
    best_n1_idx = int(np.nanargmin(total[:, n1_idx]))
    best_n1_total = float(total[best_n1_idx, n1_idx])
    multipass_worth = best_n != 1 and best_total < 0.95 * best_n1_total
    asd_dominant = float(asd_bias_std[best_theta_idx, best_n_idx]) > float(random_noise[best_theta_idx, best_n_idx])
    tmm_diff = max(
        float(np.asarray(asd_data["tmm_stackrt_diff_max_p"]).item()),
        float(np.asarray(asd_data["tmm_stackrt_diff_max_s"]).item()),
    )

    layer_lines = "\n".join(
        f"- {name}: {thick:.3f} nm" for name, thick in zip(layer_names, thickness_nm)
    )
    top_lines = "\n".join(
        "- {pol}, theta={theta_deg:.2f} deg, score={score:.4g}, "
        "ASD MC std={ASD_mc_bias_std_nm:.4g} nm, R={R_mean:.4g}".format(**row)
        for row in top_rows
    )
    n_lines = "\n".join(
        f"- N={n}: best theta={theta:.2f} deg, total RMS={rms:.4g} nm"
        for n, theta, rms in _best_by_multipass(tri_data)
    )
    phase_note = (
        "lumapi stackrt returned complex reflection coefficients."
        if phase_source == "stackrt_complex_r"
        else "lumapi stackrt did not return complex phase; internal TMM was used for ASD phase."
    )
    tmm_note = (
        f"TMM/StackRT max reflectance difference = {tmm_diff:.4g}."
        if phase_source == "internal_tmm"
        else "TMM fallback was not required."
    )

    summary_md = DATA_DIR / "summary.md"
    summary_md.write_text(
        "# Grazing Incidence Validation Summary\n\n"
        "## Layer Stack\n"
        f"Model source: `{_npz_text(stack_data['model_source'])}`\n\n"
        f"{layer_lines}\n\n"
        "## Scan Range\n"
        f"- Incident angle from surface normal: {theta_axis[0]:.2f} to {theta_axis[-1]:.2f} deg\n"
        f"- Wavelength: {wavelengths_um[0]:.3f} to {wavelengths_um[-1]:.3f} um\n"
        f"- Phase source: {phase_source}. {phase_note}\n"
        f"- {tmm_note}\n\n"
        "## Polarization Comparison\n"
        f"- p mean reflectance: {np.nanmean(R_mean_p):.4g}\n"
        f"- s mean reflectance: {np.nanmean(R_mean_s):.4g}\n"
        f"- selected polarization for triangulation: `{selected_pol}`\n\n"
        "## ASD Recommendation Table\n"
        f"{top_lines}\n\n"
        "## Multipass Result\n"
        f"{n_lines}\n\n"
        "## Recommended Scheme\n"
        f"- Recommended theta: {best_theta:.2f} deg\n"
        f"- Recommended pol: {selected_pol}\n"
        f"- Recommended N: {best_n}\n"
        f"- Best total RMS: {best_total:.4g} nm\n"
        f"- Optical power ratio at best point: {optical_power[best_theta_idx, best_n_idx]:.4g}\n"
        f"- Geometric gain at best point: {geometry[best_theta_idx, best_n_idx]:.4g}\n\n"
        "## Conclusions\n"
        f"- Multipass improves random precision: {'yes' if best_n != 1 else 'no at this optimum'}.\n"
        f"- Multipass improves absolute accuracy enough to justify itself: {'yes' if multipass_worth else 'not clearly'}.\n"
        f"- Film-stack ASD is the dominant error at the best point: {'yes' if asd_dominant else 'no'}.\n"
        "- Next step: use Zemax to verify OAP/flat-mirror real optical-path angle tolerance.\n\n"
        "## Output Files\n"
        f"- StackRT npz: `{stackrt_path}`\n"
        f"- ASD npz: `{asd_path}`\n"
        f"- Triangulation npz: `{tri_path}`\n",
        encoding="utf-8",
    )

    summary_csv = DATA_DIR / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["stackrt_npz", str(stackrt_path)])
        writer.writerow(["asd_npz", str(asd_path)])
        writer.writerow(["triangulation_npz", str(tri_path)])
        writer.writerow(["theta_min_deg", f"{theta_axis[0]:.6g}"])
        writer.writerow(["theta_max_deg", f"{theta_axis[-1]:.6g}"])
        writer.writerow(["wavelength_min_um", f"{wavelengths_um[0]:.6g}"])
        writer.writerow(["wavelength_max_um", f"{wavelengths_um[-1]:.6g}"])
        writer.writerow(["phase_source", phase_source])
        writer.writerow(["selected_pol", selected_pol])
        writer.writerow(["best_theta_deg", f"{best_theta:.6g}"])
        writer.writerow(["best_multipass_N", str(best_n)])
        writer.writerow(["best_total_rms_nm", f"{best_total:.6g}"])
        writer.writerow(["tmm_stackrt_diff_max", f"{tmm_diff:.6g}"])
        writer.writerow(["multipass_worth", str(multipass_worth)])
        writer.writerow(["asd_dominant", str(asd_dominant)])

    return summary_md, summary_csv


def run_validation(args: argparse.Namespace) -> None:
    """按入口参数执行完整流程或从已有中间结果继续。"""
    ensure_output_dirs()
    try:
        stackrt_path = Path(args.stackrt_input) if args.stackrt_input else run_stackrt(CONFIG)
    except Exception as exc:
        summary_md, summary_csv = write_unavailable_summary(str(exc))
        print_required_footer(None, None, None, summary_md, summary_csv, lumapi_available=False)
        return

    asd_path = Path(args.asd_input) if args.asd_input else solve_asd(stackrt_path, CONFIG)
    tri_path = Path(args.triangulation_input) if args.triangulation_input else simulate_triangulation(asd_path, CONFIG)
    summary_md, summary_csv = generate_summary(stackrt_path, asd_path, tri_path)
    print_required_footer(stackrt_path, asd_path, tri_path, summary_md, summary_csv, lumapi_available=True)


def print_required_footer(
    stackrt_path: Path | None,
    asd_path: Path | None,
    tri_path: Path | None,
    summary_md: Path,
    summary_csv: Path,
    lumapi_available: bool,
) -> None:
    """按 prompt 要求在终端输出新增文件、运行命令和结果路径。"""
    print("\n新增文件列表:")
    for file_path in NEW_FILES:
        print(f"- {file_path}")
    print("\n运行命令:")
    for command in RUN_COMMANDS:
        print(f"- {command}")
    if not lumapi_available:
        print("\nlumapi 不可用或 StackRT 未执行：代码已经完成，但没有执行 stackrt。")
        print(f"summary.md: {summary_md}")
        print(f"summary.csv: {summary_csv}")
        return
    print("\nlumapi 可用，完整流程已执行。")
    print(f"StackRT npz: {stackrt_path}")
    print(f"ASD npz: {asd_path}")
    print(f"Triangulation npz: {tri_path}")
    print(f"summary.md: {summary_md}")
    print(f"summary.csv: {summary_csv}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run full grazing-incidence validation workflow.")
    parser.add_argument("--stackrt-input", type=Path, default=None, help="Reuse an existing grazing_stackrt_*.npz")
    parser.add_argument("--asd-input", type=Path, default=None, help="Reuse an existing grazing_asd_*.npz")
    parser.add_argument(
        "--triangulation-input",
        type=Path,
        default=None,
        help="Reuse an existing grazing_triangulation_*.npz",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    run_validation(parse_args())


if __name__ == "__main__":
    main()
