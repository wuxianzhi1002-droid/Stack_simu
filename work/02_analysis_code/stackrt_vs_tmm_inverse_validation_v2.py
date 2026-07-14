from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tmm_inverse_validation_robust import (
    OUTPUT_ROOT,
    NOMINAL_NM,
    ValidationConfig,
    case_layers,
    default_cases,
    draw_wavelength_drift,
    fit_spectrum_multistart,
    make_tmm_observation,
    param_from_um,
    perturb_fixed_layers,
    summarize_errors,
    tmm_reflectance,
    wavelength_axis,
    draw_truth_samples,
    case_definition,
)


LUMERICAL_PATH = r"D:\Program Files\Lumerical\v241\api\python"
LUMERICAL_BIN = r"D:\Program Files\Lumerical\v241\bin"
if os.path.exists(LUMERICAL_PATH) and LUMERICAL_PATH not in sys.path:
    sys.path.append(LUMERICAL_PATH)
if os.path.exists(LUMERICAL_BIN):
    os.environ["PATH"] += os.pathsep + LUMERICAL_BIN

try:
    import lumapi  # type: ignore
except ImportError:
    lumapi = None


def stackrt_material_n(
    name: str,
    wavelengths_um: np.ndarray,
    fdtd=None,
    freqs: np.ndarray | None = None,
) -> np.ndarray:
    w = np.asarray(wavelengths_um, dtype=np.float64)
    if name == "RefReflector":
        return np.full_like(w, 5.8284, dtype=np.complex128)
    if name == "Air":
        return np.full_like(w, 1.0, dtype=np.complex128)
    if name == "HSQ":
        return np.full_like(w, 1.41, dtype=np.complex128)
    if name == "PSS":
        return np.full_like(w, 1.50 + 0.05j, dtype=np.complex128)
    if name == "SOC":
        return (1.55 + 0.005 / (w**2)).astype(np.complex128)
    if name == "TiO2":
        return (2.4 + 0.02 / (w**2)).astype(np.complex128)
    if name == "Cu":
        # V2 diagnostic: force StackRT generation to use the same simplified Cu
        # constant as the inverse TMM model. This isolates whether the large
        # v1 StackRT-vs-TMM gap mainly came from Palik-Cu/model mismatch.
        return np.full_like(w, 1.1 + 2.5j, dtype=np.complex128)
    raise ValueError(f"Unknown material: {name}")


def stackrt_reflectance(
    fdtd,
    wavelengths_um: np.ndarray,
    layer_names: list[str],
    thicknesses_um: dict[str, float],
    angle_deg: float = 0.0,
    result_key: str = "Rp",
) -> np.ndarray:
    freqs = 3.0e8 / (wavelengths_um * 1.0e-6)
    n_matrix = np.vstack([stackrt_material_n(name, wavelengths_um, fdtd, freqs) for name in layer_names])
    thicknesses_m = np.array([thicknesses_um.get(name, 0.0) * 1.0e-6 for name in layer_names])
    res = fdtd.stackrt(n_matrix, thicknesses_m, freqs, float(angle_deg))
    return np.real(np.asarray(res[result_key]).flatten())


class StackRTObservationFactory:
    def __init__(self, allow_tmm_fallback: bool = False, angle_deg: float = 0.0):
        self.allow_tmm_fallback = allow_tmm_fallback
        self.angle_deg = angle_deg
        self.fdtd = None
        self.using_fallback = False

    def __enter__(self):
        if lumapi is None:
            if not self.allow_tmm_fallback:
                raise RuntimeError(
                    "lumapi is not available. Install/configure Lumerical Python API, "
                    "or pass --allow-tmm-fallback only for a smoke test."
                )
            self.using_fallback = True
            return self
        try:
            self.fdtd = lumapi.FDTD(hide=True)
        except Exception:
            if not self.allow_tmm_fallback:
                raise
            self.using_fallback = True
            self.fdtd = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fdtd is not None:
            self.fdtd.close()
            self.fdtd = None

    @property
    def scenario_name(self) -> str:
        return "stackrt_generator_tmm_fallback" if self.using_fallback else "stackrt_generator"

    def make_observation(
        self,
        wavelengths_recorded_um: np.ndarray,
        layers: list[str],
        truth_um: dict[str, float],
        config: ValidationConfig,
        rng: np.random.Generator,
        noise_sigma: float,
    ) -> tuple[np.ndarray, dict]:
        w_true, offset_nm, scale_ppm = draw_wavelength_drift(wavelengths_recorded_um, config, rng)
        if self.using_fallback:
            clean = tmm_reflectance(w_true, layers, truth_um)
        else:
            clean = stackrt_reflectance(self.fdtd, w_true, layers, truth_um, angle_deg=self.angle_deg)
        source_scale = 1.0 + rng.normal(0.0, config.source_scale_sigma)
        source_offset = rng.normal(0.0, config.source_offset_sigma)
        observed = source_scale * clean + source_offset + rng.normal(0.0, noise_sigma, size=clean.shape)
        return observed, {
            "wavelength_offset_nm": offset_nm,
            "wavelength_scale_ppm": scale_ppm,
            "source_scale": float(source_scale),
            "source_offset": float(source_offset),
        }


def make_nominal_tmm_observation(
    wavelengths_recorded_um: np.ndarray,
    layers: list[str],
    truth_um: dict[str, float],
    config: ValidationConfig,
    rng: np.random.Generator,
    noise_sigma: float,
) -> tuple[np.ndarray, dict]:
    clean_config = replace(
        config,
        material_real_sigma_fraction=0.0,
        material_imag_sigma_fraction=0.0,
        fixed_layer_sigma_nm=0.0,
    )
    return make_tmm_observation(wavelengths_recorded_um, layers, truth_um, clean_config, rng, noise_sigma)


def save_comparison_plot(output_dir: Path, summary: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for (scenario, case, param), part in summary.groupby(["scenario", "case", "param"]):
        part = part.sort_values("noise_sigma_reflectance")
        ax.plot(
            part["noise_sigma_reflectance"],
            part["MAE_nm"],
            marker="o",
            label=f"{scenario}:{case}:{param}",
        )
    ax.set_xlabel("Additive reflectance noise sigma")
    ax.set_ylabel("MAE (nm)")
    ax.set_title("TMM inverse fitting: TMM-generated vs StackRT-generated spectra")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    path = output_dir / "stackrt_vs_tmm_mae.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_scenario(
    config: ValidationConfig,
    output_dir: Path,
    cases: list[str],
    scenario: str,
    observation_factory,
    seed_offset: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed + seed_offset)
    wavelengths_um = wavelength_axis(config)
    rows = []
    trials_per_truth = sum(1 if noise == 0.0 else config.mc_trials_per_noise for noise in config.noise_sigmas_reflectance)
    total_spectra = len(cases) * config.truth_samples_per_case * trials_per_truth
    done_spectra = 0
    scenario_start = time.time()
    print(
        f"[{scenario}] start: cases={len(cases)}, truth_samples={config.truth_samples_per_case}, "
        f"noise_levels={len(config.noise_sigmas_reflectance)}, multistarts={config.multistarts}, "
        f"wavelength_points={len(wavelengths_um)}, spectra={total_spectra}",
        flush=True,
    )
    for case_idx, case in enumerate(cases, start=1):
        definition = case_definition(case)
        params = definition["fit_params"]
        layers = case_layers(case)
        truth_samples = draw_truth_samples(case, config, rng)
        print(
            f"[{scenario}] case {case_idx}/{len(cases)}: {case}, fit_params={params}",
            flush=True,
        )
        for truth_id, base_truth_um in enumerate(truth_samples):
            truth_um = perturb_fixed_layers(case, base_truth_um, replace(config, fixed_layer_sigma_nm=0.0), rng)
            for noise_sigma in config.noise_sigmas_reflectance:
                trials = 1 if noise_sigma == 0.0 else config.mc_trials_per_noise
                for trial in range(trials):
                    step_start = time.time()
                    done_spectra += 1
                    print(
                        f"[{scenario}] spectrum {done_spectra}/{total_spectra}: "
                        f"case={case}, truth={truth_id + 1}/{len(truth_samples)}, "
                        f"noise={noise_sigma:g}, trial={trial + 1}/{trials}",
                        flush=True,
                    )
                    observed, meta = observation_factory(wavelengths_um, layers, truth_um, config, rng, noise_sigma)
                    fit = fit_spectrum_multistart(wavelengths_um, observed, case, config, rng)
                    print(
                        f"[{scenario}] done {done_spectra}/{total_spectra}: "
                        f"cost={fit['cost']:.4e}, rmse_R={fit['rmse_reflectance']:.4e}, "
                        f"near_solutions={fit['near_solution_count']}, "
                        f"elapsed_step={time.time() - step_start:.1f}s, "
                        f"elapsed_total={time.time() - scenario_start:.1f}s",
                        flush=True,
                    )
                    true_values = np.array([param_from_um(name, truth_um[name]) for name in params])
                    for name, true_value, pred_value in zip(params, true_values, fit["x"]):
                        unit_error_nm = (pred_value - true_value) * 1000.0 if name == "Air" else pred_value - true_value
                        spread = fit["near_solution_spread"].get(name, 0.0)
                        spread_nm = spread * 1000.0 if name == "Air" else spread
                        rows.append(
                            {
                                "scenario": scenario,
                                "case": case,
                                "case_label": definition["label"],
                                "truth_id": truth_id,
                                "trial": trial,
                                "noise_sigma_reflectance": float(noise_sigma),
                                "param": name,
                                "true_value": float(true_value),
                                "pred_value": float(pred_value),
                                "error_nm": float(unit_error_nm),
                                "success": bool(fit["success"]),
                                "cost": float(fit["cost"]),
                                "nfev": int(fit["nfev"]),
                                "rmse_reflectance": float(fit["rmse_reflectance"]),
                                "affine_scale": float(fit["scale"]),
                                "affine_offset": float(fit["offset"]),
                                "near_solution_count": int(fit["near_solution_count"]),
                                "equivalent_spread_nm": float(spread_nm),
                                "wavelength_offset_nm": float(meta.get("wavelength_offset_nm", 0.0)),
                                "wavelength_scale_ppm": float(meta.get("wavelength_scale_ppm", 0.0)),
                                "source_scale": float(meta.get("source_scale", 1.0)),
                                "source_offset": float(meta.get("source_offset", 0.0)),
                            }
                        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, config: ValidationConfig, summary: pd.DataFrame, plot_path: str, stackrt_mode: str) -> str:
    lines = [
        "# StackRT vs TMM Inverse Validation",
        "",
        "## Scope",
        "",
        "- `tmm_generator_nominal`: spectra generated by the local TMM forward model, then fitted by TMM.",
        "- `stackrt_generator`: spectra generated by Lumerical `fdtd.stackrt(n_matrix, thicknesses, freqs, angle)`, then fitted by the same TMM inverse model.",
        "- V2 diagnostic change: StackRT generation uses fixed `Cu = 1.1 + 2.5j`, matching the inverse TMM model instead of Lumerical Palik Cu.",
        "- Both branches include the same recorded-wavelength drift, additive reflectance noise, and affine intensity fitting.",
        "- The StackRT matrix construction follows the layer/material convention in `work/01_simulation_models/01_Lumerical_Workflow/main.py` and `main_cavity.py`.",
        f"- StackRT execution mode: `{stackrt_mode}`.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "| scenario | case | noise | param | n | success | MAE_nm | P95Abs_nm | MaxAbs_nm | equiv_count | max_equiv_spread_nm |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["scenario", "case", "param", "noise_sigma_reflectance"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    str(row["case"]),
                    f"{row['noise_sigma_reflectance']:.4g}",
                    str(row["param"]),
                    f"{int(row['n'])}",
                    f"{row['success_rate']:.2f}",
                    f"{row['MAE_nm']:.4g}",
                    f"{row['P95Abs_nm']:.4g}",
                    f"{row['MaxAbs_nm']:.4g}",
                    f"{row['mean_near_solution_count']:.2f}",
                    f"{row['max_equivalent_spread_nm']:.4g}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Output files",
        "",
        f"- fit rows: `{output_dir / 'fit_results.csv'}`",
        f"- metrics: `{output_dir / 'metrics_summary.csv'}`",
        f"- plot: `{plot_path}`",
        "",
        "## Interpretation",
        "",
        "If StackRT-generated spectra fit much worse than TMM-generated spectra under the same inverse code, the gap is a forward-model mismatch rather than optimizer noise alone.",
    ]
    path = output_dir / "summary_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def run_validation(
    config: ValidationConfig,
    cases: list[str],
    allow_tmm_fallback: bool = False,
    angle_deg: float = 0.0,
) -> Path:
    output_dir = OUTPUT_ROOT / f"stackrt_vs_tmm_inverse_validation_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"[run] output_dir={output_dir}", flush=True)
    print(f"[run] cases={cases}", flush=True)
    print(f"[run] config={json.dumps(asdict(config), ensure_ascii=False)}", flush=True)

    tmm_df = run_scenario(
        config,
        output_dir,
        cases,
        "tmm_generator_nominal",
        make_nominal_tmm_observation,
        seed_offset=0,
    )

    with StackRTObservationFactory(allow_tmm_fallback=allow_tmm_fallback, angle_deg=angle_deg) as stackrt_factory:
        print(f"[run] StackRT mode={stackrt_factory.scenario_name}", flush=True)
        stackrt_df = run_scenario(
            config,
            output_dir,
            cases,
            stackrt_factory.scenario_name,
            stackrt_factory.make_observation,
            seed_offset=0,
        )
        stackrt_mode = stackrt_factory.scenario_name

    fit_df = pd.concat([tmm_df, stackrt_df], ignore_index=True)
    summary = summarize_errors(fit_df)
    fit_csv = output_dir / "fit_results.csv"
    summary_csv = output_dir / "metrics_summary.csv"
    print("[run] writing CSV/report/plot outputs", flush=True)
    fit_df.to_csv(fit_csv, index=False, encoding="utf-8-sig", float_format="%.10g")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig", float_format="%.10g")
    plot_path = save_comparison_plot(output_dir, summary)
    (output_dir / "config_and_outputs.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "cases": cases,
                "nominal_nm": NOMINAL_NM,
                "diagnostic_change": "StackRT generation uses fixed Cu=1.1+2.5j to match inverse TMM.",
                "fit_results_csv": str(fit_csv),
                "metrics_summary_csv": str(summary_csv),
                "plot_path": plot_path,
                "stackrt_mode": stackrt_mode,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report_path = write_report(output_dir, config, summary, plot_path, stackrt_mode)
    print(f"OUTPUT_DIR={output_dir}")
    print(f"SUMMARY={summary_csv}")
    print(f"REPORT={report_path}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2 diagnostic: compare TMM inverse fitting after forcing StackRT Cu to match TMM Cu."
    )
    parser.add_argument("--cases", nargs="+", default=["B1_film_tio2", "B2_film_soc_tio2", "B3_film_all"])
    parser.add_argument("--include-cavity", action="store_true")
    parser.add_argument("--mc-trials", type=int, default=2)
    parser.add_argument("--truth-samples", type=int, default=5)
    parser.add_argument("--multistarts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260707)
    parser.add_argument("--spectral-resolution-nm", type=float, default=0.02)
    parser.add_argument("--noise-sigmas", type=float, nargs="+", default=[0.0, 0.005])
    parser.add_argument("--wavelength-offset-sigma-nm", type=float, default=0.03)
    parser.add_argument("--wavelength-scale-sigma-ppm", type=float, default=80.0)
    parser.add_argument("--allow-tmm-fallback", action="store_true")
    parser.add_argument("--angle-deg", type=float, default=0.0)
    parser.add_argument("--no-affine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = list(args.cases)
    if args.include_cavity and "A0_cavity_air" not in cases:
        cases = ["A0_cavity_air"] + cases
    unknown = sorted(set(cases) - set(default_cases()))
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")
    config = ValidationConfig(
        spectral_resolution_nm=float(args.spectral_resolution_nm),
        random_seed=int(args.random_seed),
        noise_sigmas_reflectance=tuple(float(v) for v in args.noise_sigmas),
        mc_trials_per_noise=int(args.mc_trials),
        truth_samples_per_case=int(args.truth_samples),
        multistarts=int(args.multistarts),
        max_nfev=int(args.max_nfev),
        fit_affine_intensity=not bool(args.no_affine),
        wavelength_offset_sigma_nm=float(args.wavelength_offset_sigma_nm),
        wavelength_scale_sigma_ppm=float(args.wavelength_scale_sigma_ppm),
        material_real_sigma_fraction=0.0,
        material_imag_sigma_fraction=0.0,
        fixed_layer_sigma_nm=0.0,
    )
    run_validation(config, cases, allow_tmm_fallback=bool(args.allow_tmm_fallback), angle_deg=float(args.angle_deg))


if __name__ == "__main__":
    main()



