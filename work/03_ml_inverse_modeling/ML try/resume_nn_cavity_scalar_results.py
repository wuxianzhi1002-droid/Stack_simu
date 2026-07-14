import importlib.util
import json
import time
import traceback
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR / "nn_cavity_scalar_results_20260617_085429"


def load_builder_module():
    # 该脚本只负责旧版 scalar-only 数据集续跑，固定加载版本 1，
    # 避免版本 2 的新膜厚网格和新 checkpoint 字段改变旧 run 的可复现性。
    builder_path = SCRIPT_DIR / "build_nn_cavity_dataset_v1.py"
    spec = importlib.util.spec_from_file_location("build_nn_cavity_dataset", builder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(builder):
    config_path = RUN_DIR / "00_config.json"
    with config_path.open("r", encoding="utf-8") as file:
        config_json = json.load(file)

    field_names = {field.name for field in fields(builder.Config)}
    kwargs = {key: value for key, value in config_json.items() if key in field_names}
    config = builder.Config(**kwargs)
    if config.add_noise:
        raise RuntimeError("Resume script does not support add_noise=True because prior noise draws are not replayed.")
    return config


def dump_json(path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def latest_completed_process_id():
    checkpoint_ids = []
    for path in RUN_DIR.glob("checkpoint_process_*.npz"):
        try:
            checkpoint_ids.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    return max(checkpoint_ids) if checkpoint_ids else -1


def build_process_table(builder, config):
    nominal_stacks_nm = builder.build_nominal_stacks_nm(config)
    process_ids_all = np.arange(len(nominal_stacks_nm) * config.num_process_per_nominal, dtype=int)
    train_process_ids, val_process_ids, test_process_ids = builder.split_by_process(
        process_ids_all,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed,
    )
    split_lookup = builder.process_split_lookup(
        process_ids_all,
        train_process_ids,
        val_process_ids,
        test_process_ids,
    )

    rng = np.random.default_rng(config.random_seed)
    process_rows = []
    process_id = 0
    for nominal_stack_id, nominal_stack_nm in enumerate(nominal_stacks_nm):
        film_nominal_nm = builder.nominal_film_array_nm(nominal_stack_nm)
        for process_idx in range(config.num_process_per_nominal):
            if process_idx == 0:
                film_delta_nm = np.zeros(len(builder.FILM_LAYER_NAMES), dtype=np.float32)
            else:
                film_delta_nm = builder.sample_film_delta_nm(film_nominal_nm, config, rng)
            process_rows.append(
                {
                    "process_id": process_id,
                    "nominal_stack_id": nominal_stack_id,
                    "nominal_stack_nm": nominal_stack_nm,
                    "film_nominal_nm": film_nominal_nm.copy(),
                    "film_delta_nm": film_delta_nm.copy(),
                    "film_true_nm": film_nominal_nm + film_delta_nm,
                    "split_id": split_lookup[process_id],
                }
            )
            process_id += 1

    return process_rows, train_process_ids, val_process_ids, test_process_ids


def save_process_checkpoint(builder, process_id, rows):
    checkpoint_path = RUN_DIR / f"checkpoint_process_{process_id:04d}.npz"
    np.savez_compressed(
        checkpoint_path,
        sample_id=np.asarray([row["sample_id"] for row in rows], dtype=np.int64),
        process_id=np.asarray([row["process_id"] for row in rows], dtype=np.int32),
        nominal_stack_id=np.asarray([row["nominal_stack_id"] for row in rows], dtype=np.int16),
        split_id=np.asarray([row["split_id"] for row in rows], dtype=np.int8),
        split_names=builder.SPLIT_NAMES,
        cavity_true_um=np.asarray([row["cavity_true_um"] for row in rows], dtype=np.float64),
        L_fft_um=np.asarray([row["L_fft_um"] for row in rows], dtype=np.float64),
        delta_L_um=np.asarray([row["delta_L_um"] for row in rows], dtype=np.float64),
        delta_L_nm=np.asarray([row["delta_L_nm"] for row in rows], dtype=np.float64),
        H_peak=np.asarray([row["H_peak"] for row in rows], dtype=np.float32),
        peak_count=np.asarray([row["peak_count"] for row in rows], dtype=np.int16),
        film_nominal_nm=np.asarray([row["film_nominal_nm"] for row in rows], dtype=np.float32),
        film_delta_nm=np.asarray([row["film_delta_nm"] for row in rows], dtype=np.float32),
        film_true_nm=np.asarray([row["film_true_nm"] for row in rows], dtype=np.float32),
        valid_mask=np.asarray([row["valid_mask"] for row in rows], dtype=bool),
        layer_names=np.asarray(builder.FILM_LAYER_NAMES, dtype=str),
        spectra_saved=np.array(False, dtype=bool),
    )
    return checkpoint_path


def update_manifest(next_process_id, total_processes, valid_total, failed_sim_total, failed_fft_total, resume_log_path):
    manifest_path = RUN_DIR / "resume_manifest.json"
    manifest = {
        "status": "running" if next_process_id < total_processes else "completed_checkpoints",
        "run_dir": str(RUN_DIR),
        "last_completed_process_id": int(next_process_id - 1),
        "next_process_id": int(next_process_id),
        "completed_checkpoint_count": int(next_process_id),
        "completed_sample_count_estimate": int(next_process_id * 1000),
        "valid_sample_count_resume_session": int(valid_total),
        "failed_simulation_resume_session": int(failed_sim_total),
        "failed_fft_resume_session": int(failed_fft_total),
        "remaining_process_count": int(total_processes - next_process_id),
        "resume_log": resume_log_path.name,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    dump_json(manifest_path, manifest)


def main():
    builder = load_builder_module()
    config = load_config(builder)
    nominal_stacks_nm = builder.build_nominal_stacks_nm(config)
    cavity_axis_um = builder.make_cavity_axis_um(config)
    process_rows, _, _, _ = build_process_table(builder, config)
    total_processes = len(process_rows)
    start_process_id = latest_completed_process_id() + 1
    resume_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_log_path = RUN_DIR / f"resume_failed_cases_{resume_timestamp}.json"
    failed_cases = []
    failed_fft_cases = []
    valid_total = 0

    print("=== Resume Scalar-Only NN Cavity Dataset ===")
    print(f"[Resume] Run dir: {RUN_DIR}")
    print(f"[Resume] Existing last completed process_id: {start_process_id - 1}")
    print(f"[Resume] Starting from process_id: {start_process_id}")
    print(f"[Resume] Total processes: {total_processes}")
    print(f"[Resume] Cavity points/process: {len(cavity_axis_um)}")

    if start_process_id >= total_processes:
        print("[Resume] All process checkpoints already exist.")
        return

    simulator = builder.StackRTSimulator(config)
    fft_config = builder.fft_config_dict(config)
    start_time = time.time()
    simulator.open()
    try:
        for process_id in range(start_process_id, total_processes):
            row = process_rows[process_id]
            checkpoint_path = RUN_DIR / f"checkpoint_process_{process_id:04d}.npz"
            if checkpoint_path.exists():
                print(f"[Resume] Skip existing checkpoint: {checkpoint_path.name}")
                continue

            nominal_stack_nm = row["nominal_stack_nm"]
            film_nominal_nm = row["film_nominal_nm"]
            film_delta_nm = row["film_delta_nm"]
            film_true_nm = row["film_true_nm"]
            split_id = row["split_id"]
            sample_rows = []

            print(
                "[Resume] Process "
                f"{process_id + 1}/{total_processes} "
                f"(process_id={process_id}, nominal={nominal_stack_nm['name']}, "
                f"split={builder.SPLIT_NAMES[split_id]}, "
                f"delta_nm={np.round(film_delta_nm, 4).tolist()})"
            )

            for cavity_idx, cavity_um in enumerate(cavity_axis_um):
                sample_id = process_id * len(cavity_axis_um) + cavity_idx
                try:
                    layers = builder.build_layers_from_nominal_stack(
                        nominal_stack_nm,
                        cavity_um,
                        film_delta_nm,
                    )
                    spectrum = simulator.simulate_spectrum(layers)
                    L_fft_um, H_peak, peak_count = builder.solve_single_fft(
                        simulator.wavelengths_um,
                        spectrum,
                        fft_config,
                    )
                    valid_mask = bool(np.isfinite(L_fft_um) and np.isfinite(H_peak))
                    if not valid_mask:
                        failed_fft_cases.append(
                            {
                                "sample_id": int(sample_id),
                                "process_id": int(process_id),
                                "cavity_idx": int(cavity_idx),
                                "cavity_true_um": float(cavity_um),
                                "reason": "no FFT peak detected",
                            }
                        )

                    sample_rows.append(
                        {
                            "sample_id": sample_id,
                            "process_id": process_id,
                            "nominal_stack_id": row["nominal_stack_id"],
                            "split_id": split_id,
                            "cavity_true_um": float(cavity_um),
                            "L_fft_um": L_fft_um,
                            "delta_L_um": float(cavity_um - L_fft_um),
                            "delta_L_nm": float((cavity_um - L_fft_um) * 1000.0),
                            "H_peak": H_peak,
                            "peak_count": peak_count,
                            "film_nominal_nm": film_nominal_nm,
                            "film_delta_nm": film_delta_nm,
                            "film_true_nm": film_true_nm,
                            "valid_mask": valid_mask,
                        }
                    )
                except Exception as exc:
                    failed_cases.append(
                        {
                            "sample_id": int(sample_id),
                            "process_id": int(process_id),
                            "cavity_idx": int(cavity_idx),
                            "cavity_true_um": float(cavity_um),
                            "nominal_stack_name": nominal_stack_nm["name"],
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

            if sample_rows:
                saved_path = save_process_checkpoint(builder, process_id, sample_rows)
                valid_count = int(np.count_nonzero([sample["valid_mask"] for sample in sample_rows]))
                valid_total += valid_count
                elapsed = time.time() - start_time
                print(
                    f"[Resume] Saved {saved_path.name}; "
                    f"samples={len(sample_rows)}, valid={valid_count}, "
                    f"resume_failed_sim={len(failed_cases)}, resume_failed_fft={len(failed_fft_cases)}, "
                    f"elapsed={elapsed:.1f}s"
                )

            dump_json(
                resume_log_path,
                {
                    "failed_cases": failed_cases,
                    "failed_fft_cases": failed_fft_cases,
                },
            )
            update_manifest(
                process_id + 1,
                total_processes,
                valid_total,
                len(failed_cases),
                len(failed_fft_cases),
                resume_log_path,
            )
    finally:
        simulator.close()

    print("[Resume] Done.")


if __name__ == "__main__":
    main()
