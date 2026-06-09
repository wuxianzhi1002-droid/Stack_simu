import json
import os
from datetime import datetime

import numpy as np
from scipy.signal import find_peaks


CONFIG = {
    "FFT_PEAK_HEIGHT_RATIO": 0.2,
    "FFT_IGNORE_DC_BINS": 50,
    "FFT_PEAK_DISTANCE_BINS": 100,
    "ZERO_PAD_FACTOR": 8,
    "SAVE_FFT_MATRIX": False,
}


class FFTSolver:
    @staticmethod
    def solve(wavelengths_um, intensities, config):
        wavelengths_um = np.asarray(wavelengths_um, dtype=float).reshape(-1)
        intensities = np.asarray(intensities, dtype=float).reshape(-1)

        if wavelengths_um.size != intensities.size:
            raise ValueError(
                f"wavelengths and intensities must have the same length, got "
                f"{wavelengths_um.size} and {intensities.size}."
            )

        finite_mask = np.isfinite(wavelengths_um) & np.isfinite(intensities)
        wavelengths_um = wavelengths_um[finite_mask]
        intensities = intensities[finite_mask]

        if wavelengths_um.size < 4:
            raise ValueError("At least 4 finite points are required for FFT solving.")

        sort_idx = np.argsort(wavelengths_um)
        wavelengths_um = wavelengths_um[sort_idx]
        intensities = intensities[sort_idx]

        k_raw = 2 * np.pi / wavelengths_um
        k_linear = np.linspace(k_raw.min(), k_raw.max(), len(k_raw))

        if k_raw[0] > k_raw[-1]:
            i_linear = np.interp(k_linear, k_raw[::-1], intensities[::-1])
        else:
            i_linear = np.interp(k_linear, k_raw, intensities)

        i_detrend = i_linear - np.mean(i_linear)
        i_windowed = i_detrend * np.hanning(len(i_detrend))

        n_fft = len(i_windowed) * int(config["ZERO_PAD_FACTOR"])
        fft_data = np.abs(np.fft.rfft(i_windowed, n=n_fft))

        dk = abs(k_linear[1] - k_linear[0])
        max_range_um = np.pi / dk
        distance_axis_um = np.linspace(0, max_range_um / 2, len(fft_data))

        ignore = min(int(config["FFT_IGNORE_DC_BINS"]), max(0, len(fft_data) - 1))
        search = fft_data[ignore:]
        if search.size == 0 or np.max(search) <= 0:
            peaks = np.array([], dtype=int)
        else:
            peaks, _ = find_peaks(
                search,
                height=np.max(search) * float(config["FFT_PEAK_HEIGHT_RATIO"]),
                distance=int(config["FFT_PEAK_DISTANCE_BINS"]),
            )
            peaks = peaks + ignore

        return {
            "distance_axis_um": distance_axis_um,
            "fft_data": fft_data,
            "peaks_idx": peaks,
            "peak_distances_um": distance_axis_um[peaks],
            "peak_heights": fft_data[peaks],
            "max_range_um": max_range_um / 2,
        }


class NPZSpectrumLoader:
    AXIS_PRIORITY = (
        ("angle_axis", "angle_deg"),
        ("theta_axis", "angle_deg"),
        ("cavity_axis_um", "cavity_um"),
        ("cavity_axis_m", "cavity_m"),
        ("L_t", "cavity_um"),
        ("t_axis", "time_or_scan_axis"),
    )

    @staticmethod
    def load(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        keys = set(data.files)

        if "wavelengths" not in keys:
            raise KeyError(f"{npz_path} does not contain required key 'wavelengths'.")

        spectra_key = NPZSpectrumLoader._find_spectra_key(keys)
        wavelengths = np.asarray(data["wavelengths"], dtype=float).reshape(-1)
        spectra = np.asarray(data[spectra_key], dtype=float)

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        elif spectra.ndim != 2:
            raise ValueError(f"Expected spectra to be 1D or 2D, got shape {spectra.shape}.")

        spectra, wavelengths = NPZSpectrumLoader._orient_spectra(spectra, wavelengths)
        scan_axis, scan_axis_name = NPZSpectrumLoader._find_scan_axis(data, spectra.shape[0])

        return {
            "wavelengths": wavelengths,
            "spectra": spectra,
            "spectra_key": spectra_key,
            "scan_axis": scan_axis,
            "scan_axis_name": scan_axis_name,
            "source_keys": np.array(data.files),
        }

    @staticmethod
    def _find_spectra_key(keys):
        for key in ("spectra", "R", "Rp", "Rs", "intensities"):
            if key in keys:
                return key
        raise KeyError("Could not find a spectra key. Expected spectra, R, Rp, Rs, or intensities.")

    @staticmethod
    def _orient_spectra(spectra, wavelengths):
        if spectra.shape[1] == wavelengths.size:
            return spectra, wavelengths
        if spectra.shape[0] == wavelengths.size:
            return spectra.T, wavelengths
        raise ValueError(
            f"One spectra dimension must match wavelengths length {wavelengths.size}, got {spectra.shape}."
        )

    @staticmethod
    def _find_scan_axis(data, n_spectra):
        for key, axis_name in NPZSpectrumLoader.AXIS_PRIORITY:
            if key in data.files:
                axis = np.asarray(data[key]).reshape(-1)
                if axis.size == n_spectra:
                    return axis.astype(float, copy=False), axis_name

        return np.arange(n_spectra, dtype=float), "spectrum_index"


def normalize_spectrum_selection(selection, n_spectra):
    if selection == "all":
        selected_idx = np.arange(n_spectra, dtype=int)
    elif isinstance(selection, int):
        selected_idx = np.array([selection], dtype=int)
    elif isinstance(selection, slice):
        selected_idx = np.arange(n_spectra, dtype=int)[selection]
    else:
        selected_idx = np.asarray(selection, dtype=int).reshape(-1)

    if selected_idx.size == 0:
        raise ValueError("spectrum_selection selected no spectra.")

    selected_idx = np.where(selected_idx < 0, selected_idx + n_spectra, selected_idx)
    invalid = selected_idx[(selected_idx < 0) | (selected_idx >= n_spectra)]
    if invalid.size > 0:
        raise IndexError(f"spectrum_selection contains invalid indices: {invalid.tolist()}.")

    return selected_idx


class BatchFFTSolver:
    @staticmethod
    def solve_npz(npz_path, output_path=None, config=None, spectrum_selection="all"):
        loaded = NPZSpectrumLoader.load(npz_path)
        selected_idx = normalize_spectrum_selection(spectrum_selection, loaded["spectra"].shape[0])

        loaded["spectra"] = loaded["spectra"][selected_idx, :]
        loaded["scan_axis"] = loaded["scan_axis"][selected_idx]

        return BatchFFTSolver.solve_loaded_npz(
            loaded=loaded,
            npz_path=npz_path,
            output_path=output_path,
            config=config,
            selected_idx=selected_idx,
        )

    @staticmethod
    def solve_loaded_npz(loaded, npz_path, output_path=None, config=None, selected_idx=None):
        config = dict(CONFIG if config is None else config)

        wavelengths = loaded["wavelengths"]
        spectra = loaded["spectra"]
        scan_axis = loaded["scan_axis"]
        n_spectra = spectra.shape[0]

        if selected_idx is None:
            selected_idx = np.arange(n_spectra, dtype=int)

        first_peak_distance_um = np.full(n_spectra, np.nan)
        first_peak_height = np.full(n_spectra, np.nan)
        dominant_peak_distance_um = np.full(n_spectra, np.nan)
        dominant_peak_height = np.full(n_spectra, np.nan)
        peak_count = np.zeros(n_spectra, dtype=int)
        all_peak_distances_um = np.empty(n_spectra, dtype=object)
        all_peak_heights = np.empty(n_spectra, dtype=object)

        fft_matrix = []
        distance_axis_um = None
        max_range_um = np.nan

        for i in range(n_spectra):
            res = FFTSolver.solve(wavelengths, spectra[i, :], config)

            if distance_axis_um is None:
                distance_axis_um = res["distance_axis_um"]
                max_range_um = res["max_range_um"]

            peak_distances = res["peak_distances_um"]
            peak_heights = res["peak_heights"]
            peak_count[i] = len(peak_distances)
            all_peak_distances_um[i] = peak_distances
            all_peak_heights[i] = peak_heights

            if len(peak_distances) > 0:
                first_peak_distance_um[i] = peak_distances[0]
                first_peak_height[i] = peak_heights[0]
                dominant_idx = int(np.argmax(peak_heights))
                dominant_peak_distance_um[i] = peak_distances[dominant_idx]
                dominant_peak_height[i] = peak_heights[dominant_idx]

            if config["SAVE_FFT_MATRIX"]:
                fft_matrix.append(res["fft_data"])

        if output_path is None:
            output_path = BatchFFTSolver._default_output_path(npz_path)

        save_kwargs = {
            "source_npz": np.array(os.path.abspath(npz_path)),
            "source_keys": loaded["source_keys"],
            "spectra_key": np.array(loaded["spectra_key"]),
            "scan_axis_name": np.array(loaded["scan_axis_name"]),
            "scan_axis": scan_axis,
            "selected_spectrum_indices": np.asarray(selected_idx, dtype=int),
            "selected_spectrum_count": np.array(len(selected_idx), dtype=int),
            "is_single_spectrum": np.array(len(selected_idx) == 1, dtype=bool),
            "wavelengths": wavelengths,
            "distance_axis_um": distance_axis_um,
            "max_range_um": np.array(max_range_um),
            "peak_count": peak_count,
            "first_peak_distance_um": first_peak_distance_um,
            "first_peak_height": first_peak_height,
            "dominant_peak_distance_um": dominant_peak_distance_um,
            "dominant_peak_height": dominant_peak_height,
            "all_peak_distances_um": all_peak_distances_um,
            "all_peak_heights": all_peak_heights,
            "config_json": np.array(json.dumps(config, ensure_ascii=False)),
        }

        if config["SAVE_FFT_MATRIX"]:
            save_kwargs["fft_matrix"] = np.asarray(fft_matrix)

        np.savez_compressed(output_path, **save_kwargs)
        return output_path

    @staticmethod
    def _default_output_path(npz_path):
        folder = os.path.dirname(os.path.abspath(npz_path))
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(folder, f"{stem}_fft_solved_{timestamp}.npz")


def main_direct():
    """直接在代码中指定输入参数，不使用命令行。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(script_dir, "stackrt_result", "cavity_spectra_20260609_121020.npz")
    output_path = None

    # 选择要解算的光谱：
    # "all"          -> 解算全部光谱
    # 0              -> 只解算第 0 条光谱
    # 200            -> 只解算第 200 条光谱
    # [0, 100, 200]  -> 解算指定多条光谱
    # slice(0, 50)   -> 解算第 0 到 49 条光谱
    spectrum_selection = 0

    config = {
        "FFT_PEAK_HEIGHT_RATIO": 0.2,
        "FFT_IGNORE_DC_BINS": 50,
        "FFT_PEAK_DISTANCE_BINS": 100,
        "ZERO_PAD_FACTOR": 8,
        # 单条光谱绘图会重新计算该条 FFT；只有需要保存所有 FFT 热图时才设为 True。
        "SAVE_FFT_MATRIX": True,
    }

    output_path = BatchFFTSolver.solve_npz(
        npz_path=npz_path,
        output_path=output_path,
        config=config,
        spectrum_selection=spectrum_selection,
    )
    print(f"Saved FFT solving results: {output_path}")


if __name__ == "__main__":
    main_direct()
