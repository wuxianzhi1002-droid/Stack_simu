"""CuPy float64/complex128 batched strict TMM for the frozen V9 model."""

from __future__ import annotations

from typing import Any
import numpy as np

from .source import load_v9_module, source_sha256


class BackendUnavailableError(RuntimeError):
    pass


def import_cupy():
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise BackendUnavailableError(
            "CuPy/CUDA is unavailable; install cupy-cuda12x in the GPU environment. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return cp


class CupyV9StrictTMMBackend:
    backend_name = "cuda_cupy_v9_strict_tmm"

    @classmethod
    def availability(cls) -> dict[str, Any]:
        try:
            cp = import_cupy()
            count = int(cp.cuda.runtime.getDeviceCount())
            if count < 1:
                return {"available": False, "reason": "CUDA reports no devices."}
            return {
                "available": True,
                "device_count": count,
                "cupy_version": str(cp.__version__),
            }
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    def __init__(self, wavelengths_um: np.ndarray, device_id: int = 0):
        wavelengths = np.asarray(wavelengths_um, dtype=np.float64)
        if wavelengths.ndim != 1 or wavelengths.size == 0:
            raise ValueError("wavelengths_um must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(wavelengths)) or np.any(np.diff(wavelengths) <= 0.0):
            raise ValueError("wavelengths_um must be finite and strictly increasing")
        self.wavelengths_um = np.ascontiguousarray(wavelengths)
        self.cp = import_cupy()
        count = int(self.cp.cuda.runtime.getDeviceCount())
        if not 0 <= int(device_id) < count:
            raise BackendUnavailableError(
                f"CUDA device {device_id} is unavailable; detected {count} device(s)"
            )
        self.device_id = int(device_id)
        self.device = self.cp.cuda.Device(self.device_id)
        self.device.use()

        v9 = load_v9_module()
        n_matrix = np.vstack(
            [v9.material_n(name, self.wavelengths_um) for name in v9.LAYER_NAMES]
        ).astype(np.complex128, copy=False)
        k0 = (2.0 * np.pi / (self.wavelengths_um * 1.0e-6)).astype(
            np.float64, copy=False
        )
        cp = self.cp
        with self.device:
            self.wavelengths_device = cp.asarray(self.wavelengths_um, dtype=cp.float64)
            self.n_matrix_device = cp.asarray(n_matrix, dtype=cp.complex128)
            self.k0_device = cp.asarray(k0, dtype=cp.float64)
            self.synchronize()
        self.formal_v9_sha256 = source_sha256()

    def validate_params_batch(self, values: Any) -> np.ndarray:
        params = np.asarray(values, dtype=np.float64)
        if params.ndim == 1:
            params = params[None, :]
        if params.ndim != 2 or params.shape[1] != 6 or params.shape[0] == 0:
            raise ValueError("parameters must have shape (B, 6) or (6,)")
        if not np.all(np.isfinite(params)):
            raise ValueError("parameters contain NaN or Inf")
        return np.ascontiguousarray(params)

    def parameters_to_device(self, values: Any):
        with self.device:
            return self.cp.asarray(self.validate_params_batch(values), dtype=self.cp.float64)

    def _propagation_cosine(self, tangent, n_layer):
        cp = self.cp
        value = cp.sqrt(
            cp.asarray(1.0, dtype=cp.complex128)
            - (tangent / n_layer[None, :]) ** 2
        )
        return cp.where(cp.real(value) < 0.0, -value, value)

    def predict_batch_device(self, params_device: Any):
        cp = self.cp
        with self.device:
            params = cp.asarray(params_device, dtype=cp.float64)
            if params.ndim == 1:
                params = params[None, :]
            if params.ndim != 2 or params.shape[1] != 6:
                raise ValueError("params_device must have shape (B, 6) or (6,)")
            batch = int(params.shape[0])
            n_matrix = self.n_matrix_device
            tangent = n_matrix[0][None, :] * cp.sin(cp.deg2rad(params[:, 5]))[:, None]
            cos0 = self._propagation_cosine(tangent, n_matrix[0])
            coss = self._propagation_cosine(tangent, n_matrix[-1])
            q0 = n_matrix[0][None, :] / cos0
            qs = n_matrix[-1][None, :] / coss
            shape = (batch, int(self.wavelengths_um.size))
            m11 = cp.ones(shape, dtype=cp.complex128)
            m12 = cp.zeros(shape, dtype=cp.complex128)
            m21 = cp.zeros(shape, dtype=cp.complex128)
            m22 = cp.ones(shape, dtype=cp.complex128)
            thickness_um = cp.stack(
                (
                    params[:, 0],
                    params[:, 1] / 1000.0,
                    params[:, 2] / 1000.0,
                    params[:, 3] / 1000.0,
                    params[:, 4] / 1000.0,
                ),
                axis=1,
            )
            for layer_index in range(1, 6):
                n_layer = n_matrix[layer_index]
                cosine = self._propagation_cosine(tangent, n_layer)
                q_layer = n_layer[None, :] / cosine
                thickness_m = cp.where(
                    thickness_um[:, layer_index - 1] > 0.0,
                    thickness_um[:, layer_index - 1] * 1.0e-6,
                    0.0,
                )[:, None]
                delta = (
                    self.k0_device[None, :]
                    * n_layer[None, :]
                    * cosine
                    * thickness_m
                )
                c_delta = cp.cos(delta)
                s_delta = cp.sin(delta)
                a11 = c_delta
                a12 = -1j * s_delta / q_layer
                a21 = -1j * q_layer * s_delta
                a22 = c_delta
                old11, old12, old21, old22 = m11, m12, m21, m22
                m11 = old11 * a11 + old12 * a21
                m12 = old11 * a12 + old12 * a22
                m21 = old21 * a11 + old22 * a21
                m22 = old21 * a12 + old22 * a22
            numerator = q0 * m11 + q0 * qs * m12 - m21 - qs * m22
            denominator = q0 * m11 + q0 * qs * m12 + m21 + qs * m22
            return cp.abs(numerator / denominator) ** 2

    def predict_batch(self, values: Any) -> np.ndarray:
        params = self.parameters_to_device(values)
        spectra = self.predict_batch_device(params)
        self.synchronize()
        with self.device:
            return self.cp.asnumpy(spectra).astype(np.float64, copy=False)

    def predict(self, values: Any) -> np.ndarray:
        return self.predict_batch(values)[0]

    def synchronize(self) -> None:
        with self.device:
            self.cp.cuda.get_current_stream().synchronize()

    def resident_identity(self) -> dict[str, Any]:
        arrays = {
            "wavelengths_device": self.wavelengths_device,
            "n_matrix_device": self.n_matrix_device,
            "k0_device": self.k0_device,
        }
        return {
            name: {
                "pointer": int(array.data.ptr),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "nbytes": int(array.nbytes),
            }
            for name, array in arrays.items()
        }

    def memory_stats(self) -> dict[str, Any]:
        cp = self.cp
        with self.device:
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            props = cp.cuda.runtime.getDeviceProperties(self.device_id)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            pool = cp.get_default_memory_pool()
            return {
                "backend": self.backend_name,
                "device_id": self.device_id,
                "device_name": str(name),
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "memory_pool_used_bytes": int(pool.used_bytes()),
                "memory_pool_total_bytes": int(pool.total_bytes()),
                "formal_v9_sha256": self.formal_v9_sha256,
            }
