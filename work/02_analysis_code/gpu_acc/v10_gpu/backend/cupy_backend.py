"""CuPy float64/complex128 batched implementation of V10 strict TMM."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BackendUnavailableError, ForwardBackend
from .v10_source import load_v10_module, source_sha256


def _import_cupy():
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise BackendUnavailableError(
            "cuda_cupy is unavailable: install cupy-cuda12x in a CUDA-enabled "
            f"environment. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return cp


class CupyStrictTMMBackend(ForwardBackend):
    """Phase 1 GPU backend: strict TMM only, no ILS/Jacobian/optimizer."""

    backend_name = "cuda_cupy"

    @classmethod
    def availability(cls) -> dict[str, Any]:
        try:
            cp = _import_cupy()
            count = int(cp.cuda.runtime.getDeviceCount())
            if count < 1:
                return {
                    "available": False,
                    "reason": "CuPy imported but CUDA reports no devices.",
                    "cupy_version": cp.__version__,
                }
            return {
                "available": True,
                "device_count": count,
                "cupy_version": cp.__version__,
            }
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

    def __init__(self, wavelengths_um: np.ndarray, device_id: int = 0):
        super().__init__(wavelengths_um)
        self.cp = _import_cupy()
        try:
            count = int(self.cp.cuda.runtime.getDeviceCount())
            if not 0 <= int(device_id) < count:
                raise BackendUnavailableError(
                    f"CUDA device {device_id} is unavailable; detected {count} device(s)."
                )
            self.device_id = int(device_id)
            self.device = self.cp.cuda.Device(self.device_id)
            self.device.use()
        except BackendUnavailableError:
            raise
        except Exception as exc:
            raise BackendUnavailableError(
                f"cuda_cupy initialization failed: {type(exc).__name__}: {exc}"
            ) from exc

        v10 = load_v10_module()
        n_matrix_host = np.vstack(
            [v10.material_n(name, self.wavelengths_um) for name in v10.LAYER_NAMES]
        ).astype(np.complex128, copy=False)
        k0_host = (
            2.0 * np.pi / (self.wavelengths_um * 1.0e-6)
        ).astype(np.float64, copy=False)

        # Phase 1 invariant arrays are uploaded exactly once and remain resident.
        self.wavelengths_device = self.cp.asarray(
            self.wavelengths_um, dtype=self.cp.float64
        )
        self.n_matrix_device = self.cp.asarray(
            n_matrix_host, dtype=self.cp.complex128
        )
        self.k0_device = self.cp.asarray(k0_host, dtype=self.cp.float64)
        self.formal_v10_sha256 = source_sha256()

    def parameters_to_device(self, params_batch: Any):
        params = self.validate_params_batch(params_batch)
        with self.device:
            return self.cp.asarray(params, dtype=self.cp.float64)

    def _propagation_cosine(self, tangential_index, n_layer):
        cp = self.cp
        cosine = cp.sqrt(
            cp.asarray(1.0, dtype=cp.complex128)
            - (tangential_index / n_layer[None, :]) ** 2
        )
        return cp.where(cp.real(cosine) < 0.0, -cosine, cosine)

    def predict_batch_device(self, params_device: Any):
        cp = self.cp
        with self.device:
            params = cp.asarray(params_device, dtype=cp.float64)
            if params.ndim == 1:
                params = params[None, :]
            if params.ndim != 2 or params.shape[1] != 6 or params.shape[0] == 0:
                raise ValueError("params_device must have shape (B, 6) or (6,).")

            batch_size = int(params.shape[0])
            n_matrix = self.n_matrix_device
            tangent = (
                n_matrix[0][None, :]
                * cp.sin(cp.deg2rad(params[:, 5]))[:, None]
            )

            cos0 = self._propagation_cosine(tangent, n_matrix[0])
            coss = self._propagation_cosine(tangent, n_matrix[-1])
            q0 = n_matrix[0][None, :] / cos0
            qs = n_matrix[-1][None, :] / coss

            shape = (batch_size, int(self.wavelengths_um.size))
            m11 = cp.ones(shape, dtype=cp.complex128)
            m12 = cp.zeros(shape, dtype=cp.complex128)
            m21 = cp.zeros(shape, dtype=cp.complex128)
            m22 = cp.ones(shape, dtype=cp.complex128)

            # Air is already um; the four film parameters are nm in V10.
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

    def to_host(self, values_device: Any) -> np.ndarray:
        with self.device:
            return self.cp.asnumpy(values_device).astype(np.float64, copy=False)

    def synchronize(self) -> None:
        with self.device:
            self.cp.cuda.get_current_stream().synchronize()

    def memory_stats(self) -> dict[str, Any]:
        with self.device:
            free_bytes, total_bytes = self.cp.cuda.runtime.memGetInfo()
            props = self.cp.cuda.runtime.getDeviceProperties(self.device_id)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            pool = self.cp.get_default_memory_pool()
            return {
                "backend": self.backend_name,
                "available": True,
                "device_id": self.device_id,
                "device_name": str(name),
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "memory_pool_used_bytes": int(pool.used_bytes()),
                "memory_pool_total_bytes": int(pool.total_bytes()),
                "formal_v10_sha256": self.formal_v10_sha256,
            }
