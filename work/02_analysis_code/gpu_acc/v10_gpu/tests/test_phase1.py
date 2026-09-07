"""Regression tests for the Phase 0/1 strict-TMM contract."""

from __future__ import annotations

import unittest

import numpy as np

from v10_gpu.backend import CupyStrictTMMBackend, NumpyStrictTMMBackend
from v10_gpu.backend.v10_source import load_v10_module


class Phase1BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wavelengths = np.linspace(0.45, 0.58, 257, dtype=np.float64)
        cls.params = np.asarray(
            [
                [100.0, 30.0, 10.5, 40.0, 40.0, 0.0],
                [95.5, 22.0, 3.0, 48.0, 32.0, -0.08],
                [104.5, 38.0, 18.0, 32.0, 48.0, 0.08],
            ],
            dtype=np.float64,
        )

    def test_numpy_backend_is_formal_v10(self):
        backend = NumpyStrictTMMBackend(self.wavelengths)
        v10 = load_v10_module()
        expected = v10.tmm_reflectance(self.wavelengths, self.params[0])
        actual = backend.predict(self.params[0])
        np.testing.assert_array_equal(actual, expected)

    def test_batch_matches_individual_calls(self):
        backend = NumpyStrictTMMBackend(self.wavelengths)
        batch = backend.predict_batch(self.params)
        individual = np.vstack([backend.predict(row) for row in self.params])
        np.testing.assert_array_equal(batch, individual)
        self.assertEqual(batch.shape, (3, self.wavelengths.size))
        self.assertEqual(batch.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(batch)))

    def test_invalid_parameter_shape_is_rejected(self):
        backend = NumpyStrictTMMBackend(self.wavelengths)
        with self.assertRaises(ValueError):
            backend.predict_batch(np.zeros((2, 5)))

    def test_cupy_module_imports_without_cuda(self):
        status = CupyStrictTMMBackend.availability()
        self.assertIn("available", status)
        if not status["available"]:
            self.assertTrue(status.get("reason"))


class ClusterReportTests(unittest.TestCase):
    def _fixtures(self):
        from v10_gpu.cluster_report import EXPECTED_BATCHES

        rows = [
            {
                "batch_size": batch,
                "kernel_s": float(batch),
                "total_s": float(batch),
                "candidates_per_s": 1.0,
                "nan_count": 0,
                "inf_count": 0,
            }
            for batch in EXPECTED_BATCHES
        ]
        benchmark = {
            "formal_v10_sha256": "test",
            "reference_file": "reference.npz",
            "backends": {
                "cpu_numpy": {"status": "ok", "rows": rows},
                "cuda_cupy": {
                    "status": "ok",
                    "rows": rows,
                    "peak_gpu_memory_bytes": 1024,
                    "memory": {"memory_pool_total_bytes": 1024},
                },
            },
            "cpu_gpu_validation": [
                {
                    "batch_size": batch,
                    "rmse": 0.0,
                    "max_abs": 0.0,
                    "nan_inf_count": 0,
                    "pass": True,
                }
                for batch in EXPECTED_BATCHES
            ],
        }
        baseline = {
            "backends": {
                "cpu_numpy": {
                    "rows": [
                        {"batch_size": batch, "total_s": float(batch) * 2.0}
                        for batch in EXPECTED_BATCHES
                    ]
                }
            }
        }
        machine = {
            "hostname": "test-node",
            "visible_gpu_count": 1,
            "visible_devices": [{"name": "NVIDIA GeForce RTX 5090"}],
            "driver_version": "test",
            "cuda_runtime": "12.8",
            "cuda_driver_api": "12.8",
            "cupy_version": "13.test",
            "python_version": "3.11.test",
            "slurm": {"partition": "gpu_5090"},
        }
        return machine, benchmark, baseline

    def test_cluster_report_passes_valid_phase1_data(self):
        from v10_gpu.cluster_report import build_report

        report = build_report(*self._fixtures())
        self.assertTrue(report["pass"])
        self.assertEqual(len(report["batches"]), 5)
        self.assertTrue(
            all(row["speedup_vs_existing_cpu"] == 2.0 for row in report["batches"])
        )

    def test_cluster_report_fails_rmse_threshold(self):
        from v10_gpu.cluster_report import build_report

        machine, benchmark, baseline = self._fixtures()
        benchmark["cpu_gpu_validation"][0]["rmse"] = 1.1e-10
        report = build_report(machine, benchmark, baseline)
        self.assertFalse(report["pass"])
        self.assertTrue(any("RMSE" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
