import json
from contextlib import contextmanager
import uuid
import unittest
from pathlib import Path

from v10_gpu import phase6_full_runner


@contextmanager
def workspace_output():
    root = Path.cwd() / (".phase6_test_" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        for name in (
            "phase6_401npz_progress.jsonl",
            "phase6_401npz_results.json",
            "phase6_401npz_table.csv",
            "phase6_401npz_summary.md",
        ):
            candidate = root / name
            if candidate.is_file():
                candidate.unlink()
        root.rmdir()


class Phase6FullRunnerTests(unittest.TestCase):
    def test_formal_source_hash_is_frozen(self):
        self.assertEqual(
            phase6_full_runner.source_sha256(),
            phase6_full_runner.FORMAL_V10_EXPECTED_SHA256,
        )

    def test_progress_jsonl_is_machine_readable(self):
        with workspace_output() as output:
            phase6_full_runner.append_progress(
                output,
                {"index": 1, "filename": "case.npz", "seed": 7, "error": "probe"},
            )
            lines = (output / "phase6_401npz_progress.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["filename"], "case.npz")


    def test_summary_accepts_recorded_optimizer_nonconvergence(self):
        with workspace_output() as output:
            report = {
                "backend": {"effective": "gpu", "initialization_count": 1},
                "selection": {"selected_count": 401},
                "cases": [{
                    "index": 46,
                    "filename": "case.npz",
                    "seed": 7,
                    "gpu_optimizer": {
                        "success": False,
                        "status": 0,
                        "message": "maximum evaluations exceeded",
                        "cost": 1.0,
                        "optimality": 1.0,
                        "exact_RMSE": 1.0,
                        "nfev": 600,
                        "njev": 600,
                        "runtime_s": 1.0,
                        "boundary_hits": [],
                        "fitted_parameters": {},
                    },
                    "gpu_backend": {
                        "actual_gpu_batch_evaluations": 600,
                        "cache_hits": 500,
                        "cache_misses": 600,
                    },
                    "optimizer_nonconvergence": {"classification": "formal outcome"},
                }],
                "performance": {
                    "total_401npz_runtime_s": 2.0,
                    "mean_optimizer_runtime_s": 1.0,
                    "median_optimizer_runtime_s": 1.0,
                    "total_gpu_optimizer_runtime_s": 0.0,
                    "effective_throughput_npz_per_min": 1.0,
                    "estimated_401_runtime_s": 2.0,
                    "estimated_401_runtime_hours": 2.0 / 3600.0,
                    "process_cuda_startup_s": 0.1,
                    "backend_initialization_s": 0.2,
                    "first_warmup_s": 0.3,
                    "existing_cpu_baseline_optimizer_s": 10.0,
                    "startup_amortized_gpu_runtime_per_local_attempt_s": 1.0,
                    "speedup_vs_existing_cpu_local_attempt": 10.0,
                },
                "acceptance": {
                    "pass": True,
                    "optimization_success_count": 400,
                    "optimizer_nonconvergence_count": 1,
                    "closure_pass_count": 400,
                    "A_all_files_processed_without_runtime_failure": True,
                    "B_all_successful_same_parameter_closures_pass": True,
                    "C_formal_semantics_preserved": True,
                    "D_backend_reused": True,
                    "E_no_per_npz_reinitialization": True,
                    "F_final_cpu_verification_complete_for_successful_fits": True,
                    "G_material_speedup": True,
                },
            }
            phase6_full_runner.write_outputs(output, report)
            summary = (output / "phase6_401npz_summary.md").read_text(encoding="utf-8")
            self.assertIn("Formal optimizer non-convergence outcomes: 1/401", summary)
            self.assertIn("Overall: **PASS**", summary)


if __name__ == "__main__":
    unittest.main()
