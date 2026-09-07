from __future__ import annotations
import json,unittest
from pathlib import Path
from v10_gpu.phase4_report_audit import build_audit

class Phase4AuditTests(unittest.TestCase):
    def test_completed_single_npz_audit_preserves_strict_failure(self):
        root=Path(__file__).resolve().parents[2]
        result=json.loads((root/"cluster_runs/1482539/phase4_result.json").read_text(encoding="utf-8"))
        ident=json.loads((root/"cluster_runs/1482351/phase3_identifiability.json").read_text(encoding="utf-8"))
        audit=build_audit(result,ident)
        self.assertTrue(audit["acceptance_unchanged"])
        self.assertTrue(audit["termination"]["same_status"])
        self.assertTrue(audit["termination"]["both_trajectories_cross_zero_angle"])
        self.assertGreater(audit["runtime"]["warm_optimizer_speedup"],1.0)
        self.assertGreater(audit["identifiability_context"]["max_abs_rho_air_angle"],0.999)

if __name__=="__main__": unittest.main()
