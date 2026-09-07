"""Tests for the read-only Phase 3 identifiability diagnostics."""
from __future__ import annotations
import unittest
import numpy as np
from v10_gpu.identifiability import analyze_case,angle_norm_power_law,normalized_column_correlation

class IdentifiabilityTests(unittest.TestCase):
    def test_normalized_correlation_matrix(self):
        rng=np.random.default_rng(7); jacobian=rng.normal(size=(100,6)); jacobian[:,5]=2.0*jacobian[:,0]
        norms,correlation=normalized_column_correlation(jacobian)
        self.assertEqual(norms.shape,(6,)); self.assertEqual(correlation.shape,(6,6))
        np.testing.assert_allclose(np.diag(correlation),1.0,rtol=0.0,atol=2e-15)
        self.assertAlmostEqual(correlation[0,5],1.0,places=14)

    def test_zero_column_is_explicitly_undefined_and_rank_deficient(self):
        rng=np.random.default_rng(11); cpu=rng.normal(size=(80,6)); cpu[:,5]=0.0; gpu=cpu.copy()
        result=analyze_case(np.asarray([100,30,10.5,40,40,0],dtype=np.float64),cpu,gpu)
        self.assertEqual(result["numerical_rank"],5)
        self.assertIsNone(result["rho_air_angle"])
        self.assertIsNone(result["angle_relative_l2_error"])
        self.assertTrue(result["angle_zero_reference_undefined"])
        self.assertEqual(result["jacobian_column_norms"]["Angle"],0.0)

    def test_singular_values_and_condition_number(self):
        diagonal=np.asarray([9.0,7.0,5.0,3.0,2.0,1.0]); cpu=np.diag(diagonal); gpu=cpu.copy()
        result=analyze_case(np.asarray([100,30,10.5,40,40,0.01]),cpu,gpu)
        np.testing.assert_allclose(result["singular_values"],diagonal)
        self.assertAlmostEqual(result["condition_number_J"],9.0)
        self.assertAlmostEqual(result["condition_number_JTJ"],81.0)
        self.assertEqual(result["numerical_rank"],6)

    def test_angle_norm_power_law_detects_linear_trend(self):
        cases=[]
        for angle in (1e-5,3e-5,1e-4,3e-4,1e-3):
            cases.append({"abs_reflector_angle_deg":angle,"jacobian_column_norms":{"Angle":4.2*angle}})
        fit=angle_norm_power_law(cases)
        self.assertAlmostEqual(fit["slope"],1.0,places=12)
        self.assertAlmostEqual(fit["r_squared"],1.0,places=12)

if __name__ == "__main__": unittest.main()
