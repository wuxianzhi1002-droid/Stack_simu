"""Tests for the formal V10 optimizer-coordinate adapter."""
from __future__ import annotations
import unittest
import numpy as np
from v10_gpu.optimizer_contract import effective_physical_scales,physical_to_solver,solver_jacobian_from_physical,solver_to_physical

class OptimizerContractTests(unittest.TestCase):
    def test_round_trip_matches_formal_mapping(self):
        values=np.asarray([101.2,31.0,9.0,42.0,37.0,-0.025])
        np.testing.assert_allclose(solver_to_physical(physical_to_solver(values)),values,rtol=0.0,atol=2e-14)

    def test_effective_scales_are_exact_local_derivative(self):
        values=np.asarray([100.0,30.0,10.5,40.0,40.0,0.0]); scales=effective_physical_scales(values)
        self.assertTrue(scales.defined); np.testing.assert_allclose(scales.values,[10.0,20.0,19.0,20.0,20.0,0.2])
        solver=physical_to_solver(values); epsilon=1e-7
        for index in range(6):
            plus=solver.copy(); minus=solver.copy(); plus[index]+=epsilon; minus[index]-=epsilon
            derivative=(solver_to_physical(plus)-solver_to_physical(minus))/(2*epsilon)
            self.assertAlmostEqual(derivative[index],scales.values[index],places=7)

    def test_angle_lower_bound_scale_is_not_arbitrarily_clipped(self):
        values=np.asarray([100.0,30.0,10.5,40.0,40.0,-0.1]); scales=effective_physical_scales(values)
        self.assertFalse(scales.defined); self.assertTrue(np.isnan(scales.values[5]))

    def test_solver_jacobian_is_column_scaling_only(self):
        values=np.asarray([100.0,30.0,10.5,40.0,40.0,0.0]); jacobian=np.arange(24,dtype=float).reshape(4,6)
        expected=jacobian*np.asarray([10.0,20.0,19.0,20.0,20.0,0.2])[None,:]
        np.testing.assert_array_equal(solver_jacobian_from_physical(jacobian,values),expected)

if __name__=="__main__": unittest.main()
