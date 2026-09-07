"""Regression tests for Phase 3 B=13 center differences."""
from __future__ import annotations
import unittest
import numpy as np
from v10_gpu.backend.v10_source import load_v10_module
from v10_gpu.jacobian import BatchedCenterDifferenceJacobian,build_center_difference_batch,formal_cpu_jacobian
from v10_gpu.jacobian.metrics import aggregate_case_metrics,column_metrics
from v10_gpu.spectrometer import NumpyStrictSpectrometerBackend

CONFIG={"WAVELENGTH_START_UM":0.450,"WAVELENGTH_STOP_UM":0.580,"INTERNAL_WAVELENGTH_STEP_NM":0.002,"SOURCE_REFERENCE_CENTER_NM":515.0,"SOURCE_ENVELOPE_SIGMA_NM":55.0,"SOURCE_ENVELOPE_FLOOR_REL":0.15,"SPECTROMETER":{"ILS_ENABLED":True,"ILS_SHAPE":"gaussian","ILS_FWHM_NM":0.02,"ILS_TRUNCATE_SIGMA":4.0,"QE_MODEL":"quadratic","QE_CENTER_NM":515.0,"QE_PEAK":0.70,"QE_EDGE":0.45,"OPTICAL_THROUGHPUT":0.25,"SAMPLE_EXPOSURE_S":0.010,"REFERENCE_EXPOSURE_S":0.010}}

class LinearBackend:
    def __init__(self):
        self.coefficients=np.arange(1,25,dtype=np.float64).reshape(6,4); self.calls=0
    def predict_batch(self,params):
        self.calls+=1; return np.asarray(params,dtype=np.float64)@self.coefficients

class Phase3Tests(unittest.TestCase):
    def test_batch_order_and_bounds(self):
        v10=load_v10_module(); lower,upper=v10.bounds_arrays()
        for values in ((lower+upper)/2.0,lower,upper):
            batch=build_center_difference_batch(values)
            self.assertEqual(batch.parameters.shape,(13,6))
            np.testing.assert_array_equal(batch.parameters[0],values)
            np.testing.assert_array_equal(batch.parameters[1:7],batch.plus)
            np.testing.assert_array_equal(batch.parameters[7:13],batch.minus)
            self.assertTrue(np.all(batch.parameters >= lower)); self.assertTrue(np.all(batch.parameters <= upper))

    def test_one_batch_call_and_exact_linear_jacobian(self):
        backend=LinearBackend(); scale=2.5; values=np.asarray([100.0,30.0,10.0,40.0,40.0,0.0])
        actual=BatchedCenterDifferenceJacobian(backend).jacobian(values,scale)
        self.assertEqual(backend.calls,1); np.testing.assert_allclose(actual,backend.coefficients.T/scale,rtol=1e-7,atol=1e-9)

    def test_zero_reference_column_is_marked_undefined(self):
        expected=np.zeros((5,6),dtype=np.float64); actual=expected.copy(); actual[:,5]=1.0e-7
        metrics=column_metrics(expected,actual); aggregate=aggregate_case_metrics([metrics])
        self.assertFalse(metrics[5]["relative_error_defined"])
        self.assertIsNone(aggregate[5]["max_relative_l2_error"])
        self.assertEqual(aggregate[5]["undefined_zero_reference_case_count"],1)
        self.assertAlmostEqual(aggregate[5]["max_abs_difference"],1.0e-7)

    def test_matches_formal_cpu_center_difference(self):
        reported=np.linspace(0.450,0.451,51,dtype=np.float64); margin=0.05
        backend=NumpyStrictSpectrometerBackend(reported,CONFIG,margin); values=np.asarray([100.0,30.0,10.5,40.0,40.0,0.0]); observed=backend.predict(values); scale=0.25
        expected=formal_cpu_jacobian(backend.model,values,observed,scale)
        actual=BatchedCenterDifferenceJacobian(backend).jacobian(values,scale)
        np.testing.assert_array_equal(actual,expected)

if __name__ == "__main__": unittest.main()
