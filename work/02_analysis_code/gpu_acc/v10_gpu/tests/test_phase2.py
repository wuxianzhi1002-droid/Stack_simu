"""Regression tests for the isolated Phase 2 spectrometer contract."""

from __future__ import annotations

import unittest

import numpy as np

from v10_gpu.backend.v10_source import load_v10_module
from v10_gpu.spectrometer import (
    CupyStrictSpectrometerBackend,
    NumpyStrictSpectrometerBackend,
)
from v10_gpu.spectrometer.contract import build_contract


CONFIG={
    "WAVELENGTH_START_UM":0.450,
    "WAVELENGTH_STOP_UM":0.580,
    "INTERNAL_WAVELENGTH_STEP_NM":0.002,
    "SOURCE_REFERENCE_CENTER_NM":515.0,
    "SOURCE_ENVELOPE_SIGMA_NM":55.0,
    "SOURCE_ENVELOPE_FLOOR_REL":0.15,
    "SPECTROMETER":{
        "ILS_ENABLED":True,"ILS_SHAPE":"gaussian","ILS_FWHM_NM":0.02,
        "ILS_TRUNCATE_SIGMA":4.0,"QE_MODEL":"quadratic","QE_CENTER_NM":515.0,
        "QE_PEAK":0.70,"QE_EDGE":0.45,"OPTICAL_THROUGHPUT":0.25,
        "SAMPLE_EXPOSURE_S":0.010,"REFERENCE_EXPOSURE_S":0.010,
    },
}


class Phase2BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reported=np.linspace(0.450,0.451,51,dtype=np.float64)
        cls.params=np.asarray([[100.0,30.0,10.5,40.0,40.0,0.0],[95.5,22.0,3.0,48.0,32.0,-0.08]],dtype=np.float64)
        cls.margin=0.05

    def test_contract_grid_matches_formal_v10(self):
        v10=load_v10_module(); formal=v10.SpectrometerForwardModel(self.reported,CONFIG,self.margin); contract=build_contract(self.reported,CONFIG,self.margin)
        np.testing.assert_array_equal(contract.internal_nm,formal.internal_nm)

    def test_numpy_backend_is_bitwise_formal_v10(self):
        v10=load_v10_module(); formal=v10.SpectrometerForwardModel(self.reported,CONFIG,self.margin); backend=NumpyStrictSpectrometerBackend(self.reported,CONFIG,self.margin)
        expected=np.vstack([formal.predict(row) for row in self.params]); actual=backend.predict_batch(self.params)
        np.testing.assert_array_equal(actual,expected)

    def test_batch_matches_individual_calls(self):
        backend=NumpyStrictSpectrometerBackend(self.reported,CONFIG,self.margin)
        batch=backend.predict_batch(self.params); individual=np.vstack([backend.predict(row) for row in self.params])
        np.testing.assert_array_equal(batch,individual)
        self.assertEqual(batch.shape,(2,self.reported.size)); self.assertTrue(np.all(np.isfinite(batch)))

    def test_jacobian_is_out_of_scope(self):
        backend=NumpyStrictSpectrometerBackend(self.reported,CONFIG,self.margin)
        with self.assertRaises(NotImplementedError): backend.residual_and_jacobian(self.params[0])

    def test_cupy_import_is_safe_without_cuda(self):
        status=CupyStrictSpectrometerBackend.availability(); self.assertIn("available",status)
        if not status["available"]: self.assertTrue(status.get("reason"))


if __name__ == "__main__": unittest.main()
