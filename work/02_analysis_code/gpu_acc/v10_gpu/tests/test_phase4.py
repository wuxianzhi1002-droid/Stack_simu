"""Tests for Phase 4 optimizer wiring without CUDA."""
from __future__ import annotations
import unittest
from types import SimpleNamespace
import numpy as np
from v10_gpu.optimizer_contract import physical_to_solver
from v10_gpu.phase4_optimizer import TimedResidual,first_step,run_local

class LinearPhysicalBackend:
    def __init__(self,matrix): self.matrix=matrix; self.calls=0
    def predict(self,values): self.calls+=1; return self.matrix@np.asarray(values)

class Phase4WiringTests(unittest.TestCase):
    def test_cpu_local_solver_wiring_uses_same_bounds_and_settings(self):
        rng=np.random.default_rng(4); matrix=rng.normal(size=(12,6)); target=np.asarray([100,30,10,40,40,0.02],float); observed=matrix@target
        backend=LinearPhysicalBackend(matrix); fun=TimedResidual(backend,observed,1.0,'linear'); config=SimpleNamespace(loss='linear',max_nfev=200,local_gtol=1e-10)
        start=np.asarray([101,31,11,41,39,-0.01],float); result,runtime,trajectory,callback=run_local(fun,physical_to_solver(start),config,'2-point')
        self.assertTrue(result.success); self.assertGreater(runtime,0); self.assertGreater(fun.calls,0); self.assertTrue(np.all(result.x>=0)); self.assertTrue(np.all(result.x<=1)); self.assertFalse(callback)

    def test_first_step_does_not_evaluate_model(self):
        x0=physical_to_solver(np.asarray([100,30,10,40,40,0],float)); moved=x0.copy(); moved[0]+=0.01
        record={'solver':moved.tolist(),'physical':None,'cost':2.0}; from v10_gpu.optimizer_contract import solver_to_physical; record['physical']=solver_to_physical(moved).tolist()
        step=first_step([record],x0); self.assertAlmostEqual(step['solver_step'][0],0.01); self.assertEqual(step['cost'],2.0)

if __name__=='__main__': unittest.main()
