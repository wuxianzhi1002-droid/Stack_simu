from __future__ import annotations
import unittest
import numpy as np
from types import SimpleNamespace
from v10_gpu.optimizer_contract import physical_to_solver
from v10_gpu.phase5_cache import ExactBatchResidualJacobianCache
from v10_gpu.phase5_runner import GpuLeastSquaresAdapter,patched_formal_least_squares

class LinearBatchBackend:
    def __init__(self,matrix):self.matrix=np.asarray(matrix);self.batch_calls=0
    def predict_batch(self,parameters):self.batch_calls+=1;return np.asarray(parameters)@self.matrix.T
    def synchronize(self):pass

class Phase5CacheTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(5);self.matrix=rng.normal(size=(20,6));self.backend=LinearBatchBackend(self.matrix);self.target=np.asarray([100.,30.,10.,40.,40.,0.02]);self.observed=self.matrix@self.target;self.cache=ExactBatchResidualJacobianCache(self.backend,self.observed,1.0);self.solver=physical_to_solver(np.asarray([101.,31.,11.,41.,39.,0.01]))
    def test_residual_then_jacobian_uses_one_exact_batch(self):
        residual=self.cache.residual(self.solver);jac=self.cache.jacobian(self.solver);stats=self.cache.snapshot()
        self.assertEqual(residual.shape,(20,));self.assertEqual(jac.shape,(20,6));self.assertEqual(self.backend.batch_calls,1);self.assertEqual(stats["actual_gpu_batch_evaluations"],1);self.assertEqual(stats["cache_misses"],1);self.assertEqual(stats["cache_hits"],1)
    def test_nextafter_invalidates_exact_key(self):
        self.cache.residual(self.solver);changed=self.solver.copy();changed[0]=np.nextafter(changed[0],np.inf);self.cache.residual(changed)
        self.assertEqual(self.backend.batch_calls,2);self.assertEqual(self.cache.snapshot()["cache_misses"],2)
    def test_least_squares_adapter_uses_cached_fun_and_jac(self):
        adapter=GpuLeastSquaresAdapter(self.cache);fake=SimpleNamespace(least_squares=lambda *a,**k:None);original=fake.least_squares
        config={"bounds":(np.zeros(6),np.ones(6)),"loss":"linear","max_nfev":30,"x_scale":1.0,"ftol":1e-8,"xtol":1e-8,"gtol":1e-8}
        result=adapter(lambda x:x,self.solver,**config)
        self.assertTrue(result.success);self.assertGreater(result.njev,0);self.assertLess(self.backend.batch_calls,self.cache.residual_calls+self.cache.jacobian_calls)
        with patched_formal_least_squares(fake,adapter):self.assertIs(fake.least_squares,adapter)
        self.assertIs(fake.least_squares,original)

if __name__=="__main__":unittest.main()
