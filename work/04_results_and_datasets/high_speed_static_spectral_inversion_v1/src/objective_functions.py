from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectrum_preprocess import PreprocessedSpectrum
from tmm_stackrt_matched import StackRTMatchedTMM


def variable_projection(model: np.ndarray, measured: np.ndarray) -> tuple[float, float, np.ndarray]:
    design = np.column_stack((model, np.ones_like(model)))
    coefficients, _, _, _ = np.linalg.lstsq(design, measured, rcond=None)
    scale = max(float(coefficients[0]), 1e-9)
    offset = float(np.mean(measured - scale * model))
    fitted = scale * model + offset
    return scale, offset, fitted


def robust_cost(normalized_residual: np.ndarray, loss: str) -> float:
    z = np.asarray(normalized_residual, dtype=float) ** 2
    if loss == "linear":
        return float(np.sum(z))
    if loss == "soft_l1":
        return float(np.sum(2.0 * (np.sqrt(1.0 + z) - 1.0)))
    raise ValueError("loss must be 'linear' or 'soft_l1'.")


@dataclass
class FitEvaluation:
    residual: np.ndarray
    cost: float
    scale: float
    offset: float
    fitted: np.ndarray


class FitProblem:
    """Optimizer-facing object. It intentionally contains no truth fields."""

    def __init__(self, measurement: PreprocessedSpectrum, loss: str):
        self.measurement = measurement
        self.loss = loss
        self.model = StackRTMatchedTMM(measurement.wavelengths_um)

    @property
    def n_forward_evaluations(self) -> int:
        return self.model.n_forward_evaluations

    def evaluate(self, values: np.ndarray) -> FitEvaluation:
        spectrum = self.model.reflectance(values)
        scale, offset, fitted = variable_projection(spectrum, self.measurement.intensity)
        residual = (fitted - self.measurement.intensity) / self.measurement.sigma
        return FitEvaluation(residual, robust_cost(residual, self.loss), scale, offset, fitted)

    def residual(self, values: np.ndarray) -> np.ndarray:
        return self.evaluate(values).residual

    def cost(self, values: np.ndarray) -> float:
        return self.evaluate(values).cost
