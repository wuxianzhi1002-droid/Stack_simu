"""Column-wise Jacobian closure metrics, including zero-derivative handling."""
from __future__ import annotations
import numpy as np
from .contract import PARAMETER_NAMES
ZERO_REFERENCE_NORM_LIMIT = 1.0e-12

def column_metrics(expected: np.ndarray, actual: np.ndarray) -> list[dict]:
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if expected.shape != actual.shape or expected.ndim != 2 or expected.shape[1] != 6:
        raise ValueError("Jacobians must share shape (samples, 6).")
    rows = []
    for j, name in enumerate(PARAMETER_NAMES):
        ref = expected[:, j]
        got = actual[:, j]
        delta = got - ref
        ref_norm = float(np.linalg.norm(ref))
        got_norm = float(np.linalg.norm(got))
        defined = ref_norm > ZERO_REFERENCE_NORM_LIMIT
        relative = float(np.linalg.norm(delta) / ref_norm) if defined else None
        cosine = (
            float(np.clip(np.dot(ref, got) / (ref_norm * got_norm), -1.0, 1.0))
            if defined and got_norm > 0.0
            else None
        )
        rows.append({
            "column_index": j,
            "parameter": name,
            "reference_l2_norm": ref_norm,
            "actual_l2_norm": got_norm,
            "relative_error_defined": defined,
            "relative_l2_error": relative,
            "cosine_similarity": cosine,
            "max_abs_difference": float(np.max(np.abs(delta))),
        })
    return rows

def aggregate_case_metrics(cases: list[list[dict]]) -> list[dict]:
    if not cases:
        raise ValueError("At least one case is required.")
    rows = []
    for j, name in enumerate(PARAMETER_NAMES):
        column = [case[j] for case in cases]
        defined = [row for row in column if row["relative_error_defined"]]
        if defined:
            max_relative = max(float(row["relative_l2_error"]) for row in defined)
            mean_relative = float(np.mean([row["relative_l2_error"] for row in defined]))
            cosines = [float(row["cosine_similarity"]) for row in defined if row["cosine_similarity"] is not None]
            min_cosine = min(cosines) if cosines else None
        else:
            max_relative = None
            mean_relative = None
            min_cosine = None
        rows.append({
            "column_index": j,
            "parameter": name,
            "defined_case_count": len(defined),
            "undefined_zero_reference_case_count": len(column) - len(defined),
            "max_relative_l2_error": max_relative,
            "mean_relative_l2_error": mean_relative,
            "min_cosine_similarity": min_cosine,
            "max_abs_difference": max(float(row["max_abs_difference"]) for row in column),
        })
    return rows
