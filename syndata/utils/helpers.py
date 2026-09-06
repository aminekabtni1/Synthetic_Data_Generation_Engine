"""Utility helpers."""
import pandas as pd
import numpy as np


def is_id_column(series: pd.Series, threshold: float = 0.95) -> bool:
    """Heuristic: name contains 'id' or high cardinality + uniqueness."""
    name = series.name.lower() if isinstance(series.name, str) else ""
    if "id" in name:
        if series.nunique() / max(len(series), 1) > 0.9:
            return True
    # Unique high cardinality generic column
    if series.nunique() == len(series) and len(series) > 10:
        # If dtype is int and monotonic or uuid-like
        return True
    return False


def clip_uniform(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(u, eps, 1 - eps)


def empirical_quantile(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Interpolated quantile mapping."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    # uniform grid for empirical CDF
    grid = np.linspace(0, 1, n)
    return np.interp(quantiles, grid, sorted_vals)
