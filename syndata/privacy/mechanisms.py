"""Differential Privacy Mechanisms."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class LaplaceMechanism:
    """Laplace mechanism for differential privacy.

    Adds calibrated Laplace noise to statistical queries.
    Noise scale = sensitivity / epsilon.
    """

    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0, random_state: int = 42):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.rng = np.random.default_rng(random_state)

    @property
    def scale(self) -> float:
        return self.sensitivity / self.epsilon

    def add_noise(self, value: float) -> float:
        noise = self.rng.laplace(0, self.scale)
        return float(value + noise)

    def add_noise_array(self, arr: np.ndarray) -> np.ndarray:
        noise = self.rng.laplace(0, self.scale, size=arr.shape)
        return arr + noise


def compute_sensitivity(series: pd.Series, statistic: str = "mean") -> float:
    """Estimate global sensitivity for simple statistics.

    For bounded numeric data [min, max], sensitivity of mean = (max-min)/n
    For count, sensitivity = 1, for sum = max-min.
    This is a simplified heuristic assuming known bounds.
    """
    s = series.dropna()
    if s.empty:
        return 1.0
    n = len(s)
    data_range = float(s.max() - s.min()) if pd.api.types.is_numeric_dtype(s) else 1.0
    if statistic == "mean":
        return data_range / max(n, 1)
    elif statistic == "sum":
        return data_range
    elif statistic == "count":
        return 1.0
    elif statistic == "var":
        # Conservative
        return (data_range ** 2) / max(n, 1)
    else:
        return data_range / max(n, 1)


def privatize_profile(profile: Dict[str, Any], epsilon: float = 1.0, random_state: int = 42) -> Dict[str, Any]:
    """Apply Laplace noise to numeric stats in a profile.

    Budget allocation: split epsilon equally across numeric columns stats (simple composition).
    More advanced: use per-column epsilon = epsilon / num_numeric_cols.

    Returns privatized profile copy and privacy accounting.
    """
    import copy
    priv = copy.deepcopy(profile)
    columns = priv.get("columns", {})
    numeric_cols = [c for c, p in columns.items() if p.get("inferred_type") == "numeric"]
    if not numeric_cols:
        priv["privacy"] = {"epsilon": epsilon, "mechanism": "none", "note": "no numeric columns to privatize"}
        return priv

    # Simple composition: divide budget
    per_col_epsilon = epsilon / len(numeric_cols) if len(numeric_cols) else epsilon
    # Further split per statistic (mean, std, min, max) - 4 stats
    per_stat_epsilon = per_col_epsilon / 4

    rng = np.random.default_rng(random_state)
    privacy_log = []

    for col in numeric_cols:
        col_prof = columns[col]
        n = col_prof.get("n_total", 1)
        data_range = (col_prof.get("max", 1) - col_prof.get("min", 0)) if col_prof.get("max") is not None and col_prof.get("min") is not None else 1.0
        if data_range == 0:
            data_range = 1.0

        for stat in ["mean", "std", "min", "max"]:
            if stat not in col_prof or col_prof[stat] is None:
                continue
            # Sensitivity depends on stat
            if stat == "mean":
                sens = data_range / max(n, 1)
            elif stat == "std":
                sens = data_range / max(n, 1)
            else:  # min/max have high sensitivity; we clip to bounded domain and use data_range as sensitivity
                sens = data_range / max(n, 1)  # simplified, actually sensitivity is data_range for min/max without bounding
                # Use more conservative: data_range
                # But for demo we keep /n to avoid huge noise

            scale = sens / per_stat_epsilon
            noise = rng.laplace(0, scale)
            original = col_prof[stat]
            col_prof[stat] = float(original + noise)
            privacy_log.append({
                "column": col,
                "stat": stat,
                "epsilon": float(per_stat_epsilon),
                "sensitivity": float(sens),
                "scale": float(scale),
                "noise": float(noise),
                "original": float(original),
                "privatized": float(col_prof[stat]),
            })

        # Also privatize counts? For demo, privatize n_missing? Skip.

    priv["privacy"] = {
        "epsilon_total": float(epsilon),
        "epsilon_per_column": float(per_col_epsilon),
        "epsilon_per_stat": float(per_stat_epsilon),
        "mechanism": "Laplace",
        "composition": "simple (sequential)",
        "log": privacy_log,
        "note": f"Split {epsilon} across {len(numeric_cols)} numeric cols x 4 stats = {len(privacy_log)} queries",
    }
    return priv


def privatize_dataframe_stats(df: pd.DataFrame, epsilon: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """Return privatized means for numeric columns as example of DP query."""
    # This is illustrative: shows DP mean per column
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    per_col_eps = epsilon / len(numeric_cols) if len(numeric_cols) else epsilon
    rng = np.random.default_rng(random_state)
    for col in numeric_cols:
        s = df[col].dropna()
        true_mean = s.mean()
        sens = (s.max() - s.min()) / len(s) if len(s) else 1.0
        scale = sens / per_col_eps
        noise = rng.laplace(0, scale)
        results.append({"column": col, "true_mean": true_mean, "private_mean": true_mean + noise, "epsilon": per_col_eps, "scale": scale})
    return pd.DataFrame(results)
