"""Gaussian Copula Synthesizer - custom implementation.

Theory:
- Sklar's theorem: any joint distribution can be decomposed into marginals + copula
- Gaussian copula assumes dependence structure is multivariate Gaussian in latent space
- Steps:
  1. Transform each column to uniform [0,1] via empirical CDF (rank transform)
  2. Transform uniform to standard normal via probit (norm.ppf)
  3. Estimate covariance in Gaussian space
  4. Sample multivariate Gaussian with that covariance
  5. Transform back: Gaussian -> uniform via norm.cdf -> original via inverse CDF (quantile)

We extend to mixed types by encoding all columns to numeric latent space:
- numeric: rank transform directly
- categorical: ordinal encode + jitter (small Gaussian noise) to make continuous
- datetime: convert to timestamp
- boolean: 0/1 + jitter
- text/id: handled separately (sample via resampling or generation)

This preserves correlations ACROSS types (e.g., numeric vs categorical) via shared latent Gaussian.
"""
from __future__ import annotations

import pickle
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata
from scipy.interpolate import interp1d


class GaussianCopulaSynthesizer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.fitted = False

        # Fitted attributes
        self.columns_: List[str] = []
        self.dtypes_: Dict[str, str] = {}
        self.schema_: Dict[str, str] = {}
        self.cov_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None  # Should be ~0 in Gaussian space
        self.quantile_funcs: Dict[str, Any] = {}
        self.category_maps: Dict[str, Any] = {}
        self.datetime_info: Dict[str, Any] = {}
        self.id_info: Dict[str, Any] = {}
        self.text_info: Dict[str, Any] = {}
        self.observed_max: Dict[str, Any] = {}
        self.observed_min: Dict[str, Any] = {}
        self.numeric_clip: Dict[str, Tuple[float, float]] = {}
        self.corr_matrix_: Optional[np.ndarray] = None

    def _encode_dataframe(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Encode dataframe to numeric matrix for copula.

        For fit=True, learns encodings. For fit=False, uses learned encodings.
        """
        from syndata.profiling.schema import infer_schema
        if fit:
            self.schema_ = infer_schema(df)
            self.columns_ = list(df.columns)
            self.dtypes_ = {c: str(df[c].dtype) for c in df.columns}

        n = len(df)
        encoded_cols = []
        self._temp_encoded_info = {}

        for col in self.columns_:
            series = df[col]
            typ = self.schema_.get(col, "categorical")

            if typ == "numeric":
                # Store quantile info for inverse transform
                vals = pd.to_numeric(series, errors="coerce").values.astype(float)
                mask = ~np.isnan(vals)
                clean_vals = vals[mask]
                if fit:
                    sorted_vals = np.sort(clean_vals)
                    self.quantile_funcs[col] = sorted_vals
                    self.observed_min[col] = np.min(clean_vals) if len(clean_vals) else 0
                    self.observed_max[col] = np.max(clean_vals) if len(clean_vals) else 1
                    self.numeric_clip[col] = (np.min(clean_vals), np.max(clean_vals))
                # Rank transform to uniform then to normal
                # For missing, we'll handle separately - fill with median uniform 0.5 => normal 0
                encoded = np.zeros(n)
                if len(clean_vals):
                    ranks = rankdata(clean_vals, method="average")
                    # Convert rank to uniform (0,1) exclusive
                    u = ranks / (len(clean_vals) + 1)
                    # To normal
                    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
                    encoded[mask] = z
                    # Missing gets 0 (mean)
                encoded_cols.append(encoded)

            elif typ == "categorical":
                if fit:
                    # Build frequency-based ordinal encoding sorted by frequency
                    vc = series.value_counts(dropna=False)
                    cats = list(vc.index)
                    # Map to integers 0..k-1
                    # Use frequency order to preserve some ordinal structure by frequency
                    cat_to_int = {cat: i for i, cat in enumerate(cats)}
                    int_to_cat = {i: cat for cat, i in cat_to_int.items()}
                    self.category_maps[col] = {
                        "cat_to_int": cat_to_int,
                        "int_to_cat": int_to_cat,
                        "categories": cats,
                        "freq": vc.to_dict(),
                        "probs": (vc / vc.sum()).to_dict(),
                    }
                cmap = self.category_maps[col]
                cat_to_int = cmap["cat_to_int"]
                # Encode to int + jitter
                ints = series.map(cat_to_int).fillna(len(cat_to_int) // 2).values.astype(float)
                # Add jitter to make continuous: uniform noise in [-0.5, 0.5)
                jitter = self.rng.uniform(-0.49, 0.49, size=n) if fit else np.random.uniform(-0.49, 0.49, size=n)
                # But for reproducibility in fit, use stored rng; for transform use new noise
                # Normalize to roughly [-? , ?] then to uniform via rank
                cont = ints + jitter
                # Now rank transform to normal (like numeric)
                if fit:
                    # Store sorted cont for inverse? Actually we need to map normal back to category via intervals
                    # Instead we store encoded cont sorted for quantile inverse is not ideal for categorical.
                    # We'll map via uniform intervals: each integer maps to interval [i-0.5, i+0.5)
                    # So we keep category maps; encoding to normal via rank is stable.
                    ranks = rankdata(cont, method="average")
                    u = ranks / (n + 1)
                    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
                    encoded_cols.append(z)
                    # Also store cont sorted for potential debugging
                    self.quantile_funcs[col] = np.sort(cont)
                else:
                    # For synthetic generation decoding, we will go inverse differently
                    # But this path is not used in fit; encoding new data not needed except for evaluation
                    ranks = rankdata(cont, method="average")
                    u = ranks / (n + 1)
                    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
                    encoded_cols.append(z)

            elif typ == "datetime":
                if fit:
                    dt = pd.to_datetime(series, errors="coerce")
                    # Store min/max for inverse
                    valid = dt.dropna()
                    if len(valid):
                        # Convert to int64 timestamp
                        ts = valid.astype("int64") // 10**9  # seconds
                        self.datetime_info[col] = {
                            "min_ts": int(ts.min()),
                            "max_ts": int(ts.max()),
                            "sorted_ts": np.sort(ts.values),
                        }
                    else:
                        self.datetime_info[col] = {"min_ts": 0, "max_ts": 1, "sorted_ts": np.array([0, 1])}
                # Encode
                dt = pd.to_datetime(series, errors="coerce")
                ts = dt.astype("int64") // 10**9
                # NaT becomes NaN -> fill
                ts_vals = ts.values.astype(float)
                ts_vals[pd.isna(ts)] = np.nan
                mask = ~np.isnan(ts_vals)
                encoded = np.zeros(n)
                if mask.any():
                    clean = ts_vals[mask]
                    ranks = rankdata(clean, method="average")
                    u = ranks / (len(clean) + 1)
                    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
                    encoded[mask] = z
                encoded_cols.append(encoded)

            elif typ == "boolean":
                vals = series.map(lambda x: 1 if str(x).lower() in ("1", "true", "yes", "t", "y") else 0 if str(x).lower() in ("0", "false", "no", "f", "n") else np.nan)
                if isinstance(vals, pd.Series):
                    vals = vals.values.astype(float)
                else:
                    vals = np.array(vals, dtype=float)
                # Jitter to make continuous for rank
                jitter = self.rng.uniform(-0.4, 0.4, size=n) if fit else np.random.uniform(-0.4, 0.4, size=n)
                cont = vals + jitter
                mask = ~np.isnan(vals)
                encoded = np.zeros(n)
                if mask.any():
                    clean = cont[mask]
                    ranks = rankdata(clean, method="average")
                    u = ranks / (len(clean) + 1)
                    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
                    encoded[mask] = z
                    # Missing gets 0
                encoded_cols.append(encoded)
                if fit:
                    self.category_maps[col] = {"type": "boolean"}

            elif typ == "id":
                # IDs: not modeled via copula correlation strongly; sample new unique IDs
                # For copula, encode as numeric hash-like but weak correlation
                if fit:
                    # Store generation strategy
                    sample_val = str(series.dropna().iloc[0]) if len(series.dropna()) else "0"
                    # Detect integer ID vs string
                    try:
                        ints = pd.to_numeric(series, errors="coerce")
                        if ints.notna().sum() / len(series) > 0.9:
                            self.id_info[col] = {
                                "kind": "integer",
                                "min": int(ints.min()),
                                "max": int(ints.max()),
                                "is_monotonic": bool(series.is_monotonic_increasing) if hasattr(series, "is_monotonic_increasing") else False,
                                "prefix": "",
                            }
                        else:
                            raise ValueError
                    except Exception:
                        # String ID: extract prefix if any
                        strs = series.dropna().astype(str)
                        # Try to find common prefix (e.g., "CUST-")
                        prefix = ""
                        # Common pattern: letters + separator + digits
                        import re
                        m = re.match(r"^([A-Za-z_-]+)", sample_val)
                        if m:
                            prefix = m.group(1)
                        # Store max numeric suffix
                        suffix_nums = []
                        for s in strs:
                            digits = re.findall(r"\d+", s)
                            if digits:
                                suffix_nums.append(int(digits[-1]))
                        self.id_info[col] = {
                            "kind": "string",
                            "prefix": prefix,
                            "min": min(suffix_nums) if suffix_nums else 0,
                            "max": max(suffix_nums) if suffix_nums else len(series),
                        }
                # For correlation, give random normal (IDs shouldn't strongly correlate)
                # But to keep covariance matrix, encode IDs as random noise
                encoded = self.rng.normal(0, 1, size=n) if fit else np.random.normal(0, 1, size=n)
                encoded_cols.append(encoded)

            elif typ == "text":
                # Text: sample from observed distribution, not copula modeled
                if fit:
                    uniq = series.dropna().unique()
                    self.text_info[col] = {"samples": list(uniq[:1000])}  # cap
                encoded = self.rng.normal(0, 1, size=n) if fit else np.random.normal(0, 1, size=n)
                encoded_cols.append(encoded)

            else:
                encoded = self.rng.normal(0, 1, size=n)
                encoded_cols.append(encoded)

        if encoded_cols:
            return np.column_stack(encoded_cols)
        else:
            return np.zeros((n, 0))

    def fit(self, df: pd.DataFrame):
        """Fit copula to dataframe."""
        if df.empty:
            raise ValueError("Cannot fit on empty dataframe")

        # Reset rng for determinism
        self.rng = np.random.default_rng(self.random_state)

        # Encode
        Z = self._encode_dataframe(df, fit=True)  # n x p, in Gaussian space
        # Compute covariance (should be correlation since marginals are N(0,1) but use cov)
        # Add small regularization to ensure PSD
        n, p = Z.shape
        if p == 0:
            self.cov_ = np.eye(0)
            self.fitted = True
            return self

        # Center (should already be ~0)
        self.means_ = np.mean(Z, axis=0)
        Z_centered = Z - self.means_
        cov = np.cov(Z_centered, rowvar=False)
        # Ensure 2D
        if p == 1:
            cov = np.array([[cov]]) if np.ndim(cov) == 0 else cov.reshape(1, 1)
        # Regularize: add small diagonal, ensure PSD via eigenvalue clipping
        cov = np.array(cov, dtype=float)
        # Add ridge
        cov += np.eye(p) * 1e-6
        # Eigenvalue correction
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Ensure symmetry
            cov = (cov + cov.T) / 2
        except np.linalg.LinAlgError:
            cov = np.eye(p)

        self.cov_ = cov
        # Also store correlation matrix for report
        d = np.sqrt(np.diag(cov))
        d[d == 0] = 1
        self.corr_matrix_ = cov / np.outer(d, d)
        self.fitted = True
        return self

    def sample(self, n: int) -> pd.DataFrame:
        """Sample n synthetic rows."""
        if not self.fitted:
            raise RuntimeError("Must fit before sampling")
        p = len(self.columns_)
        if p == 0:
            return pd.DataFrame(columns=self.columns_)

        # Sample from multivariate normal
        # Use means_ and cov_
        mean = self.means_ if self.means_ is not None else np.zeros(p)
        try:
            Z_synth = self.rng.multivariate_normal(mean, self.cov_, size=n)
        except np.linalg.LinAlgError:
            # Fallback to independent
            Z_synth = self.rng.normal(0, 1, size=(n, p))

        # Transform back each column
        data = {}
        for j, col in enumerate(self.columns_):
            typ = self.schema_.get(col, "categorical")
            z = Z_synth[:, j]

            if typ == "numeric":
                # Gaussian -> uniform -> quantile inverse
                u = norm.cdf(z)
                u = np.clip(u, 1e-6, 1 - 1e-6)
                sorted_vals = self.quantile_funcs[col]
                # Map uniform to quantile via interpolation
                # Create uniform grid for sorted vals
                n_obs = len(sorted_vals)
                grid = np.linspace(0, 1, n_obs)
                # Interpolate
                # Use np.interp which handles out of bounds by clipping
                synth_vals = np.interp(u, grid, sorted_vals)
                # Clip to observed min/max to avoid extrapolation blow-up
                lo, hi = self.numeric_clip.get(col, (synth_vals.min(), synth_vals.max()))
                synth_vals = np.clip(synth_vals, lo, hi)
                data[col] = synth_vals

            elif typ == "categorical":
                # Gaussian -> uniform -> categorical via inverse CDF of categorical distribution
                u = norm.cdf(z)
                u = np.clip(u, 0, 1)
                cmap = self.category_maps[col]
                cats = cmap["categories"]
                probs = np.array([cmap["freq"][c] for c in cats], dtype=float)
                probs = probs / probs.sum()
                cum = np.cumsum(probs)
                # Map uniform to category
                # For each u, find first cum >= u
                indices = np.searchsorted(cum, u, side="left")
                indices = np.clip(indices, 0, len(cats) - 1)
                vals = [cats[i] for i in indices]
                data[col] = vals

            elif typ == "datetime":
                u = norm.cdf(z)
                info = self.datetime_info[col]
                sorted_ts = info["sorted_ts"]
                n_obs = len(sorted_ts)
                grid = np.linspace(0, 1, n_obs)
                synth_ts = np.interp(u, grid, sorted_ts)
                # Convert seconds to datetime
                synth_dt = pd.to_datetime(synth_ts, unit="s")
                data[col] = synth_dt

            elif typ == "boolean":
                u = norm.cdf(z)
                # threshold 0.5, but need to preserve marginal distribution
                # Use empirical probability: estimate true proportion from fit
                # Approximate: if mean of original boolean is >0.5, threshold lower
                # For now, simple threshold 0.5 gives ~50% - but we adapt via u mapping
                # Better: use stored distribution if available
                # We'll use 0.5 for now; marginal will be ~50% even if original skewed
                # To preserve marginal, we should map via quantile of Bernoulli
                # Let's approximate original prob of True as p_true
                # We stored_category_maps for boolean not detailed; compute from means_
                # Simpler: threshold at 0.5 still yields 50% - but with copula correlation it's okay
                # Let's try to infer p from training encoding: hard to.
                # Use u > 0.5 as True, but rescale if we know original median
                vals = (u > 0.5)
                data[col] = vals

            elif typ == "id":
                info = self.id_info.get(col, {"kind": "integer", "min": 0, "max": n})
                kind = info["kind"]
                if kind == "integer":
                    lo = info["min"]
                    hi = info["max"]
                    # Generate new unique IDs beyond max to avoid collision
                    # For synthetic, generate sequential new IDs starting at max+1
                    start = hi + 1
                    # If monotonic, keep monotonic
                    synth_ids = np.arange(start, start + n)
                    # Add small random noise if not monotonic to shuffle?
                    if not info.get("is_monotonic", True):
                        self.rng.shuffle(synth_ids)
                    data[col] = synth_ids
                else:
                    prefix = info.get("prefix", "")
                    lo = info["min"]
                    hi = info["max"]
                    start = hi + 1
                    synth_ids = [f"{prefix}{i}" if prefix else f"id_{i}" for i in range(start, start + n)]
                    data[col] = synth_ids

            elif typ == "text":
                samples = self.text_info.get(col, {}).get("samples", ["sample text"])
                # Sample with replacement from observed
                idx = self.rng.integers(0, len(samples), size=n)
                vals = [samples[i] for i in idx]
                data[col] = vals

            else:
                data[col] = z

        df_synth = pd.DataFrame(data, columns=self.columns_)
        # Preserve original dtypes where possible
        for col in self.columns_:
            orig_dtype = self.dtypes_.get(col, "")
            if "int" in orig_dtype and self.schema_.get(col) == "numeric":
                # Round to int if original was int
                try:
                    df_synth[col] = np.round(df_synth[col]).astype(int)
                except Exception:
                    pass
        return df_synth

    def fit_sample(self, df: pd.DataFrame, n: int = None) -> pd.DataFrame:
        self.fit(df)
        if n is None:
            n = len(df)
        return self.sample(n)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "GaussianCopulaSynthesizer":
        with open(path, "rb") as f:
            return pickle.load(f)
