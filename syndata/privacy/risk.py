"""Membership Inference / Re-identification Risk via Nearest Neighbor."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def _prepare_numeric_matrix(df: pd.DataFrame, schema: Dict[str, str] = None, scaler: StandardScaler = None) -> Tuple[np.ndarray, List[str]]:
    """Prepare numeric matrix for distance: encode categorical as one-hot or ordinal, scale numeric.
    If scaler provided, use it to transform (ensure same scaling for real & synth).
    """
    from syndata.profiling.schema import infer_schema
    if schema is None:
        schema = infer_schema(df)
    cols = []
    arrays = []
    for col, typ in schema.items():
        if col not in df.columns:
            continue
        ser = df[col]
        if typ == "numeric":
            med = ser.median() if not ser.isna().all() else 0
            # handle case where median is NaN
            try:
                med_val = float(med) if not pd.isna(med) else 0
            except Exception:
                med_val = 0
            vals = pd.to_numeric(ser, errors="coerce").fillna(med_val).values.reshape(-1, 1)
            arrays.append(vals)
            cols.append(col)
        elif typ == "boolean":
            vals = ser.map(lambda x: 1 if str(x).lower() in ("1", "true", "yes", "t", "y") else 0).fillna(0).values.reshape(-1, 1)
            arrays.append(vals)
            cols.append(col)
        elif typ == "categorical":
            freq = ser.value_counts(normalize=True).to_dict()
            vals = ser.map(freq).fillna(0).values.reshape(-1, 1)
            arrays.append(vals)
            cols.append(col + "_freq")
        elif typ == "datetime":
            dt = pd.to_datetime(ser, errors="coerce")
            ts = dt.astype("int64") // 10**9
            ts = pd.Series(ts).replace({pd.NA: np.nan, -9223372036854775808: np.nan})
            med_ts = ts.median() if ts.notna().any() else 0
            vals = ts.fillna(med_ts).values.reshape(-1, 1)
            arrays.append(vals)
            cols.append(col)
        # id/text: skip
    if not arrays:
        return np.zeros((len(df), 1)), ["dummy"]
    mat = np.hstack(arrays).astype(float)
    mat = np.nan_to_num(mat, nan=0, posinf=0, neginf=0)
    if scaler is None:
        scaler = StandardScaler()
        mat_scaled = scaler.fit_transform(mat)
        # stash scaler for caller if needed via attribute? Return scaled only
        return mat_scaled, cols
    else:
        mat_scaled = scaler.transform(mat)
        return mat_scaled, cols


def _prepare_matrices(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: Dict[str, str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare aligned matrices with same scaler fitted on real."""
    from syndata.profiling.schema import infer_schema
    from sklearn.preprocessing import StandardScaler
    if schema is None:
        schema = infer_schema(real_df)
    # Build real matrix with fit scaler
    # We need to fit scaler on real, then transform both
    # So we call _prepare_numeric_matrix logic but share scaler
    # Extract raw arrays for both with same column order
    cols = []
    real_arrays = []
    synth_arrays = []
    for col, typ in schema.items():
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        ser_r = real_df[col]
        ser_s = synth_df[col]
        if typ == "numeric":
            med = ser_r.median() if not ser_r.isna().all() else 0
            try:
                med_val = float(med) if not pd.isna(med) else 0
            except Exception:
                med_val = 0
            vals_r = pd.to_numeric(ser_r, errors="coerce").fillna(med_val).values.reshape(-1, 1)
            vals_s = pd.to_numeric(ser_s, errors="coerce").fillna(med_val).values.reshape(-1, 1)
            real_arrays.append(vals_r); synth_arrays.append(vals_s); cols.append(col)
        elif typ == "boolean":
            vals_r = ser_r.map(lambda x: 1 if str(x).lower() in ("1", "true", "yes", "t", "y") else 0).fillna(0).values.reshape(-1, 1)
            vals_s = ser_s.map(lambda x: 1 if str(x).lower() in ("1", "true", "yes", "t", "y") else 0).fillna(0).values.reshape(-1, 1)
            real_arrays.append(vals_r); synth_arrays.append(vals_s); cols.append(col)
        elif typ == "categorical":
            freq = ser_r.value_counts(normalize=True).to_dict()
            vals_r = ser_r.map(freq).fillna(0).values.reshape(-1, 1)
            vals_s = ser_s.map(freq).fillna(0).values.reshape(-1, 1)
            real_arrays.append(vals_r); synth_arrays.append(vals_s); cols.append(col + "_freq")
        elif typ == "datetime":
            dt_r = pd.to_datetime(ser_r, errors="coerce")
            dt_s = pd.to_datetime(ser_s, errors="coerce")
            ts_r = dt_r.astype("int64") // 10**9
            ts_s = dt_s.astype("int64") // 10**9
            ts_r = pd.Series(ts_r).replace({pd.NA: np.nan, -9223372036854775808: np.nan})
            ts_s = pd.Series(ts_s).replace({pd.NA: np.nan, -9223372036854775808: np.nan})
            med_ts = ts_r.median() if ts_r.notna().any() else 0
            vals_r = ts_r.fillna(med_ts).values.reshape(-1, 1)
            vals_s = ts_s.fillna(med_ts).values.reshape(-1, 1)
            real_arrays.append(vals_r); synth_arrays.append(vals_s); cols.append(col)
    if not real_arrays:
        return np.zeros((len(real_df),1)), np.zeros((len(synth_df),1))
    real_mat = np.hstack(real_arrays).astype(float)
    synth_mat = np.hstack(synth_arrays).astype(float)
    real_mat = np.nan_to_num(real_mat, nan=0, posinf=0, neginf=0)
    synth_mat = np.nan_to_num(synth_mat, nan=0, posinf=0, neginf=0)
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_mat)
    synth_scaled = scaler.transform(synth_mat)
    return real_scaled, synth_scaled


def nearest_neighbor_distance(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: Dict[str, str] = None) -> Dict[str, Any]:
    """Compute nearest neighbor distances from synthetic to real."""
    real_mat, synth_mat = _prepare_matrices(real_df, synth_df, schema)

    # Fit NN on real
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(real_mat)
    distances, indices = nbrs.kneighbors(synth_mat)

    dists = distances.flatten()
    # Also compute real-to-real distances for baseline
    nbrs_real = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(real_mat)
    real_dists, _ = nbrs_real.kneighbors(real_mat)
    # First neighbor is self (distance 0), so take second
    real_nn_dists = real_dists[:, 1] if real_dists.shape[1] > 1 else real_dists[:, 0]

    # Risk metrics
    mean_dist = float(np.mean(dists))
    median_dist = float(np.median(dists))
    min_dist = float(np.min(dists))
    std_dist = float(np.std(dists))

    # Threshold for "too close": e.g., distance < 5th percentile of real-real distances
    threshold = float(np.percentile(real_nn_dists, 5)) if len(real_nn_dists) else 0.1
    if threshold == 0:
        threshold = float(np.percentile(dists, 5)) if len(dists) else 0.1

    flagged = int(np.sum(dists < threshold))
    flagged_ratio = float(flagged / len(dists)) if len(dists) else 0

    # Also compute Distance to Closest Record (DCR) distribution
    return {
        "mean_distance": mean_dist,
        "median_distance": median_dist,
        "min_distance": min_dist,
        "std_distance": std_dist,
        "threshold": float(threshold),
        "flagged_count": flagged,
        "flagged_ratio": flagged_ratio,
        "real_real_mean": float(np.mean(real_nn_dists)) if len(real_nn_dists) else 0,
        "real_real_median": float(np.median(real_nn_dists)) if len(real_nn_dists) else 0,
        "distances": dists.tolist()[:1000],  # sample for report
        "real_distances": real_nn_dists.tolist()[:1000],
    }


def membership_inference_risk(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: Dict[str, str] = None) -> Dict[str, Any]:
    """Simple membership inference risk score.

    Idea: if synthetic data is too close to real, attacker could infer membership.
    Score 0-100, lower is better (more private).
    Also identifies riskiest columns via leave-one-column-out distance change.
    """
    nn_info = nearest_neighbor_distance(real_df, synth_df, schema)
    # Risk score heuristic
    # Compare synth-real mean distance vs real-real mean
    r_mean = nn_info["real_real_mean"]
    s_mean = nn_info["mean_distance"]
    # If synth distances much smaller than real-real, risk high
    # Risk = 100 * (1 - min(s_mean / r_mean, 1)) if r_mean>0
    # Also incorporate flagged ratio
    if r_mean > 0:
        distance_ratio = s_mean / r_mean
        # distance_ratio <1 means synthetic closer than real neighbors => high risk
        base_risk = max(0, 1 - distance_ratio) * 100
    else:
        base_risk = 50 if s_mean < 0.5 else 10

    # Add flagged ratio component
    flagged_component = nn_info["flagged_ratio"] * 50  # up to 50

    # Combined
    risk_score = float(min(100, base_risk * 0.6 + flagged_component * 0.4 + (10 if nn_info["min_distance"] < 0.01 else 0)))
    # Clamp
    risk_score = float(np.clip(risk_score, 0, 100))

    # Level
    if risk_score < 20:
        level = "low"
    elif risk_score < 50:
        level = "medium"
    else:
        level = "high"

    # Riskiest columns: try dropping each column and see distance change
    riskiest = []
    if schema is None:
        from syndata.profiling.schema import infer_schema
        schema = infer_schema(real_df)
    base_mean = nn_info["mean_distance"]
    for col in list(schema.keys())[:10]:  # limit to 10 for speed
        try:
            sub_real = real_df.drop(columns=[col])
            sub_synth = synth_df.drop(columns=[col])
            sub_schema = {k: v for k, v in schema.items() if k != col}
            sub_info = nearest_neighbor_distance(sub_real, sub_synth, sub_schema)
            delta = sub_info["mean_distance"] - base_mean
            # If removing column increases distance a lot, that column was making synthetic close => risky
            riskiest.append({"column": col, "delta_distance": float(delta), "mean_without": float(sub_info["mean_distance"])})
        except Exception:
            continue
    riskiest.sort(key=lambda x: x["delta_distance"], reverse=True)

    return {
        "risk_score": float(risk_score),
        "risk_level": level,
        "nearest_neighbor": nn_info,
        "riskiest_columns": riskiest[:5],
        "interpretation": {
            "low": "Synthetic data is sufficiently distant from real records. Low re-identification risk.",
            "medium": "Some synthetic records are close to real. Review flagged records.",
            "high": "High similarity to real data. Consider increasing privacy (lower epsilon) or filtering close records."
        }[level],
    }


def filter_close_records(real_df: pd.DataFrame, synth_df: pd.DataFrame, threshold: float = None, schema: Dict[str, str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter out synthetic records that are too close to real."""
    nn_info = nearest_neighbor_distance(real_df, synth_df, schema)
    if threshold is None:
        threshold = nn_info["threshold"]
    real_mat, synth_mat = _prepare_matrices(real_df, synth_df, schema)
    nbrs = NearestNeighbors(n_neighbors=1).fit(real_mat)
    distances, _ = nbrs.kneighbors(synth_mat)
    dists_full = distances.flatten()
    keep_mask = dists_full >= threshold
    filtered = synth_df[keep_mask].reset_index(drop=True)
    return filtered, {
        "original_count": len(synth_df),
        "filtered_count": len(filtered),
        "removed_count": int(len(synth_df) - len(filtered)),
        "threshold": float(threshold),
    }
