"""Utility/Quality Evaluation."""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def ks_test_numeric(real: pd.Series, synth: pd.Series) -> Dict[str, float]:
    r = pd.to_numeric(real, errors="coerce").dropna()
    s = pd.to_numeric(synth, errors="coerce").dropna()
    if len(r) < 5 or len(s) < 5:
        return {"ks_stat": 0, "p_value": 1}
    ks, p = stats.ks_2samp(r, s)
    return {"ks_stat": float(ks), "p_value": float(p)}


def chi_square_categorical(real: pd.Series, synth: pd.Series) -> Dict[str, float]:
    # Align categories
    cats = list(set(real.dropna().unique()) | set(synth.dropna().unique()))
    if not cats:
        return {"chi2": 0, "p_value": 1}
    r_counts = real.value_counts()
    s_counts = synth.value_counts()
    # Build contingency: 2 x k
    obs = []
    for c in cats:
        obs.append([r_counts.get(c, 0), s_counts.get(c, 0)])
    obs = np.array(obs).T  # 2 x k
    try:
        chi2, p, dof, _ = stats.chi2_contingency(obs)
    except ValueError:
        return {"chi2": 0, "p_value": 1}
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def correlation_preservation(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: Dict[str, str] = None) -> Dict[str, Any]:
    from syndata.correlation.relationships import compute_correlations
    real_corr = compute_correlations(real_df, schema)
    synth_corr = compute_correlations(synth_df, schema)

    # Compare matrices
    real_mat = real_corr.get("correlation_matrix", {})
    synth_mat = synth_corr.get("correlation_matrix", {})
    if not real_mat or not synth_mat:
        return {"mae": None, "preservation_score": None, "real": real_corr, "synth": synth_corr}

    # Flatten to vector
    # Get numeric cols intersection
    cols = list(set(real_mat.keys()) & set(synth_mat.keys()))
    real_vals = []
    synth_vals = []
    for c1 in cols:
        for c2 in cols:
            if c1 < c2:  # lexicographic to avoid double
                try:
                    real_vals.append(real_mat[c1][c2])
                    synth_vals.append(synth_mat[c1][c2])
                except KeyError:
                    pass
    if not real_vals:
        return {"mae": None, "preservation_score": None}

    real_vals = np.array(real_vals, dtype=float)
    synth_vals = np.array(synth_vals, dtype=float)
    mae = float(np.mean(np.abs(real_vals - synth_vals)))
    # Preservation score 0-100, higher better. 100*(1 - mae/2) since corr in [-1,1] max diff 2
    score = float(max(0, 100 * (1 - mae)))
    # Also Pearson between correlation vectors (how well ordering preserved)
    try:
        corr_of_corr = float(np.corrcoef(real_vals, synth_vals)[0, 1]) if len(real_vals) > 2 else 1.0
    except Exception:
        corr_of_corr = 0
    return {
        "mae": mae,
        "preservation_score": score,
        "correlation_of_correlations": float(corr_of_corr) if not np.isnan(corr_of_corr) else 0,
        "real_corr_matrix": real_mat,
        "synth_corr_matrix": synth_mat,
    }


def tstr_test(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: Dict[str, str] = None, target_col: str = None) -> Dict[str, Any]:
    """Train on synthetic, test on real."""
    from syndata.profiling.schema import infer_schema
    if schema is None:
        schema = infer_schema(real_df)

    # Choose target if not provided: last categorical or numeric with reasonable cardinality
    if target_col is None or target_col not in real_df.columns:
        # Prefer categorical with 2-10 values
        candidates = [c for c, t in schema.items() if t == "categorical" and 2 <= real_df[c].nunique() <= 10]
        if candidates:
            target_col = candidates[-1]
        else:
            # Try numeric
            num_candidates = [c for c, t in schema.items() if t == "numeric"]
            if num_candidates:
                target_col = num_candidates[-1]
            else:
                return {"error": "No suitable target column for TSTR", "target": None}

    typ = schema.get(target_col, "categorical")
    # Prepare features: exclude id/text/target
    exclude = {target_col}
    for c, t in schema.items():
        if t in ("id", "text"):
            exclude.add(c)
    feature_cols = [c for c in real_df.columns if c not in exclude]
    if not feature_cols:
        return {"error": "No feature columns for TSTR", "target": target_col}

    # Preprocess: encode categorical, fill NA, scale?
    def preprocess(df_input: pd.DataFrame) -> pd.DataFrame:
        X = df_input[feature_cols].copy()
        for col in feature_cols:
            ct = schema.get(col, "categorical")
            if ct == "categorical":
                # Label encode
                X[col] = X[col].astype(str).fillna("MISSING")
                # Simple factorization
                X[col] = pd.factorize(X[col])[0]
            elif ct == "datetime":
                dt = pd.to_datetime(X[col], errors="coerce")
                X[col] = dt.astype("int64") // 10**9
                X[col] = X[col].replace(-9223372036854775808, np.nan)
            elif ct == "boolean":
                X[col] = X[col].map(lambda v: 1 if str(v).lower() in ("1", "true", "yes") else 0).fillna(0)
            else:  # numeric
                X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        return X

    try:
        # Split real into train/test for baseline; but we need synth train, real test
        realX = preprocess(real_df)
        synthX = preprocess(synth_df)
        realy = real_df[target_col]
        synthy = synth_df[target_col]

        # Align if target type conversion needed
        if typ == "categorical":
            # Encode target
            all_vals = pd.concat([realy.astype(str), synthy.astype(str)], ignore_index=True)
            codes, uniques = pd.factorize(all_vals)
            realy_enc = codes[:len(realy)]
            synthy_enc = codes[len(realy):]
            # For realy_enc we have mixed; better factorize separately but ensure same mapping
            # Use mapping from all_vals uniques
            map_dict = {v: i for i, v in enumerate(uniques)}
            realy_enc = realy.astype(str).map(map_dict).values
            synthy_enc = synthy.astype(str).map(map_dict).values

            # Train/test split for real test set
            # Use realX/realy split for evaluation; train on synth, test on real holdout
            _, realX_test, _, realy_test = train_test_split(realX, realy_enc, test_size=0.3, random_state=42)
            # Also train baseline on real train
            realX_train, _, realy_train, _ = train_test_split(realX, realy_enc, test_size=0.3, random_state=42)

            # Models
            clf_synth = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
            clf_real = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
            clf_synth.fit(synthX, synthy_enc)
            clf_real.fit(realX_train, realy_train)

            pred_synth = clf_synth.predict(realX_test)
            pred_real = clf_real.predict(realX_test)

            acc_synth = float(accuracy_score(realy_test, pred_synth))
            acc_real = float(accuracy_score(realy_test, pred_real))
            f1_synth = float(f1_score(realy_test, pred_synth, average="weighted", zero_division=0))
            f1_real = float(f1_score(realy_test, pred_real, average="weighted", zero_division=0))

            utility_ratio = float(acc_synth / acc_real) if acc_real != 0 else 0
            return {
                "target": target_col,
                "task": "classification",
                "accuracy_synthetic": acc_synth,
                "accuracy_real": acc_real,
                "f1_synthetic": f1_synth,
                "f1_real": f1_real,
                "utility_ratio": utility_ratio,
                "utility_score": float(min(100, utility_ratio * 100)),
            }
        else:
            # Regression
            realy_num = pd.to_numeric(realy, errors="coerce").fillna(realy.median() if not realy.isna().all() else 0)
            synthy_num = pd.to_numeric(synthy, errors="coerce").fillna(synthy.median() if not synthy.isna().all() else 0)
            _, realX_test, _, realy_test = train_test_split(realX, realy_num, test_size=0.3, random_state=42)
            realX_train, _, realy_train, _ = train_test_split(realX, realy_num, test_size=0.3, random_state=42)

            reg_synth = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
            reg_real = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
            reg_synth.fit(synthX, synthy_num)
            reg_real.fit(realX_train, realy_train)

            pred_synth = reg_synth.predict(realX_test)
            pred_real = reg_real.predict(realX_test)

            r2_synth = float(r2_score(realy_test, pred_synth))
            r2_real = float(r2_score(realy_test, pred_real))
            # Handle negative R2
            if r2_real <= 0:
                ratio = 1.0 if r2_synth > 0 else 0
            else:
                ratio = float(r2_synth / r2_real) if r2_real != 0 else 0

            return {
                "target": target_col,
                "task": "regression",
                "r2_synthetic": r2_synth,
                "r2_real": r2_real,
                "utility_ratio": ratio,
                "utility_score": float(max(0, min(100, ratio * 100))),
            }
    except Exception as e:
        return {"error": str(e), "target": target_col}


def evaluate_quality(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: Dict[str, str] = None) -> Dict[str, Any]:
    """Full quality evaluation."""
    from syndata.profiling.schema import infer_schema
    if schema is None:
        schema = infer_schema(real_df)

    # Per-column similarity
    col_metrics = {}
    for col, typ in schema.items():
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        if typ == "numeric":
            ks = ks_test_numeric(real_df[col], synth_df[col])
            # Also mean/std comparison
            r_mean = float(pd.to_numeric(real_df[col], errors="coerce").mean())
            s_mean = float(pd.to_numeric(synth_df[col], errors="coerce").mean())
            r_std = float(pd.to_numeric(real_df[col], errors="coerce").std())
            s_std = float(pd.to_numeric(synth_df[col], errors="coerce").std())
            col_metrics[col] = {
                "type": typ,
                "ks_stat": ks["ks_stat"],
                "ks_p": ks["p_value"],
                "similarity": float(max(0, 1 - ks["ks_stat"])),  # 1 = identical
                "real_mean": r_mean,
                "synth_mean": s_mean,
                "mean_diff_pct": float(abs(r_mean - s_mean) / (abs(r_mean) + 1e-9) * 100) if r_mean != 0 else 0,
                "real_std": r_std,
                "synth_std": s_std,
            }
        elif typ == "categorical":
            chi = chi_square_categorical(real_df[col], synth_df[col])
            # TV distance for categorical
            r_dist = real_df[col].value_counts(normalize=True)
            s_dist = synth_df[col].value_counts(normalize=True)
            all_cats = set(r_dist.index) | set(s_dist.index)
            tv = 0.5 * sum(abs(r_dist.get(c, 0) - s_dist.get(c, 0)) for c in all_cats)
            col_metrics[col] = {
                "type": typ,
                "chi2": chi["chi2"],
                "chi2_p": chi["p_value"],
                "tv_distance": float(tv),
                "similarity": float(max(0, 1 - tv)),
            }
        else:
            col_metrics[col] = {"type": typ, "note": "skipped detailed metric"}

    # Overall statistical similarity average
    similarities = [v["similarity"] for v in col_metrics.values() if "similarity" in v]
    overall_stat = float(np.mean(similarities)) if similarities else 0

    corr_pres = correlation_preservation(real_df, synth_df, schema)
    tstr = tstr_test(real_df, synth_df, schema)

    # Quality score 0-100 aggregate
    # Weights: statistical 40%, correlation 30%, utility 30%
    stat_score = overall_stat * 100
    corr_score = corr_pres.get("preservation_score") or 0
    util_score = tstr.get("utility_score") or tstr.get("utility_score", 0)
    # If TSTR failed, don't count it heavily
    if "error" in tstr:
        overall_quality = float(0.6 * stat_score + 0.4 * corr_score)
    else:
        overall_quality = float(0.4 * stat_score + 0.3 * corr_score + 0.3 * util_score)

    return {
        "per_column": col_metrics,
        "overall_statistical_similarity": overall_stat,
        "statistical_score": float(stat_score),
        "correlation_preservation": corr_pres,
        "tstr": tstr,
        "overall_quality_score": float(overall_quality),
        "quality_level": "excellent" if overall_quality > 80 else "good" if overall_quality > 60 else "fair" if overall_quality > 40 else "poor",
    }
