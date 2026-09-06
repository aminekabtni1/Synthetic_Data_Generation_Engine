"""Correlation + Relationship Preservation."""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats


def compute_correlations(df: pd.DataFrame, schema: Dict[str, str] = None) -> Dict[str, Any]:
    """Compute correlations between numeric columns.

    Returns dict with correlation matrix, p-values, strong pairs.
    """
    if schema is None:
        from syndata.profiling.schema import infer_schema
        schema = infer_schema(df)

    numeric_cols = [c for c, t in schema.items() if t == "numeric"]
    if len(numeric_cols) < 2:
        return {
            "correlation_matrix": {},
            "strong_correlations": [],
            "numeric_columns": numeric_cols,
        }

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(method="pearson")
    # Spearman as well for non-linear monotonic
    spearman = numeric_df.corr(method="spearman")

    # p-values for pearson
    pvals = pd.DataFrame(np.ones(corr.shape), index=corr.index, columns=corr.columns)
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:
                # drop na pairwise
                pair = numeric_df[[col1, col2]].dropna()
                if len(pair) >= 3:
                    try:
                        _, p = stats.pearsonr(pair[col1], pair[col2])
                        pvals.loc[col1, col2] = p
                        pvals.loc[col2, col1] = p
                    except Exception:
                        pass

    strong = []
    for i, c1 in enumerate(numeric_cols):
        for j, c2 in enumerate(numeric_cols):
            if i < j:
                val = corr.loc[c1, c2]
                if abs(val) > 0.3 and pvals.loc[c1, c2] < 0.05:
                    strong.append({
                        "col1": c1,
                        "col2": c2,
                        "pearson": float(val),
                        "spearman": float(spearman.loc[c1, c2]),
                        "p_value": float(pvals.loc[c1, c2]),
                        "strength": "strong" if abs(val) > 0.7 else "moderate" if abs(val) > 0.5 else "weak",
                    })
    # Sort by abs correlation
    strong.sort(key=lambda x: abs(x["pearson"]), reverse=True)

    return {
        "correlation_matrix": corr.to_dict(),
        "spearman_matrix": spearman.to_dict(),
        "p_values": pvals.to_dict(),
        "strong_correlations": strong,
        "numeric_columns": numeric_cols,
    }


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V for categorical-categorical association."""
    confusion = pd.crosstab(x, y)
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return 0.0
    try:
        chi2 = stats.chi2_contingency(confusion)[0]
    except ValueError:
        return 0.0
    n = confusion.sum().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom == 0:
        return 0.0
    return np.sqrt(phi2corr / denom)


def categorical_dependencies(df: pd.DataFrame, schema: Dict[str, str] = None, top_n: int = 10) -> Dict[str, Any]:
    """Detect dependencies between categorical columns."""
    if schema is None:
        from syndata.profiling.schema import infer_schema
        schema = infer_schema(df)

    cat_cols = [c for c, t in schema.items() if t == "categorical"]
    if len(cat_cols) < 2:
        return {"dependencies": [], "categorical_columns": cat_cols}

    deps = []
    for i, c1 in enumerate(cat_cols):
        for j, c2 in enumerate(cat_cols):
            if i < j:
                # Drop NA
                pair = df[[c1, c2]].dropna()
                if len(pair) < 10:
                    continue
                v = cramers_v(pair[c1], pair[c2])
                if v > 0.1:
                    deps.append({
                        "col1": c1,
                        "col2": c2,
                        "cramers_v": float(v),
                        "strength": "strong" if v > 0.5 else "moderate" if v > 0.3 else "weak",
                    })
    deps.sort(key=lambda x: x["cramers_v"], reverse=True)
    return {
        "categorical_columns": cat_cols,
        "dependencies": deps[:top_n],
    }


def numerical_categorical_association(df: pd.DataFrame, schema: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """ANOVA-like: does categorical explain variance in numeric? Using correlation ratio (eta)."""
    if schema is None:
        from syndata.profiling.schema import infer_schema
        schema = infer_schema(df)

    num_cols = [c for c, t in schema.items() if t == "numeric"]
    cat_cols = [c for c, t in schema.items() if t == "categorical"]
    results = []
    for num in num_cols:
        for cat in cat_cols:
            try:
                sub = df[[num, cat]].dropna()
                if sub[cat].nunique() < 2:
                    continue
                groups = [g[num].values for _, g in sub.groupby(cat)]
                if len(groups) < 2:
                    continue
                # ANOVA F-test
                f, p = stats.f_oneway(*groups)
                # Eta squared approximation
                grand_mean = sub[num].mean()
                ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
                ss_total = ((sub[num] - grand_mean) ** 2).sum()
                eta_sq = ss_between / ss_total if ss_total != 0 else 0
                if eta_sq > 0.05 and p < 0.05:
                    results.append({
                        "numeric": num,
                        "categorical": cat,
                        "eta_squared": float(eta_sq),
                        "f_stat": float(f),
                        "p_value": float(p),
                    })
            except Exception:
                continue
    results.sort(key=lambda x: x["eta_squared"], reverse=True)
    return results


def analyze_all(df: pd.DataFrame, schema: Dict[str, str] = None) -> Dict[str, Any]:
    """Comprehensive relationship analysis."""
    corr = compute_correlations(df, schema)
    cat_deps = categorical_dependencies(df, schema)
    num_cat = numerical_categorical_association(df, schema)
    return {
        "numeric_correlations": corr,
        "categorical_dependencies": cat_deps,
        "numeric_categorical_associations": num_cat,
    }
