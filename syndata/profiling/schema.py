"""Schema + Profile Inference.
Detects column types, computes statistical profiles, detects keys/relationships.
"""
from __future__ import annotations

import re
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats


def infer_column_type(series: pd.Series) -> str:
    """Infer semantic type of a column.

    Returns one of: id, boolean, numeric, categorical, datetime, text
    """
    s = series.dropna()
    if s.empty:
        return "categorical"
    name = str(series.name).lower() if series.name else ""

    # Boolean detection
    bool_vals = {"true", "false", "0", "1", "yes", "no", "y", "n", "t", "f"}
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    # Check boolean-like
    unique_lower = set(str(x).lower() for x in s.unique()[:20])
    if unique_lower.issubset(bool_vals) and len(s.unique()) <= 2:
        return "boolean"
    if set(s.unique()).issubset({0, 1, True, False, "0", "1"}) and len(s.unique()) <= 2:
        return "boolean"

    n_unique = s.nunique()
    n_total = len(series)
    uniqueness = n_unique / max(n_total, 1)

    # ID detection - strict heuristics to avoid misclassifying continuous numeric as ID
    # 1) Name contains 'id' + high uniqueness => ID (covers customer_id, order_id)
    if "id" in name and uniqueness > 0.95:
        return "id"
    # 2) UUID-like string
    if uniqueness == 1.0 and n_total > 10:
        uuid_pattern = re.compile(r"^[0-9a-f]{8}-", re.I)
        try:
            sample = str(s.iloc[0])
            if uuid_pattern.match(sample):
                return "id"
        except Exception:
            pass
        # Integer monotonic unique => likely ID (e.g., auto-increment)
        # Only for integer dtype, not float continuous
        if pd.api.types.is_integer_dtype(series):
            # monotonic increasing unique ints are strong ID signal
            try:
                if s.is_monotonic_increasing and uniqueness == 1.0:
                    return "id"
            except Exception:
                pass
        # For object/string IDs with prefix pattern (e.g., P00001, CUST-123)
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Check if values look like prefixed IDs: common prefix + digits
            sample_str = str(s.iloc[0])
            # pattern: letters/underscore/dash then digits
            if re.match(r"^[A-Za-z]+[-_]?[0-9]+$", sample_str) and s.astype(str).str.match(r"^[A-Za-z]+[-_]?[0-9]+$").mean() > 0.9:
                return "id"

    # Datetime detection
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Try to parse as datetime if object
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        # Sample few values to test datetime parsing
        try:
            sample = s.head(20)
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() / len(sample) > 0.8:
                return "datetime"
        except Exception:
            pass

    # Numeric detection
    if pd.api.types.is_numeric_dtype(series):
        # Check if actually categorical encoded as numeric but low cardinality
        if n_unique <= 10 and n_unique < n_total * 0.1:
            # Might be categorical, but keep as numeric unless name suggests category
            # For now, treat low cardinality ints as categorical if name hints
            cat_hints = ["category", "type", "status", "level", "grade", "rating", "code"]
            if any(h in name for h in cat_hints):
                return "categorical"
        # Text that is numeric but stored as string would have been handled above
        # Distinguish text vs categorical vs numeric
        return "numeric"

    # Try coercing to numeric for object columns that are actually numeric
    try:
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().sum() / len(s) > 0.9:
            return "numeric"
    except Exception:
        pass

    # Text vs categorical: high cardinality + long strings => text
    avg_len = s.astype(str).str.len().mean()
    if n_unique > 50 and avg_len > 30:
        return "text"
    if n_unique > min(100, n_total * 0.5) and avg_len > 20:
        return "text"

    return "categorical"


def _distribution_shape(series: pd.Series) -> str:
    """Classify numeric distribution shape."""
    s = series.dropna()
    if len(s) < 20:
        return "unknown"
    skew = stats.skew(s)
    kurt = stats.kurtosis(s)
    # Use skew thresholds
    if abs(skew) < 0.5 and abs(kurt) < 1:
        return "normal"
    if skew > 1:
        return "right_skewed"
    if skew < -1:
        return "left_skewed"
    if kurt > 2:
        return "heavy_tailed"
    return "moderate_skew"


def profile_column(series: pd.Series, epsilon: Optional[float] = None) -> Dict[str, Any]:
    """Compute statistical profile for a single column.

    If epsilon is provided, Laplace noise will be added to counts/means via privacy layer.
    For now, we compute non-private stats; privacy layer wraps this.
    """
    col_type = infer_column_type(series)
    n_total = len(series)
    n_missing = series.isna().sum()
    n_unique = series.nunique(dropna=True)
    profile: Dict[str, Any] = {
        "name": series.name,
        "inferred_type": col_type,
        "n_total": n_total,
        "n_missing": int(n_missing),
        "missing_rate": float(n_missing / n_total) if n_total else 0,
        "n_unique": int(n_unique),
        "uniqueness": float(n_unique / n_total) if n_total else 0,
    }

    s = series.dropna()
    if col_type == "numeric":
        profile.update({
            "mean": float(s.mean()) if len(s) else None,
            "std": float(s.std()) if len(s) else None,
            "min": float(s.min()) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
            "median": float(s.median()) if len(s) else None,
            "q25": float(s.quantile(0.25)) if len(s) else None,
            "q75": float(s.quantile(0.75)) if len(s) else None,
            "skew": float(stats.skew(s)) if len(s) >= 3 else None,
            "kurtosis": float(stats.kurtosis(s)) if len(s) >= 4 else None,
            "distribution_shape": _distribution_shape(s) if len(s) >= 20 else "unknown",
        })
    elif col_type == "categorical":
        vc = s.value_counts(normalize=True).head(20)
        profile["top_values"] = vc.to_dict()
        profile["value_counts"] = s.value_counts().head(50).to_dict()
        # Entropy
        probs = s.value_counts(normalize=True).values
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        profile["entropy"] = float(entropy)
    elif col_type == "datetime":
        try:
            dt = pd.to_datetime(s, errors="coerce").dropna()
            if len(dt):
                profile.update({
                    "min": str(dt.min()),
                    "max": str(dt.max()),
                    "range_days": int((dt.max() - dt.min()).days),
                    "day_of_week_dist": dt.dt.dayofweek.value_counts(normalize=True).to_dict(),
                    "month_dist": dt.dt.month.value_counts(normalize=True).to_dict(),
                })
        except Exception:
            pass
    elif col_type == "boolean":
        vc = s.value_counts(normalize=True).to_dict()
        profile["value_distribution"] = {str(k): float(v) for k, v in vc.items()}
    elif col_type == "text":
        lens = s.astype(str).str.len()
        profile.update({
            "avg_length": float(lens.mean()) if len(lens) else None,
            "min_length": int(lens.min()) if len(lens) else None,
            "max_length": int(lens.max()) if len(lens) else None,
        })
    elif col_type == "id":
        profile["is_primary_key_candidate"] = bool(n_unique == n_total and n_total > 0)
        # check monotonic
        try:
            profile["is_monotonic"] = bool(s.is_monotonic_increasing)
        except Exception:
            profile["is_monotonic"] = False

    return profile


def infer_schema(df: pd.DataFrame) -> Dict[str, str]:
    """Infer schema: column name -> inferred type."""
    return {col: infer_column_type(df[col]) for col in df.columns}


def profile_dataframe(df: pd.DataFrame, epsilon: Optional[float] = None) -> Dict[str, Any]:
    """Profile entire dataframe.

    Args:
        df: input dataframe
        epsilon: if provided, will be passed to privacy layer elsewhere; here we just record it.
    Returns:
        dict with column profiles and table-level stats.
    """
    schema = infer_schema(df)
    columns = {col: profile_column(df[col]) for col in df.columns}
    # Override inferred type with schema to ensure consistency
    for col, typ in schema.items():
        columns[col]["inferred_type"] = typ

    # Detect primary keys
    pks = detect_primary_keys(df, schema)

    table_profile = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "schema": schema,
        "columns": columns,
        "primary_keys": pks,
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
    }
    return table_profile


def detect_primary_keys(df: pd.DataFrame, schema: Optional[Dict[str, str]] = None) -> List[str]:
    """Detect primary key candidates."""
    if schema is None:
        schema = infer_schema(df)
    pks = []
    for col, typ in schema.items():
        if typ == "id":
            # Check uniqueness
            if df[col].nunique() == len(df) and df[col].notna().all():
                pks.append(col)
        else:
            # Also check uniqueness even if not typed as id
            if df[col].nunique() == len(df) and df[col].notna().all():
                # Heuristic: short column name or single column unique => potential PK
                # Only add if name suggests id or table has no other PK
                if "id" in col.lower() or len(pks) == 0:
                    # Verify not a high-cardinality text
                    if typ not in ("text",):
                        pks.append(col)
    return pks


def detect_foreign_keys(tables: Dict[str, pd.DataFrame],
                        schemas: Optional[Dict[str, Dict[str, str]]] = None) -> List[Dict[str, Any]]:
    """Detect foreign key-like relationships between tables.

    Heuristic: for each column in each table, check if its values are subset of
    a primary key column in another table, with high overlap.

    Returns list of dicts: {from_table, from_column, to_table, to_column, coverage}
    """
    if schemas is None:
        schemas = {name: infer_schema(df) for name, df in tables.items()}

    # Find PKs per table
    pks_per_table: Dict[str, List[str]] = {}
    for name, df in tables.items():
        pks_per_table[name] = detect_primary_keys(df, schemas[name])

    relationships = []
    for from_table, from_df in tables.items():
        for col in from_df.columns:
            from_vals = set(from_df[col].dropna().unique())
            if not from_vals:
                continue
            for to_table, to_df in tables.items():
                if from_table == to_table:
                    continue
                for pk in pks_per_table.get(to_table, []):
                    to_vals = set(to_df[pk].dropna().unique())
                    if not to_vals:
                        continue
                    overlap = len(from_vals & to_vals)
                    coverage = overlap / len(from_vals) if from_vals else 0
                    col_lower = col.lower()
                    pk_lower = pk.lower()
                    to_lower = to_table.lower().rstrip("s")
                    # Name similarity heuristic: FK column often equals PK name or contains table name
                    name_match = (col_lower == pk_lower) or (to_lower in col_lower) or (pk_lower in col_lower)
                    # Require high coverage AND name match to avoid spurious sequential overlaps
                    if coverage > 0.9 and len(from_vals) > 5 and name_match:
                        relationships.append({
                            "from_table": from_table,
                            "from_column": col,
                            "to_table": to_table,
                            "to_column": pk,
                            "coverage": float(coverage),
                            "from_unique": len(from_vals),
                            "to_unique": len(to_vals),
                        })
                    elif to_lower in col_lower and coverage > 0.5:
                        if coverage > 0.5 and not any(r["from_column"] == col and r["to_column"] == pk and r["from_table"] == from_table and r["to_table"] == to_table for r in relationships):
                            relationships.append({
                                "from_table": from_table,
                                "from_column": col,
                                "to_table": to_table,
                                "to_column": pk,
                                "coverage": float(coverage),
                                "from_unique": len(from_vals),
                                "to_unique": len(to_vals),
                            })
    return relationships
