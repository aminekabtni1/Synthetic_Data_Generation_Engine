import pandas as pd
import numpy as np
from syndata.evaluation.metrics import ks_test_numeric, chi_square_categorical, correlation_preservation, evaluate_quality

def test_ks_identical():
    s = pd.Series([1,2,3,4,5]*20)
    res = ks_test_numeric(s, s)
    assert res["ks_stat"] == 0
    assert res["p_value"] > 0.9

def test_ks_different():
    r = pd.Series(np.random.normal(0,1,200))
    s = pd.Series(np.random.normal(5,1,200))
    res = ks_test_numeric(r, s)
    assert res["ks_stat"] > 0.5
    assert res["p_value"] < 0.05

def test_chi_identical():
    r = pd.Series(["A"]*50 + ["B"]*50)
    s = pd.Series(["A"]*50 + ["B"]*50)
    res = chi_square_categorical(r, s)
    assert res["chi2"] < 1
    assert res["p_value"] > 0.5

def test_correlation_preservation_perfect():
    rng = np.random.default_rng(0)
    x = rng.normal(0,1,200)
    y = x*2 + rng.normal(0,0.1,200)
    df = pd.DataFrame({"x": x, "y": y})
    # Synthetic same as real should have high preservation
    res = correlation_preservation(df, df)
    assert res["preservation_score"] > 95
    assert res["mae"] < 0.05

def test_evaluate_quality_end_to_end():
    rng = np.random.default_rng(42)
    n=200
    df_real = pd.DataFrame({
        "age": rng.normal(40,10,n).astype(int),
        "income": rng.normal(50000,15000,n),
        "cat": rng.choice(["A","B","C"], n),
    })
    # Synthetic close but not identical
    df_synth = pd.DataFrame({
        "age": rng.normal(40,10,n).astype(int),
        "income": rng.normal(50000,15000,n),
        "cat": rng.choice(["A","B","C"], n, p=[0.5,0.3,0.2]),
    })
    res = evaluate_quality(df_real, df_synth)
    assert "overall_quality_score" in res
    assert 0 <= res["overall_quality_score"] <= 100
    assert "per_column" in res
    assert "correlation_preservation" in res
    assert "tstr" in res
