import pandas as pd
import numpy as np
from syndata.generation.copula import GaussianCopulaSynthesizer
from syndata.profiling.schema import profile_dataframe
from scipy.stats import pearsonr

def test_gaussian_copula_preserves_correlation():
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(50, 10, n)
    y = x * 0.8 + rng.normal(0, 5, n)  # correlated
    z = rng.normal(100, 20, n)  # independent-ish
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    synth = GaussianCopulaSynthesizer(random_state=42)
    synth.fit(df)
    df_synth = synth.sample(1000)
    # Check correlations preserved within tolerance
    real_corr = np.corrcoef(df["x"], df["y"])[0,1]
    synth_corr = np.corrcoef(df_synth["x"], df_synth["y"])[0,1]
    assert abs(real_corr - synth_corr) < 0.15, f"real {real_corr:.3f} synth {synth_corr:.3f}"
    # Check means close
    assert abs(df["x"].mean() - df_synth["x"].mean()) < 5
    assert len(df_synth)==1000
    assert set(df_synth.columns)=={"x","y","z"}

def test_mixed_types():
    df = pd.DataFrame({
        "age": [20,30,40,50,60]*20,
        "gender": ["M","F"]*50,
        "signup": pd.to_datetime(["2020-01-01"]*50 + ["2021-01-01"]*50),
        "is_prem": [True, False]*50,
        "id": list(range(100)),
    })
    synth = GaussianCopulaSynthesizer(random_state=1)
    synth.fit(df)
    out = synth.sample(50)
    assert len(out)==50
    assert set(out.columns)==set(df.columns)
    # IDs should be new unique
    assert out["id"].nunique()==50
    assert out["id"].min() > df["id"].max()

def test_categorical_distribution():
    df = pd.DataFrame({"cat": ["A"]*70 + ["B"]*30})
    synth = GaussianCopulaSynthesizer(random_state=0)
    synth.fit(df)
    out = synth.sample(1000)
    # Distribution approx preserved
    a_ratio = (out["cat"]=="A").mean()
    assert 0.6 < a_ratio < 0.8

def test_not_fitted_error():
    synth = GaussianCopulaSynthesizer()
    try:
        synth.sample(10)
        assert False, "should raise"
    except RuntimeError:
        pass
