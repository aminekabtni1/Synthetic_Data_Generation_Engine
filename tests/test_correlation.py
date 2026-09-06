import pandas as pd
import numpy as np
from syndata.correlation.relationships import compute_correlations, categorical_dependencies, cramers_v

def test_compute_correlations_strong():
    rng = np.random.default_rng(0)
    x = rng.normal(0,1,200)
    y = x*2 + rng.normal(0,0.2,200)  # strong correlation
    z = rng.normal(0,1,200)  # weak
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    res = compute_correlations(df)
    strong = res["strong_correlations"]
    # x-y should be strong
    pair = [s for s in strong if set([s["col1"], s["col2"]])=={"x","y"}]
    assert len(pair)==1
    assert abs(pair[0]["pearson"]) > 0.9
    # x-z should not be strong (maybe not in list)
    pair_xz = [s for s in strong if set([s["col1"], s["col2"]])=={"x","z"}]
    assert len(pair_xz)==0

def test_cramers_v():
    # Perfect association: x==y
    df = pd.DataFrame({"a": ["x","x","y","y"]*25, "b": ["x","x","y","y"]*25})
    v = cramers_v(df["a"], df["b"])
    assert v > 0.9
    # No association random
    rng = np.random.default_rng(1)
    df2 = pd.DataFrame({"a": rng.choice(["x","y"],100), "b": rng.choice(["p","q"],100)})
    v2 = cramers_v(df2["a"], df2["b"])
    assert v2 < 0.3

def test_categorical_dependencies():
    rng = np.random.default_rng(2)
    # city depends on country
    country = rng.choice(["USA","UK"], 200)
    city = [rng.choice(["NYC","LA"]) if c=="USA" else rng.choice(["London","Manchester"]) for c in country]
    df = pd.DataFrame({"country": country, "city": city})
    res = categorical_dependencies(df)
    assert len(res["dependencies"])>0
    assert res["dependencies"][0]["cramers_v"] > 0.5
