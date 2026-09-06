import pandas as pd
import numpy as np
from syndata.profiling.schema import infer_column_type, profile_dataframe, detect_primary_keys, detect_foreign_keys

def test_infer_types():
    df = pd.DataFrame({
        "id": [1,2,3,4,5],
        "age": [20,30,40,50,60],
        "gender": ["M","F","M","F","M"],
        "signup": pd.to_datetime(["2020-01-01","2020-02-01","2020-03-01","2020-04-01","2020-05-01"]),
        "is_premium": [True, False, True, False, True],
        "notes": ["hello world this is a long text " *2]*5,  # text
    })
    # id with unique 5 but n=5 small, may not be id due to threshold; test after more rows
    assert infer_column_type(df["age"]) == "numeric"
    assert infer_column_type(df["gender"]) == "categorical"
    assert infer_column_type(df["signup"]) == "datetime"
    assert infer_column_type(df["is_premium"]) == "boolean"

def test_profile_numeric():
    df = pd.DataFrame({"x": [1,2,3,4,5,6,7,8,9,10]})
    prof = profile_dataframe(df)
    col = prof["columns"]["x"]
    assert col["inferred_type"] == "numeric"
    assert col["mean"] == 5.5
    assert col["distribution_shape"] in ("normal","moderate_skew","unknown")

def test_primary_keys():
    df = pd.DataFrame({"customer_id": [1,2,3], "val": [10,20,30]})
    pks = detect_primary_keys(df)
    assert "customer_id" in pks

def test_foreign_keys():
    customers = pd.DataFrame({"customer_id": [1,2,3], "name": ["a","b","c"]})
    orders = pd.DataFrame({"order_id": [1,2,3], "customer_id": [1,2,1]})
    rels = detect_foreign_keys({"customers": customers, "orders": orders})
    assert any(r["from_column"]=="customer_id" and r["to_table"]=="customers" for r in rels)
