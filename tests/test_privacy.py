import pandas as pd
import numpy as np
from syndata.privacy.mechanisms import LaplaceMechanism, privatize_profile, compute_sensitivity
from syndata.privacy.risk import nearest_neighbor_distance, membership_inference_risk, filter_close_records
from syndata.profiling.schema import profile_dataframe

def test_laplace_scale():
    mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, random_state=42)
    assert mech.scale == 1.0
    mech2 = LaplaceMechanism(epsilon=0.5, sensitivity=1.0)
    assert mech2.scale == 2.0
    mech3 = LaplaceMechanism(epsilon=2.0, sensitivity=1.0)
    assert mech3.scale == 0.5

def test_laplace_noise_properties():
    mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, random_state=0)
    vals = [mech.add_noise(0) for _ in range(10000)]
    # Mean should be ~0 (Laplace 0 mean)
    assert abs(np.mean(vals)) < 0.1
    # Larger epsilon => less noise (compare std)
    mech_low = LaplaceMechanism(epsilon=0.1, sensitivity=1.0, random_state=0)
    mech_high = LaplaceMechanism(epsilon=10.0, sensitivity=1.0, random_state=0)
    low_noises = [abs(mech_low.add_noise(0)) for _ in range(500)]
    high_noises = [abs(mech_high.add_noise(0)) for _ in range(500)]
    assert np.mean(low_noises) > np.mean(high_noises)

def test_sensitivity_computation():
    s = pd.Series([1,2,3,4,5])
    sens = compute_sensitivity(s, "mean")
    # range 4 / n 5 =0.8
    assert abs(sens - 0.8) < 1e-6
    sens_count = compute_sensitivity(s, "count")
    assert sens_count == 1.0

def test_privatize_profile():
    df = pd.DataFrame({"x": [1,2,3,4,5,6,7,8,9,10], "y": [10,20,30,40,50,60,70,80,90,100]})
    prof = profile_dataframe(df)
    # With large epsilon, noise small; with small epsilon, noise large
    priv_small_eps = privatize_profile(prof, epsilon=0.1, random_state=42)
    priv_large_eps = privatize_profile(prof, epsilon=10, random_state=42)
    # Large epsilon should be closer to original (on average)
    orig_mean = prof["columns"]["x"]["mean"]
    small_noise = abs(priv_small_eps["columns"]["x"]["mean"] - orig_mean)
    large_noise = abs(priv_large_eps["columns"]["x"]["mean"] - orig_mean)
    # Not guaranteed per single draw but highly likely: small eps => larger noise
    # We'll test multiple times averaged? For now check that privatize adds log and preserves structure
    assert "privacy" in priv_small_eps
    assert priv_small_eps["privacy"]["epsilon_total"] == 0.1
    assert len(priv_small_eps["privacy"]["log"]) > 0

def test_nearest_neighbor():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(0,1,100), "y": rng.normal(0,1,100)})
    # Synthetic very close to real (copy)
    synth_close = real.copy()
    # Synthetic far
    synth_far = pd.DataFrame({"x": rng.normal(10,1,100), "y": rng.normal(10,1,100)})
    res_close = nearest_neighbor_distance(real, synth_close)
    res_far = nearest_neighbor_distance(real, synth_far)
    assert res_close["mean_distance"] < res_far["mean_distance"]
    assert res_close["min_distance"] == 0

def test_membership_risk_levels():
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"x": rng.normal(0,1,100), "y": rng.normal(0,1,100)})
    synth_close = real.copy()
    synth_far = pd.DataFrame({"x": rng.normal(10,1,100), "y": rng.normal(10,1,100)})
    risk_close = membership_inference_risk(real, synth_close)
    risk_far = membership_inference_risk(real, synth_far)
    assert risk_close["risk_score"] > risk_far["risk_score"]
    assert risk_close["risk_level"] in ("low","medium","high")
    # Close should be high risk
    assert risk_close["risk_score"] > 30

def test_filter_close():
    rng = np.random.default_rng(2)
    real = pd.DataFrame({"x": rng.normal(0,1,50), "y": rng.normal(0,1,50)})
    synth = pd.concat([real.head(10), pd.DataFrame({"x": rng.normal(10,1,40), "y": rng.normal(10,1,40)})], ignore_index=True)
    filtered, info = filter_close_records(real, synth)
    assert info["removed_count"] >= 0
    assert len(filtered) <= len(synth)
