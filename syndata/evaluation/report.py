"""HTML Report generation."""
from __future__ import annotations
import pandas as pd
import json
from typing import Dict, Any
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _plot_histogram(real: pd.Series, synth: pd.Series, col: str) -> str:
    plt.figure(figsize=(6, 3))
    try:
        r = pd.to_numeric(real, errors="coerce").dropna()
        s = pd.to_numeric(synth, errors="coerce").dropna()
        plt.hist(r, bins=30, alpha=0.5, label="Real", density=True)
        plt.hist(s, bins=30, alpha=0.5, label="Synthetic", density=True)
        plt.title(f"Distribution: {col}")
        plt.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        plt.close()
        return ""


def _plot_categorical(real: pd.Series, synth: pd.Series, col: str) -> str:
    plt.figure(figsize=(6, 3))
    try:
        r = real.value_counts(normalize=True).head(10)
        s = synth.value_counts(normalize=True).head(10)
        cats = list(set(r.index) | set(s.index))
        x = np.arange(len(cats))
        width = 0.35
        r_vals = [r.get(c, 0) for c in cats]
        s_vals = [s.get(c, 0) for c in cats]
        plt.bar(x - width/2, r_vals, width, label="Real", alpha=0.7)
        plt.bar(x + width/2, s_vals, width, label="Synthetic", alpha=0.7)
        plt.xticks(x, [str(c)[:10] for c in cats], rotation=30, ha="right")
        plt.title(f"Categorical: {col}")
        plt.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        plt.close()
        return ""


def generate_html_report(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                         profile: Dict[str, Any] = None,
                         quality: Dict[str, Any] = None,
                         privacy: Dict[str, Any] = None,
                         correlations: Dict[str, Any] = None) -> str:
    """Generate HTML quality+privacy report."""
    from syndata.profiling.schema import infer_schema
    schema = infer_schema(real_df)
    # Build plots
    plots = {}
    for col, typ in list(schema.items())[:6]:  # limit 6 plots
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        if typ == "numeric":
            plots[col] = _plot_histogram(real_df[col], synth_df[col], col)
        elif typ == "categorical":
            plots[col] = _plot_categorical(real_df[col], synth_df[col], col)

    # Quality summary - handle None
    overall = quality.get("overall_quality_score", 0) if quality else 0
    if overall is None:
        overall = 0
    stat_sim = quality.get("overall_statistical_similarity", 0)*100 if quality else 0
    if stat_sim is None:
        stat_sim = 0
    corr_score = quality.get("correlation_preservation", {}).get("preservation_score", 0) if quality else 0
    if corr_score is None:
        corr_score = 0
    tstr_score = quality.get("tstr", {}).get("utility_score", 0) if quality else 0
    if tstr_score is None:
        tstr_score = 0

    privacy_score = privacy.get("risk_score", 0) if privacy else 0
    privacy_level = privacy.get("risk_level", "unknown") if privacy else "unknown"

    html_plots = ""
    for col, b64 in plots.items():
        if b64:
            html_plots += f'<div style="margin:10px;display:inline-block"><img src="{b64}" style="max-width:500px"/></div>'

    per_col_rows = ""
    if quality and "per_column" in quality:
        for col, metrics in quality["per_column"].items():
            sim = metrics.get("similarity", 0) or 0
            ks = metrics.get('ks_stat', metrics.get('tv_distance',''))
            if ks is None or ks == '':
                ks = ''
            else:
                try:
                    ks = f"{float(ks):.3f}"
                except Exception:
                    ks = str(ks)
            per_col_rows += f"<tr><td>{col}</td><td>{metrics.get('type','')}</td><td>{sim:.3f}</td><td>{ks}</td></tr>"

    corr_rows = ""
    if correlations and "numeric_correlations" in correlations:
        strong = correlations["numeric_correlations"].get("strong_correlations", [])[:5]
        for s in strong:
            corr_rows += f"<tr><td>{s['col1']} vs {s['col2']}</td><td>{s['pearson']:.3f}</td><td>{s['strength']}</td></tr>"
        if not strong:
            corr_rows = "<tr><td colspan=3>No strong correlations detected</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Synthetic Data Quality Report</title>
<style>
 body{{font-family:Inter,Arial,sans-serif;margin:30px;background:#f8fafc;color:#1e293b}}
 h1{{color:#0f172a}}
 .card{{background:white;border-radius:12px;padding:20px;margin:15px 0;box-shadow:0 2px 8px rgba(0,0,0,0.08)}}
 .metric{{display:inline-block;margin:10px 20px;text-align:center}}
 .metric .val{{font-size:28px;font-weight:700;color:#2563eb}}
 .badge{{padding:4px 10px;border-radius:20px;font-size:12px;color:white}}
 .low{{background:#16a34a}} .medium{{background:#eab308;color:#422006}} .high{{background:#dc2626}}
 table{{width:100%;border-collapse:collapse}} th,td{{padding:8px;border-bottom:1px solid #e2e8f0;text-align:left}}
 th{{background:#f1f5f9}}
</style></head><body>
<h1> Synthetic Data Quality & Privacy Report</h1>
<div class="card">
  <h2>Overall Scores</h2>
  <div class="metric"><div class="val">{overall:.1f}</div><div>Overall Quality /100</div></div>
  <div class="metric"><div class="val">{stat_sim:.1f}</div><div>Statistical Similarity</div></div>
  <div class="metric"><div class="val">{corr_score:.1f}</div><div>Correlation Preservation</div></div>
  <div class="metric"><div class="val">{tstr_score:.1f}</div><div>ML Utility (TSTR)</div></div>
  <div class="metric"><div class="val">{privacy_score:.1f}</div><div>Privacy Risk <span class="badge {privacy_level}">{privacy_level}</span></div></div>
</div>

<div class="card">
  <h2>Distribution Comparison</h2>
  {html_plots}
</div>

<div class="card">
  <h2>Per-Column Similarity</h2>
  <table><tr><th>Column</th><th>Type</th><th>Similarity (0-1)</th><th>KS / TV</th></tr>
  {per_col_rows}
  </table>
</div>

<div class="card">
  <h2>Strong Correlations (Real Data)</h2>
  <table><tr><th>Pair</th><th>Pearson</th><th>Strength</th></tr>
  {corr_rows}
  </table>
  <p>Correlation preservation MAE: {quality.get('correlation_preservation',{}).get('mae','N/A') if quality else 'N/A'}</p>
</div>

<div class="card">
  <h2>Privacy Report</h2>
  <p><b>Risk score:</b> {privacy_score:.1f}/100 ({privacy_level})</p>
  <p>{privacy.get('interpretation','') if privacy else ''}</p>
  <p><b>Flagged close records:</b> {privacy.get('nearest_neighbor',{}).get('flagged_count',0) if privacy else 0} / {len(synth_df)} ({privacy.get('nearest_neighbor',{}).get('flagged_ratio',0)*100:.1f}% if privacy else 0%)</p>
  <p><b>Mean distance synthetic→real:</b> {privacy.get('nearest_neighbor',{}).get('mean_distance',0):.3f} vs real→real {privacy.get('nearest_neighbor',{}).get('real_real_mean',0):.3f} if privacy else 0</p>
  <h4>Riskiest columns</h4>
  <ul>
    {''.join(f"<li>{c['column']}: delta {c['delta_distance']:.3f}</li>" for c in (privacy.get('riskiest_columns',[]) if privacy else []))}
  </ul>
</div>

<div class="card">
  <h2>TSTR (Train Synthetic Test Real)</h2>
  <pre>{json.dumps(quality.get('tstr',{}) if quality else {}, indent=2)}</pre>
</div>

<div class="card"><h2>Dataset Info</h2>
<p>Real rows: {len(real_df)}, Synthetic rows: {len(synth_df)}, Columns: {len(real_df.columns)}</p>
<p>Profile: {json.dumps(profile.get('schema',{}) if profile else {}, indent=2)}</p>
</div>

</body></html>
"""
    return html
