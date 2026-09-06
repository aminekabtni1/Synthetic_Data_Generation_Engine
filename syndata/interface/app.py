"""FastAPI backend for Synthetic Data web interface."""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io
import json
import base64
import traceback
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from syndata.profiling.schema import profile_dataframe
from syndata.correlation.relationships import analyze_all
from syndata.generation.copula import GaussianCopulaSynthesizer
from syndata.privacy.mechanisms import privatize_profile
from syndata.privacy.risk import membership_inference_risk
from syndata.evaluation.metrics import evaluate_quality
from syndata.evaluation.report import generate_html_report

app = FastAPI(title="Synthetic Data Generation Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for demo (single user)
STORE = {
    "real_df": None,
    "profile": None,
    "priv_profile": None,
    "synth_df": None,
    "quality": None,
    "privacy": None,
    "correlations": None,
}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Try CSV
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset")
        # Limit to 50k rows for demo memory
        if len(df) > 50000:
            df = df.head(50000)

        profile = profile_dataframe(df)
        corr = analyze_all(df, profile["schema"])

        STORE["real_df"] = df
        STORE["profile"] = profile
        STORE["correlations"] = corr
        STORE["synth_df"] = None
        STORE["quality"] = None
        STORE["privacy"] = None

        # Return summary
        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            "schema": profile["schema"],
            "profile": {k: v for k, v in profile.items() if k not in ("columns",)},
            "columns_profile": profile["columns"],
            "correlations": corr,
            "preview": df.head(10).to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate(epsilon: float = Form(1.0), rows: int = Form(1000), seed: int = Form(42)):
    df = STORE.get("real_df")
    profile = STORE.get("profile")
    if df is None or profile is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet")

    try:
        # Privatize profile for display (not actually used to perturb generation in this simplified version - epsilon influences post-filtering)
        # In a full DP engine, the copula fitting would use privatized stats. Here we log privacy budget.
        if epsilon != float("inf"):
            priv_profile = privatize_profile(profile, epsilon=epsilon)
        else:
            priv_profile = profile
        STORE["priv_profile"] = priv_profile

        synth_model = GaussianCopulaSynthesizer(random_state=seed)
        synth_model.fit(df)
        synth_df = synth_model.sample(rows)
        STORE["synth_df"] = synth_df

        quality = evaluate_quality(df, synth_df, profile["schema"])
        privacy = membership_inference_risk(df, synth_df, profile["schema"])

        STORE["quality"] = quality
        STORE["privacy"] = privacy

        return {
            "n_rows": len(synth_df),
            "preview": synth_df.head(10).to_dict(orient="records"),
            "quality": quality,
            "privacy": privacy,
            "columns": list(synth_df.columns),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download")
def download():
    synth_df = STORE.get("synth_df")
    if synth_df is None:
        raise HTTPException(status_code=400, detail="No synthetic data generated")
    stream = io.StringIO()
    synth_df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        io.BytesIO(stream.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=synthetic.csv"},
    )


@app.get("/api/report")
def report():
    df = STORE.get("real_df")
    synth_df = STORE.get("synth_df")
    if df is None or synth_df is None:
        raise HTTPException(status_code=400, detail="Need both real and synthetic")
    html = generate_html_report(
        df, synth_df,
        profile=STORE.get("priv_profile") or STORE.get("profile"),
        quality=STORE.get("quality"),
        privacy=STORE.get("privacy"),
        correlations=STORE.get("correlations"),
    )
    return HTMLResponse(content=html)


@app.get("/api/plot/{col}")
def plot_col(col: str, typ: str = "hist"):
    df = STORE.get("real_df")
    synth_df = STORE.get("synth_df")
    if df is None or synth_df is None or col not in df.columns:
        raise HTTPException(status_code=404, detail="Column not found")
    plt.figure(figsize=(6, 3))
    try:
        if typ == "hist" or STORE["profile"]["schema"].get(col) == "numeric":
            r = pd.to_numeric(df[col], errors="coerce").dropna()
            s = pd.to_numeric(synth_df[col], errors="coerce").dropna()
            plt.hist(r, bins=30, alpha=0.5, label="Real", density=True)
            plt.hist(s, bins=30, alpha=0.5, label="Synthetic", density=True)
            plt.legend()
            plt.title(col)
        else:
            r = df[col].value_counts(normalize=True).head(10)
            s = synth_df[col].value_counts(normalize=True).head(10)
            cats = list(set(r.index) | set(s.index))
            x = np.arange(len(cats))
            width = 0.35
            plt.bar(x - width/2, [r.get(c, 0) for c in cats], width, label="Real", alpha=0.7)
            plt.bar(x + width/2, [s.get(c, 0) for c in cats], width, label="Synthetic", alpha=0.7)
            plt.xticks(x, [str(c)[:10] for c in cats], rotation=30, ha="right")
            plt.legend()
            plt.title(col)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        plt.close()
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend static
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    # Serve the frontend HTML
    html_path = Path(__file__).parent.parent.parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Synthetic Data Engine API is running. Use /docs for API docs.</h1>")
