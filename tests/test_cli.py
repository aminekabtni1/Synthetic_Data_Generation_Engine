import subprocess
import sys
import os
import pandas as pd
from pathlib import Path

def test_cli_generate(tmp_path):
    # Create small CSV
    df = pd.DataFrame({"age": [20,30,40,50,60]*10, "cat": ["A","B"]*25})
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    rep = tmp_path / "report.html"
    df.to_csv(inp, index=False)
    cmd = [sys.executable, "-m", "syndata.interface.cli", "generate", "--input", str(inp), "--output", str(out), "--rows", "20", "--epsilon", "1.0", "--report", str(rep)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, result.stderr
    assert out.exists()
    out_df = pd.read_csv(out)
    assert len(out_df)==20
    assert rep.exists()
