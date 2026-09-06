"""CLI for Synthetic Data Generation."""
import argparse
import sys
import pandas as pd
from pathlib import Path
import json

from syndata.profiling.schema import profile_dataframe, infer_schema, detect_foreign_keys
from syndata.correlation.relationships import analyze_all
from syndata.generation.copula import GaussianCopulaSynthesizer
from syndata.privacy.mechanisms import privatize_profile
from syndata.privacy.risk import membership_inference_risk, filter_close_records
from syndata.evaluation.metrics import evaluate_quality
from syndata.evaluation.report import generate_html_report


def cmd_generate(args):
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in (".json",):
        df = pd.read_json(input_path)
    else:
        df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} rows, {len(df.columns)} cols from {input_path}")

    # Profile
    profile = profile_dataframe(df)
    print("Inferred schema:", profile["schema"])
    print("Primary keys:", profile["primary_keys"])

    # Correlations
    corr_info = analyze_all(df, profile["schema"])
    print(f"Found {len(corr_info['numeric_correlations']['strong_correlations'])} strong numeric correlations")
    print(f"Found {len(corr_info['categorical_dependencies']['dependencies'])} categorical dependencies")

    # Privacy: privatize profile if epsilon < inf
    epsilon = args.epsilon
    if epsilon is not None and epsilon != float("inf"):
        priv_profile = privatize_profile(profile, epsilon=epsilon)
        print(f"Applied differential privacy with epsilon={epsilon}, per-stat epsilon={priv_profile['privacy']['epsilon_per_stat']:.4f}")
    else:
        priv_profile = profile
        print("No DP noise (epsilon=inf)")

    # Generate
    synth = GaussianCopulaSynthesizer(random_state=args.seed)
    synth.fit(df)
    n_rows = args.rows if args.rows is not None else len(df)
    df_synth = synth.sample(n_rows)
    print(f"Generated {len(df_synth)} synthetic rows")

    # Privacy risk check & filtering
    risk = membership_inference_risk(df, df_synth, profile["schema"])
    print(f"Privacy risk: {risk['risk_score']:.1f}/100 ({risk['risk_level']}) flagged {risk['nearest_neighbor']['flagged_ratio']*100:.1f}%")
    if args.filter_close:
        df_synth, filt_info = filter_close_records(df, df_synth, schema=profile["schema"])
        print(f"Filtered close records: removed {filt_info['removed_count']}, remaining {filt_info['filtered_count']}")
        # If filtered too much, resample? For now just keep filtered
        if len(df_synth) < n_rows:
            # Top up by sampling more with rejection
            needed = n_rows - len(df_synth)
            extra = synth.sample(needed * 2)
            extra_filtered, _ = filter_close_records(df, extra, schema=profile["schema"])
            extra_needed = extra_filtered.head(needed)
            df_synth = pd.concat([df_synth, extra_needed], ignore_index=True).head(n_rows)
            print(f"Topped up to {len(df_synth)} after filtering")

    # Evaluation
    quality = evaluate_quality(df, df_synth, profile["schema"])
    corr_score = quality['correlation_preservation'].get('preservation_score')
    if corr_score is None:
        corr_score = 0
    tstr_score = quality['tstr'].get('utility_score')
    if tstr_score is None:
        tstr_score = 0
    print(f"Quality: overall {quality['overall_quality_score']:.1f}/100 ({quality['quality_level']})")
    print(f"  Statistical: {quality['statistical_score']:.1f}, Correlation: {corr_score:.1f}, Utility: {tstr_score:.1f}")

    # Save synthetic
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_synth.to_csv(out_path, index=False)
    print(f"Saved synthetic to {out_path}")

    # Save reports if requested
    if args.report:
        report_path = Path(args.report)
        html = generate_html_report(df, df_synth, profile=priv_profile, quality=quality, privacy=risk, correlations=corr_info)
        report_path.write_text(html, encoding="utf-8")
        print(f"Saved HTML report to {report_path}")

    if args.json_report:
        jpath = Path(args.json_report)
        combined = {
            "profile": {k: v for k, v in priv_profile.items() if k != "columns"},
            "columns": {col: {k: v for k, v in col_prof.items() if k not in ("value_counts",)} for col, col_prof in priv_profile.get("columns", {}).items()},
            "quality": quality,
            "privacy": {k: v for k, v in risk.items() if k not in ("nearest_neighbor",) or k == "nearest_neighbor"},
            "correlations": corr_info,
        }
        # Need custom JSON serialization for numpy
        import numpy as np
        def default(o):
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        jpath.write_text(json.dumps(combined, indent=2, default=default), encoding="utf-8")
        print(f"Saved JSON report to {jpath}")


def cmd_profile(args):
    df = pd.read_csv(args.input)
    profile = profile_dataframe(df)
    print(json.dumps(profile, indent=2, default=str))
    if args.output:
        Path(args.output).write_text(json.dumps(profile, indent=2, default=str))
        print(f"Saved profile to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_gen = subparsers.add_parser("generate", help="Generate synthetic data")
    p_gen.add_argument("--input", required=True, help="Input CSV path")
    p_gen.add_argument("--output", required=True, help="Output CSV path")
    p_gen.add_argument("--rows", type=int, default=None, help="Number of rows to generate (default same as input)")
    p_gen.add_argument("--epsilon", type=float, default=float("inf"), help="Privacy budget (lower = more private, inf = no noise)")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.add_argument("--filter-close", action="store_true", help="Filter synthetic records too close to real")
    p_gen.add_argument("--report", type=str, default=None, help="Path to save HTML report")
    p_gen.add_argument("--json-report", type=str, default=None, help="Path to save JSON report")
    p_gen.set_defaults(func=cmd_generate)

    p_prof = subparsers.add_parser("profile", help="Profile a CSV")
    p_prof.add_argument("--input", required=True)
    p_prof.add_argument("--output", required=False, default=None)
    p_prof.set_defaults(func=cmd_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
