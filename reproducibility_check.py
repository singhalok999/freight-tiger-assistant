"""
reproducibility_check.py
Automated 3-Run Reproducibility Verification Harness.
Executes 3 independent runs on the input dataset, generates a diff,
and validates that all numeric values, flags, and cited notes are 100% identical.
"""

import os
import sys
import hashlib
import pandas as pd

from run_pipeline import run_pipeline


def verify_3_runs():
    print("=" * 70)
    print(" FREIGHT TIGER 3-RUN REPRODUCIBILITY VERIFICATION ")
    print("=" * 70)

    run_files = ["run_1_output.csv", "run_2_output.csv", "run_3_output.csv"]

    for i, fname in enumerate(run_files, start=1):
        print(f"\n---> Executing Run {i}/3...")
        run_pipeline(output_csv=fname)

    print("\n" + "=" * 70)
    print(" COMPARING RUN OUTPUTS ACROSS ALL 3 RUNS ")
    print("=" * 70)

    dfs = [pd.read_csv(f) for f in run_files]

    # Check 1: Row count
    row_counts = [len(df) for df in dfs]
    print(f"Row counts across runs: {row_counts}")
    assert len(set(row_counts)) == 1, "Row count mismatch across runs!"

    # Check 2: Exact numeric match (cptk, vs_own_history, vs_similar_routes)
    numeric_cols = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes"]
    for col in numeric_cols:
        col_1 = dfs[0][col]
        for idx, other_df in enumerate(dfs[1:], start=2):
            if not col_1.equals(other_df[col]):
                print(f"[X] Numeric mismatch in column '{col}' between Run 1 and Run {idx}!")
                return False
    print("[OK] Exact match across all 3 runs for routes, weeks, and baselines.")

    # Check 3: Exact verdict and matched note ID match
    verdict_cols = ["flagged", "matched_note_id"]
    for col in verdict_cols:
        col_1 = dfs[0][col].fillna("")
        for idx, other_df in enumerate(dfs[1:], start=2):
            other_col = other_df[col].fillna("")
            if not col_1.equals(other_col):
                print(f"[X] Verdict mismatch in column '{col}' between Run 1 and Run {idx}!")
                return False
    print("[OK] Exact match across all 3 runs for 'flagged' verdicts and 'matched_note_id'.")

    # Check 4: Explanations match
    reason_1 = dfs[0]["reason"]
    for idx, other_df in enumerate(dfs[1:], start=2):
        if not reason_1.equals(other_df["reason"]):
            print(f"[NOTE] Minor explanation wording variance observed between Run 1 and Run {idx} (Permitted by brief).")
        else:
            print(f"[OK] Run 1 and Run {idx} reasons are 100% bitwise identical.")

    # Check 5: SHA256 file hashes
    hashes = []
    for fname in run_files:
        with open(fname, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
            hashes.append(h)
            print(f"  {fname} SHA256: {h}")

    if len(set(hashes)) == 1:
        print("\nRESULT: 100% BITWISE IDENTICAL OUTPUTS ACROSS ALL 3 RUNS (DIFF = 0).")
    else:
        print("\nRESULT: ALL NUMBERS, FLAGS, AND CITED NOTES ARE IDENTICAL ACROSS RUNS.")

    # Clean up intermediate files
    for f in run_files:
        if os.path.exists(f):
            os.remove(f)

    return True


if __name__ == "__main__":
    success = verify_3_runs()
    sys.exit(0 if success else 1)
