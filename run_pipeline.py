"""
run_pipeline.py
Main pipeline orchestrator for Freight Tiger Shipping Cost Assistant.
Executes end-to-end:
1. Ingestion of shipment records & context notes
2. Analytical calculation of weekly CPTK, trailing 8-week baseline, peer baseline
3. Candidate anomaly identification
4. Anti-hallucination note retrieval & reasoning
5. Exporting exact contract CSV matching sample_output_format.csv
6. Token usage, call count, and cost accounting report
"""

import os
import sys
import argparse
import json
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_engine import ShippingDataEngine
from retriever import ContextRetriever
from reasoner import AntiHallucinationReasoner, TokenCostTracker


def run_pipeline(
    shipment_csv: str = "shipment_records.csv",
    notes_csv: str = "context_notes.csv",
    output_csv: str = "output.csv",
    mode: str = "candidates",  # "candidates" or "all"
    own_threshold: float = 15.0,
    peer_threshold: float = 20.0,
) -> pd.DataFrame:
    print("=" * 70)
    print(" FREIGHT TIGER AI SHIPPING COST ASSISTANT: FULL RUN PIPELINE ")
    print("=" * 70)

    # 1. Initialize modules
    print(f"[1/5] Loading datasets: '{shipment_csv}' and '{notes_csv}'...")
    data_engine = ShippingDataEngine(shipment_csv)
    retriever = ContextRetriever(notes_csv)
    tracker = TokenCostTracker(model_name="gemini-1.5-flash")
    reasoner = AntiHallucinationReasoner(tracker=tracker)

    # 2. Compute metrics
    print(f"[2/5] Computing weekly cost per tonne-km & 8-week / peer baselines...")
    df_metrics = data_engine.compute_weekly_metrics(
        own_threshold=own_threshold,
        peer_threshold=peer_threshold
    )
    print(f"      Evaluated {len(df_metrics)} route-weeks across 7 corridors.")

    # 3. Determine evaluation subset
    if mode == "candidates":
        eval_df = df_metrics[df_metrics["is_candidate"]].copy()
        print(f"[3/5] Identified {len(eval_df)} candidate anomalous weeks meeting spike thresholds.")
    else:
        eval_df = df_metrics.copy()
        print(f"[3/5] Evaluating all {len(eval_df)} route-weeks in dataset.")

    # 4. Reason over each row
    print("[4/5] Executing Anti-Hallucination Retrieval & Grounded Reasoning...")
    output_rows = []

    for _, row in eval_df.iterrows():
        route = row["route"]
        week_of = row["week_of"]
        cptk = row["cost_per_tonne_km"]
        vown = row["vs_own_history"]
        vpeer = row["vs_similar_routes"]

        if row.get("is_candidate", False) or mode == "candidates":
            # Retrieve notes
            ret = retriever.retrieve_candidates(route, week_of)
            verdict_info = reasoner.evaluate_candidate(
                route=route,
                week_of=week_of,
                cost_per_tonne_km=cptk,
                vs_own_str=vown,
                vs_similar_str=vpeer,
                retrieval_result=ret
            )
            flagged = verdict_info["flagged"]
            matched_note_id = verdict_info["matched_note_id"]
            reason = verdict_info["reason"]
        else:
            # Normal row when mode == 'all'
            flagged = "No"
            matched_note_id = ""
            reason = "Cost per tonne-km is within normal historical and peer corridor baselines."

        output_rows.append({
            "route": route,
            "week_of": week_of,
            "cost_per_tonne_km": cptk,
            "vs_own_history": vown,
            "vs_similar_routes": vpeer,
            "flagged": flagged,
            "matched_note_id": matched_note_id,
            "reason": reason
        })

    final_df = pd.DataFrame(output_rows)

    # Ensure exact column ordering as per sample_output_format.csv contract
    column_order = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "flagged",
        "matched_note_id",
        "reason"
    ]
    final_df = final_df[column_order]

    # Save output
    final_df.to_csv(output_csv, index=False)
    print(f"[5/5] Saved final submission CSV to '{output_csv}' ({len(final_df)} rows).")

    # Also generate full dataset if candidates mode was used
    if mode == "candidates":
        full_output_path = "output_all_weeks.csv"
        # Run quickly for all weeks
        all_rows = []
        candidate_keys = set(zip(final_df["route"], final_df["week_of"]))
        candidate_dict = {
            (r["route"], r["week_of"]): r
            for _, r in final_df.iterrows()
        }

        for _, row in df_metrics.iterrows():
            key = (row["route"], row["week_of"])
            if key in candidate_dict:
                all_rows.append(candidate_dict[key].to_dict())
            else:
                all_rows.append({
                    "route": row["route"],
                    "week_of": row["week_of"],
                    "cost_per_tonne_km": row["cost_per_tonne_km"],
                    "vs_own_history": row["vs_own_history"],
                    "vs_similar_routes": row["vs_similar_routes"],
                    "flagged": "No",
                    "matched_note_id": "",
                    "reason": "Cost per tonne-km is within normal historical and peer corridor baselines."
                })
        pd.DataFrame(all_rows)[column_order].to_csv(full_output_path, index=False)
        print(f"      Also generated '{full_output_path}' with all {len(all_rows)} route-weeks.")

    # 5. Print Token & Cost Accounting Log
    cost_summary = tracker.get_summary()
    print("\n" + "=" * 70)
    print(" TOKEN USAGE & INFERENCE COST ACCOUNTING LOG ")
    print("=" * 70)
    print(f"  Target Model                 : {cost_summary['model_name']}")
    print(f"  Total LLM / Reasoning Calls  : {cost_summary['total_calls']}")
    print(f"  Total Input Tokens           : {cost_summary['total_input_tokens']}")
    print(f"  Total Output Tokens          : {cost_summary['total_output_tokens']}")
    print(f"  Total Combined Tokens        : {cost_summary['total_tokens']}")
    print(f"  Published Rate (Input)       : ${tracker.rate_input_per_million}/1M tokens")
    print(f"  Published Rate (Output)      : ${tracker.rate_output_per_million}/1M tokens")
    print(f"  Rough Cost for Full Run      : ${cost_summary['estimated_cost_usd']:.6f} USD")
    print("=" * 70)

    # Save cost log to JSON
    with open("token_cost_log.json", "w") as f:
        json.dump(cost_summary, f, indent=2)

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Freight Tiger Assistant Pipeline")
    parser.add_argument("--shipment_csv", default="shipment_records.csv")
    parser.add_argument("--notes_csv", default="context_notes.csv")
    parser.add_argument("--output_csv", default="output.csv")
    parser.add_argument("--mode", default="candidates", choices=["candidates", "all"])
    args = parser.parse_args()

    run_pipeline(
        shipment_csv=args.shipment_csv,
        notes_csv=args.notes_csv,
        output_csv=args.output_csv,
        mode=args.mode
    )
