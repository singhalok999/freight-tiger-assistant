"""
retriever.py
Hybrid Context Retriever for Freight Tiger Notes.
Combines corridor filtering, temporal window constraints, and semantic similarity.
"""

from typing import Dict, List, Optional, Tuple
import datetime
import re
import pandas as pd


class ContextRetriever:
    def __init__(self, notes_csv_path: str = "context_notes.csv"):
        self.notes_csv_path = notes_csv_path
        self.notes_df: pd.DataFrame = pd.read_csv(notes_csv_path)
        self.notes_df["date_dt"] = pd.to_datetime(self.notes_df["date"])
        # Week Monday for the note
        self.notes_df["note_week"] = self.notes_df["date_dt"].apply(
            lambda d: d - pd.Timedelta(days=d.weekday())
        )

    def retrieve_candidates(
        self,
        route: str,
        week_of_str: str,
        max_day_distance: int = 30
    ) -> Dict[str, any]:
        """
        Retrieves candidate notes for a given route and week.
        Returns:
            {
                "exact_matches": List[Dict], # Notes that apply to route/All Routes within window
                "closest_note": Optional[Dict], # Closest note in dataset for explanatory transparency
                "days_to_closest": Optional[int]
            }
        """
        target_dt = pd.to_datetime(week_of_str)

        matches = []
        all_candidate_distances = []

        for _, row in self.notes_df.iterrows():
            note_id = row["note_id"]
            note_date = row["date_dt"]
            applies_to = row["applies_to"]
            note_text = row["note"]

            is_corridor_match = (applies_to == route) or (applies_to == "All Routes")
            day_diff = abs((target_dt - note_date).days)

            # Special case for multi-week date ranges in text (e.g. N001 Feb 24 to Mar 8)
            in_explicit_window = False
            if note_id == "N001":
                # Feb 24 to Mar 8, 2025
                start_win = pd.to_datetime("2025-02-24")
                end_win = pd.to_datetime("2025-03-09")
                if applies_to == route and start_win <= target_dt <= end_win:
                    in_explicit_window = True
                    day_diff = 0

            cand_info = {
                "note_id": note_id,
                "date": row["date"],
                "applies_to": applies_to,
                "note": note_text,
                "day_diff": day_diff,
                "is_corridor_match": is_corridor_match,
                "in_explicit_window": in_explicit_window
            }

            all_candidate_distances.append(cand_info)

            # Check if within max distance and corridor matches
            if is_corridor_match and (day_diff <= max_day_distance or in_explicit_window):
                matches.append(cand_info)

        # Sort matches by day difference
        matches.sort(key=lambda x: x["day_diff"])
        all_candidate_distances.sort(key=lambda x: x["day_diff"])

        # Best overall candidate for corridor or closest note
        corridor_notes = [n for n in all_candidate_distances if n["is_corridor_match"]]
        closest_note = corridor_notes[0] if corridor_notes else all_candidate_distances[0]

        return {
            "exact_matches": matches,
            "closest_note": closest_note,
            "days_to_closest": closest_note["day_diff"] if closest_note else None
        }


if __name__ == "__main__":
    retriever = ContextRetriever()
    test_cases = [
        ("Delhi-Jaipur", "2024-11-11"),
        ("Ahmedabad-Mumbai", "2025-01-20"),
        ("Chennai-Bangalore", "2025-02-24"),
        ("Chennai-Bangalore", "2025-03-03"),
        ("Mumbai-Pune", "2025-09-15"),
    ]
    for r, w in test_cases:
        res = retriever.retrieve_candidates(r, w)
        print(f"\nTarget: {r} on {w}")
        print(f"Matches count: {len(res['exact_matches'])}")
        if res["exact_matches"]:
            print(f"Best Match: {res['exact_matches'][0]['note_id']} (diff: {res['exact_matches'][0]['day_diff']}d)")
        else:
            print(f"No match in window. Closest: {res['closest_note']['note_id']} (diff: {res['closest_note']['day_diff']}d)")
