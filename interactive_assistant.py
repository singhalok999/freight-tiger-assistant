"""
interactive_assistant.py
Stretch Goal Implementation: Conversational Shipping Cost Assistant.
Enables operations and finance managers to query cost spikes and anomalies in plain English.
Example queries:
- "Why did Ahmedabad-Mumbai spike in January 2025?"
- "What happened on Chennai-Bangalore in March 2025?"
- "Why was Delhi-Jaipur flagged in November 2024?"
- "List all unexplained cost surges."
"""

import os
import sys
import argparse
import re
import pandas as pd


class InteractiveShippingAssistant:
    def __init__(
        self,
        output_csv: str = "output.csv",
        notes_csv: str = "context_notes.csv"
    ):
        self.output_csv = output_csv
        self.notes_csv = notes_csv
        self.df_results = pd.read_csv(output_csv)
        self.df_results["week_dt"] = pd.to_datetime(self.df_results["week_of"])
        self.notes_df = pd.read_csv(notes_csv)

    def answer_query(self, query: str) -> str:
        q_lower = query.lower()

        # Check for listing unexplained surges
        if "unexplained" in q_lower or "unjustified" in q_lower or "all flags" in q_lower:
            unexplained = self.df_results[self.df_results["flagged"] == "Yes"]
            lines = [f"Found {len(unexplained)} unexplained cost anomalies flagged for review:"]
            for _, r in unexplained.iterrows():
                lines.append(f"• {r['route']} (Week of {r['week_of']}): CPTK INR {r['cost_per_tonne_km']} ({r['vs_own_history']})")
                lines.append(f"  Note: {r['reason']}\n")
            return "\n".join(lines)

        # Check for listing justified spikes
        if "justified" in q_lower or "explained" in q_lower:
            justified = self.df_results[self.df_results["flagged"] == "No (justified)"]
            lines = [f"Found {len(justified)} justified cost rises backed by verified notes:"]
            for _, r in justified.iterrows():
                lines.append(f"• {r['route']} (Week of {r['week_of']}): CPTK INR {r['cost_per_tonne_km']} (Note {r['matched_note_id']})")
                lines.append(f"  Explanation: {r['reason']}\n")
            return "\n".join(lines)

        # Detect route mentioned in query
        detected_route = None
        for r in self.df_results["route"].unique():
            parts = r.lower().split("-")
            if r.lower() in q_lower or (parts[0] in q_lower and parts[1] in q_lower):
                detected_route = r
                break

        # Detect year/month
        year_match = re.search(r"\b(2024|2025)\b", q_lower)
        year = int(year_match.group(1)) if year_match else None

        months = {
            "january": 1, "jan": 1, "february": 2, "feb": 2,
            "march": 3, "mar": 3, "april": 4, "apr": 4,
            "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11,
            "december": 12, "dec": 12
        }
        detected_month = None
        for m_name, m_num in months.items():
            if re.search(r"\b" + m_name + r"\b", q_lower):
                detected_month = m_num
                break

        # Filter results
        filtered = self.df_results.copy()
        if detected_route:
            filtered = filtered[filtered["route"] == detected_route]
        if year:
            filtered = filtered[filtered["week_dt"].dt.year == year]
        if detected_month:
            filtered = filtered[filtered["week_dt"].dt.month == detected_month]

        if len(filtered) == 0:
            if detected_route:
                return (
                    f"No anomalous cost spikes were flagged for route '{detected_route}' "
                    f"during the specified timeframe ({year or 'any year'}-{detected_month or 'any month'}). "
                    f"Costs remained within standard trailing-8-week and corridor peer parameters."
                )
            return (
                "I couldn't identify the specific corridor or timeframe from your query. "
                "Try asking about a specific route (e.g., 'Ahmedabad-Mumbai in January 2025' or 'Chennai-Bangalore in March 2025')."
            )

        # Synthesize response from findings
        resp_lines = [f"Analysis for {detected_route or 'Queried Corridors'}:"]
        for _, row in filtered.iterrows():
            resp_lines.append(f"\n[Week of {row['week_of']}]")
            resp_lines.append(f"- Cost per tonne-km: INR {row['cost_per_tonne_km']:.2f}")
            resp_lines.append(f"- Comparison vs Route History: {row['vs_own_history']}")
            resp_lines.append(f"- Comparison vs Peer Corridors: {row['vs_similar_routes']}")
            resp_lines.append(f"- Verdict: {'FLAGGED AS UNEXPLAINED' if row['flagged'] == 'Yes' else 'VERIFIED AS JUSTIFIED'}")
            if pd.notna(row['matched_note_id']) and row['matched_note_id'] != '':
                resp_lines.append(f"- Supporting Note ID: {row['matched_note_id']}")
            resp_lines.append(f"- Reason: {row['reason']}")

        return "\n".join(resp_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freight Tiger Interactive Shipping Assistant")
    parser.add_argument("--query", type=str, help="Plain English query to ask the assistant")
    args = parser.parse_args()

    assistant = InteractiveShippingAssistant()

    if args.query:
        print(assistant.answer_query(args.query))
    else:
        print("=" * 70)
        print(" FREIGHT TIGER INTERACTIVE ASSISTANT (Interactive Mode) ")
        print(" Ask any question in plain English (type 'exit' or 'quit' to stop).")
        print(" Example: 'Why did Ahmedabad-Mumbai spike in January 2025?'")
        print("=" * 70)
        while True:
            try:
                user_q = input("\nYour Question > ").strip()
                if user_q.lower() in ["exit", "quit", "q"]:
                    break
                if not user_q:
                    continue
                print("\nAssistant Response:")
                print(assistant.answer_query(user_q))
            except (KeyboardInterrupt, EOFError):
                break
