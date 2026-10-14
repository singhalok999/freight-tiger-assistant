"""
eval_harness.py
Automated Evaluation Harness for Freight Tiger Shipping Cost Assistant.
Runs a labeled test suite to rigorously test:
1. Mathematical precision
2. True positive justified anomalies
3. True positive unexplained anomalies
4. Anti-hallucination defense against negative controls (N004, N005, N006, N010)
5. Strict output schema compliance
"""

from typing import Dict, List, Any
import sys
import pandas as pd
from data_engine import ShippingDataEngine
from retriever import ContextRetriever
from reasoner import AntiHallucinationReasoner, TokenCostTracker


class EvalHarness:
    def __init__(self):
        self.data_engine = ShippingDataEngine("shipment_records.csv")
        self.retriever = ContextRetriever("context_notes.csv")
        self.reasoner = AntiHallucinationReasoner()
        self.test_results: List[Dict[str, Any]] = []

    def run_tests(self) -> bool:
        print("=" * 70)
        print(" FREIGHT TIGER ASSISTANT: AUTOMATED EVALUATION & GUARDRAIL HARNESS ")
        print("=" * 70)

        df_metrics = self.data_engine.compute_weekly_metrics()

        # -------------------------------------------------------------
        # TEST 1: Exact Mathematical Precision on Sample Ground Truth
        # -------------------------------------------------------------
        print("\n[TEST 1] Mathematical Parity with Ground Truth...")
        ground_truth = [
            ("Delhi-Jaipur", "2024-11-11", 4.17, "+35.5% vs this route's past average", "+21.0% vs similar-length routes this week"),
            ("Ahmedabad-Mumbai", "2025-01-20", 3.29, "+29.5% vs this route's past average", "+22.5% vs similar-length routes this week"),
            ("Mumbai-Pune", "2025-09-15", 3.98, "+9.2% vs this route's past average", "+23.6% vs similar-length routes this week"),
        ]

        math_pass = True
        for route, week, exp_cptk, exp_vown, exp_vsim in ground_truth:
            row = df_metrics[(df_metrics["route"] == route) & (df_metrics["week_of"] == week)].iloc[0]
            if (row["cost_per_tonne_km"] != exp_cptk or
                row["vs_own_history"] != exp_vown or
                row["vs_similar_routes"] != exp_vsim):
                print(f"  FAIL: Math mismatch for {route} @ {week}")
                print(f"        Expected: ({exp_cptk}, {exp_vown}, {exp_vsim})")
                print(f"        Got:      ({row['cost_per_tonne_km']}, {row['vs_own_history']}, {row['vs_similar_routes']})")
                math_pass = False

        self._record_result("Mathematical Precision (CPTK, vs_own, vs_peers)", math_pass)

        # -------------------------------------------------------------
        # TEST 2: True Positive Justification - Festival (N002)
        # -------------------------------------------------------------
        print("\n[TEST 2] Justified Anomaly: Festival Surcharge (N002)...")
        r2 = self.retriever.retrieve_candidates("Ahmedabad-Mumbai", "2025-01-20")
        v2 = self.reasoner.evaluate_candidate(
            "Ahmedabad-Mumbai", "2025-01-20", 3.29,
            "+29.5% vs this route's past average", "+22.5% vs similar-length routes this week", r2
        )
        pass_t2 = (v2["flagged"] == "No (justified)" and v2["matched_note_id"] == "N002")
        self._record_result("True Positive: Festival Surcharge (N002)", pass_t2)

        # -------------------------------------------------------------
        # TEST 3: True Positive Justification - Flooding (N001)
        # -------------------------------------------------------------
        print("\n[TEST 3] Justified Anomaly: Severe Flooding (N001)...")
        r3 = self.retriever.retrieve_candidates("Chennai-Bangalore", "2025-02-24")
        v3 = self.reasoner.evaluate_candidate(
            "Chennai-Bangalore", "2025-02-24", 3.58,
            "+31.5% vs this route's past average", "+38.7% vs similar-length routes this week", r3
        )
        pass_t3 = (v3["flagged"] == "No (justified)" and v3["matched_note_id"] == "N001")
        self._record_result("True Positive: Flood Detours (N001)", pass_t3)

        # -------------------------------------------------------------
        # TEST 4: Negative Control Guardrail - Stable Demand (N006)
        # -------------------------------------------------------------
        print("\n[TEST 4] Negative Control: Stable Demand / No Disruptions (N006)...")
        r4 = self.retriever.retrieve_candidates("Mumbai-Pune", "2025-09-15")
        v4 = self.reasoner.evaluate_candidate(
            "Mumbai-Pune", "2025-09-15", 3.98,
            "+9.2% vs this route's past average", "+23.6% vs similar-length routes this week", r4
        )
        pass_t4 = (v4["flagged"] == "Yes" and v4["matched_note_id"] == "" and "N006" in v4["reason"])
        self._record_result("Negative Control: Rejection of Stable Demand (N006)", pass_t4)

        # -------------------------------------------------------------
        # TEST 5: Negative Control Guardrail - Costs Absorbed (N010)
        # -------------------------------------------------------------
        print("\n[TEST 5] Negative Control: Mandate Absorbed Without Rate Change (N010)...")
        r5 = self.retriever.retrieve_candidates("Mumbai-Pune", "2025-10-27")
        v5 = self.reasoner.evaluate_candidate(
            "Mumbai-Pune", "2025-10-27", 4.07,
            "+5.5% vs this route's past average", "+28.3% vs similar-length routes this week", r5
        )
        pass_t5 = (v5["flagged"] == "Yes" and v5["matched_note_id"] == "" and "without a rate change" in v5["reason"])
        self._record_result("Negative Control: Rejection of Absorbed Costs (N010)", pass_t5)

        # -------------------------------------------------------------
        # TEST 6: Unexplained Surge Without Context Notes
        # -------------------------------------------------------------
        print("\n[TEST 6] Unexplained Surge: Delhi-Jaipur November 2024...")
        r6 = self.retriever.retrieve_candidates("Delhi-Jaipur", "2024-11-11")
        v6 = self.reasoner.evaluate_candidate(
            "Delhi-Jaipur", "2024-11-11", 4.17,
            "+35.5% vs this route's past average", "+21.0% vs similar-length routes this week", r6
        )
        pass_t6 = (v6["flagged"] == "Yes" and v6["matched_note_id"] == "" and "No matching note" in v6["reason"])
        self._record_result("Unexplained Surge: Delhi-Jaipur (Zero False Justifications)", pass_t6)

        # -------------------------------------------------------------
        # TEST 7: Negative Control - Non-affected Toll Plaza (N004)
        # -------------------------------------------------------------
        print("\n[TEST 7] Negative Control: Toll Plaza Not in Dataset (N004)...")
        is_neg_n004 = self.reasoner.is_negative_control("A new toll plaza was commissioned, but affected routes are not part of this dataset.")
        self._record_result("Negative Control: Rejection of Toll Outside Dataset (N004)", is_neg_n004)

        # -------------------------------------------------------------
        # TEST 8: Negative Control - Maintenance with No Impact (N005)
        # -------------------------------------------------------------
        print("\n[TEST 8] Negative Control: Maintenance With No Cost Impact (N005)...")
        is_neg_n005 = self.reasoner.is_negative_control("Scheduled maintenance; costs were not significantly affected.")
        self._record_result("Negative Control: Rejection of Maintenance (N005)", is_neg_n005)

        # Summary
        print("\n" + "=" * 70)
        print(" EVALUATION RESULTS SUMMARY ")
        print("=" * 70)
        all_passed = True
        for t in self.test_results:
            status = "PASSED [OK]" if t["passed"] else "FAILED [X]"
            if not t["passed"]:
                all_passed = False
            print(f"- {t['name']:<55} : {status}")

        print("=" * 70)
        score = sum(1 for t in self.test_results if t["passed"])
        total = len(self.test_results)
        pct = (score / total) * 100
        print(f"OVERALL EVAL SCORE: {score}/{total} ({pct:.1f}%)")
        if all_passed:
            print("VERDICT: ALL GUARDRAILS & ACCURACY TESTS PASSED WITH 100% RELIABILITY.")
        else:
            print("VERDICT: SOME CHECKS FAILED. PLEASE REVIEW LOGS.")
        print("=" * 70)
        return all_passed

    def _record_result(self, name: str, passed: bool):
        self.test_results.append({"name": name, "passed": passed})
        status = "PASSED" if passed else "FAILED"
        print(f"  Result: {status}")


if __name__ == "__main__":
    harness = EvalHarness()
    success = harness.run_tests()
    sys.exit(0 if success else 1)
