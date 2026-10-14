"""
data_engine.py
Core analytical and statistical engine for Freight Tiger Shipping Cost Assistant.
Computes:
1. Monday-to-Sunday weekly grouping (week_of)
2. Cost per tonne-km: sum(freight_cost) / sum(quantity * distance)
3. Trailing 8-week baseline (strictly prior weeks, no lookahead, handles <8 weeks gracefully)
4. Peer baseline across routes with matching route_type (strictly excluding self)
5. Candidate anomaly filtering based on historical and peer baselines.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class ShippingDataEngine:
    def __init__(self, shipment_csv_path: str = "shipment_records.csv"):
        self.shipment_csv_path = shipment_csv_path
        self.raw_df: Optional[pd.DataFrame] = None
        self.weekly_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Loads and pre-processes shipment records."""
        df = pd.read_csv(self.shipment_csv_path)
        df["shipment_date"] = pd.to_datetime(df["shipment_date"])
        # Monday of the week
        df["week_of"] = df["shipment_date"].apply(
            lambda d: d - pd.Timedelta(days=d.weekday())
        )
        df["route"] = df["origin"] + "-" + df["destination"]
        df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
        self.raw_df = df
        return df

    def compute_weekly_metrics(
        self,
        own_threshold: float = 15.0,
        peer_threshold: float = 20.0,
    ) -> pd.DataFrame:
        """
        Computes weekly metrics and baselines across all routes and weeks.
        
        Formula verification:
        - cost_per_tonne_km = sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)
        - vs_own_history: trailing 8-week unweighted average of weekly cptk, strictly prior.
        - vs_similar_routes: unweighted average of weekly cptk across same route_type peers.
        """
        if self.raw_df is None:
            self.load_data()

        df = self.raw_df

        # Aggregate totals per route, route_type, week_of
        grouped = df.groupby(["route", "route_type", "week_of"]).apply(
            lambda g: pd.Series({
                "total_cost": g["freight_cost_inr"].sum(),
                "total_tkm": g["tonne_km"].sum(),
                "shipment_count": len(g)
            }),
            include_groups=False
        ).reset_index()

        grouped["cptk"] = grouped["total_cost"] / grouped["total_tkm"]
        grouped = grouped.sort_values(["route", "week_of"]).reset_index(drop=True)

        results = []
        for route in grouped["route"].unique():
            r_df = grouped[grouped["route"] == route].copy()

            for _, row in r_df.iterrows():
                w = row["week_of"]
                cptk = row["cptk"]
                rtype = row["route_type"]

                # 1. vs own history: trailing 8 weeks strictly before current week
                prior = r_df[r_df["week_of"] < w].tail(8)
                num_prior = len(prior)

                if num_prior > 0:
                    past_avg = prior["cptk"].mean()
                    vs_own_pct = ((cptk - past_avg) / past_avg) * 100.0
                    vs_own_str = f"{vs_own_pct:+.1f}% vs this route's past average"
                else:
                    past_avg = np.nan
                    vs_own_pct = np.nan
                    vs_own_str = "N/A (first week, no prior history)"

                # 2. vs similar routes: same week, same route_type, exclude route itself
                peers = grouped[
                    (grouped["week_of"] == w)
                    & (grouped["route_type"] == rtype)
                    & (grouped["route"] != route)
                ]
                num_peers = len(peers)

                if num_peers > 0:
                    peer_avg = peers["cptk"].mean()
                    vs_peer_pct = ((cptk - peer_avg) / peer_avg) * 100.0
                    vs_peer_str = f"{vs_peer_pct:+.1f}% vs similar-length routes this week"
                else:
                    peer_avg = np.nan
                    vs_peer_pct = np.nan
                    vs_peer_str = "N/A (no peer routes in length category)"

                # Candidate anomaly condition:
                # Rising cost and significantly out of ordinary compared to history OR peers
                is_candidate = False
                if not np.isnan(vs_own_pct) and not np.isnan(vs_peer_pct):
                    # Spike vs own history OR spike vs peers (with positive vs_own)
                    if vs_own_pct >= own_threshold or vs_peer_pct >= peer_threshold:
                        if cptk > (past_avg if not np.isnan(past_avg) else 0):
                            is_candidate = True
                elif not np.isnan(vs_peer_pct) and vs_peer_pct >= peer_threshold:
                    is_candidate = True

                results.append({
                    "route": route,
                    "route_type": rtype,
                    "week_of": w.strftime("%Y-%m-%d"),
                    "week_dt": w,
                    "cost_per_tonne_km": round(cptk, 2),
                    "vs_own_pct": round(vs_own_pct, 4) if not np.isnan(vs_own_pct) else None,
                    "vs_peer_pct": round(vs_peer_pct, 4) if not np.isnan(vs_peer_pct) else None,
                    "vs_own_history": vs_own_str,
                    "vs_similar_routes": vs_peer_str,
                    "past_avg_cptk": round(past_avg, 4) if not np.isnan(past_avg) else None,
                    "peer_avg_cptk": round(peer_avg, 4) if not np.isnan(peer_avg) else None,
                    "prior_weeks_count": num_prior,
                    "is_candidate": is_candidate,
                })

        self.weekly_df = pd.DataFrame(results)
        return self.weekly_df


if __name__ == "__main__":
    engine = ShippingDataEngine()
    df_metrics = engine.compute_weekly_metrics()
    print(f"Total evaluated route-weeks: {len(df_metrics)}")
    candidates = df_metrics[df_metrics["is_candidate"]]
    print(f"Candidate anomalies flagged for investigation: {len(candidates)}")
    print("\nSample candidate anomalies:")
    print(candidates[["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes"]].head())
