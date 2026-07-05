#!/usr/bin/env python3
"""
How accurate are the model's MOST CONFIDENT picks?

    python top_picks.py                          # uses out/backtest_games_2026.csv
    python top_picks.py --file out/backtest_games_2025.csv --n 1 3 5

For each day, ranks games by confidence (distance of the win probability
from 50%, either side), takes the top N, and grades the favored side.
Reports accuracy at each N alongside all-games accuracy, plus the same cut
by confidence threshold (e.g. every pick the model made at 60%+).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="out/backtest_games_2026.csv")
    ap.add_argument("--n", type=int, nargs="+", default=[1, 3, 5])
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"{path} not found — run backtest_games.py first.")
        return 1

    df = pd.read_csv(path)
    df["confidence"] = (df["home_wp"] - 0.5).abs()
    df["pick_home"] = df["home_wp"] > 0.5
    df["pick_won"] = (df["pick_home"] & (df["home_won"] == 1)) | \
                     (~df["pick_home"] & (df["home_won"] == 0))
    df["pick_wp"] = df["home_wp"].where(df["pick_home"], 1 - df["home_wp"])

    overall = df["pick_won"].mean()
    print("=" * 62)
    print(f"TOP-PICK ANALYSIS: {len(df)} games, "
          f"{df['date'].min()} -> {df['date'].max()}")
    print("=" * 62)
    print(f"\nAll games:            {overall:.1%}  ({df['pick_won'].sum()}"
          f"-{(~df['pick_won']).sum()})")

    print("\nMost confident N picks per day:")
    for n in sorted(args.n):
        top = (df.sort_values("confidence", ascending=False)
                 .groupby("date").head(n))
        acc = top["pick_won"].mean()
        wp = top["pick_wp"].mean()
        print(f"  top {n}/day:  {acc:.1%}  ({top['pick_won'].sum()}"
              f"-{(~top['pick_won']).sum()}, {len(top)} picks; "
              f"model expected {wp:.1%})")

    print("\nBy confidence threshold (all games at or above it):")
    for thr in (0.55, 0.58, 0.60, 0.65):
        sub = df[df["pick_wp"] >= thr]
        if len(sub) == 0:
            continue
        print(f"  >= {thr:.0%} picks: {sub['pick_won'].mean():.1%}  "
              f"({sub['pick_won'].sum()}-{(~sub['pick_won']).sum()}, "
              f"{len(sub)} picks; model expected {sub['pick_wp'].mean():.1%})")

    print("\nRead: confident picks SHOULD beat the all-games rate, and the")
    print("actual %% should track what the model expected. Small pick counts")
    print("swing hard — judge thresholds on 50+ picks, not 15.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
