#!/usr/bin/env python3
"""
Backtest the matchup engine against a past season (or the current one
to-date).

    python backtest.py --year 2025
    python backtest.py --year 2026            # season to date
    python backtest.py --year 2025 --start 2025-05-01 --end 2025-09-28

For every start in the window it rebuilds the projection using ONLY data
available before that day (no leakage), then compares against the runs and
strikeouts the starter actually recorded. Reports:

  1. Error vs. naive baselines (always-predict-the-mean, and season ERA).
  2. Rank quality: Spearman correlation between grade and actual runs.
  3. Calibration table: average actual runs/K per start by grade bucket —
     the number that matters most for "should I trust a 60-grade spot?"
  4. Top-decile vs bottom-decile spread.

Expectations, honestly: per-start run scoring is extremely noisy. A good
result here is NOT a tiny MAE — it's (a) beating the baselines and
(b) a calibration table that slopes the right way with a meaningful
spread between grade buckets.
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from mlb_matchup import data as D
from mlb_matchup import model as ML

# Average innings per start; used to put exp_runs_per_9 on a per-start scale.
IP_PER_START = 5.2


def evaluate(frame: pd.DataFrame) -> None:
    n = len(frame)
    if n < 100:
        print(f"Only {n} starts in window — results will be unstable. "
              f"Widen the date range.")
    frame = frame.copy()
    frame["pred_runs_start"] = frame["exp_runs_per_9"] * IP_PER_START / 9.0
    # grade on the same 0-100 scale predict_today prints
    from mlb_matchup.engine import LEAGUE
    frame["grade"] = (50 + (LEAGUE["xwoba"] - frame["exp_xwoba_allowed"]) * 500
                      + (frame["exp_k_pct"] - LEAGUE["k_pct"]) * 100).clip(0, 100)

    y = frame["y_runs"]
    pred = frame["pred_runs_start"]

    # --- 1. error vs baselines -------------------------------------------
    mae_model = (pred - y).abs().mean()
    mae_mean = (y.mean() - y).abs().mean()
    print("=" * 66)
    print(f"BACKTEST: {n} starts, {frame['date'].min()} -> {frame['date'].max()}")
    print("=" * 66)
    print(f"\nMAE, runs allowed per start")
    print(f"  model:                {mae_model:.3f}")
    print(f"  always-predict-mean:  {mae_mean:.3f}"
          f"   (model {'BEATS' if mae_model < mae_mean else 'LOSES TO'} baseline "
          f"by {abs(mae_mean - mae_model):.3f})")

    mae_k = (frame["exp_k_pct"] * frame["y_bf"] - frame["y_k"]).abs().mean() \
        if "y_bf" in frame else None
    if mae_k is not None:
        print(f"  strikeouts/start MAE: {mae_k:.3f}")

    # --- 2. rank quality ---------------------------------------------------
    rho = frame["grade"].corr(y, method="spearman")
    print(f"\nSpearman(grade, actual runs): {rho:+.3f}  "
          f"(negative is good: higher grade -> fewer runs)")

    # --- 3. calibration by grade bucket -------------------------------------
    bins = [0, 40, 45, 50, 55, 60, 100]
    labels = ["<40 avoid", "40-45", "45-50", "50-55", "55-60", "60+ strong"]
    frame["bucket"] = pd.cut(frame["grade"], bins=bins, labels=labels)
    cal = frame.groupby("bucket", observed=True).agg(
        starts=("y_runs", "size"),
        actual_runs=("y_runs", "mean"),
        predicted_runs=("pred_runs_start", "mean"),
        actual_k=("y_k", "mean"),
    ).round(2)
    print("\nCalibration by grade bucket (want actual_runs to fall as grade rises):")
    print(cal.to_string())

    # --- 4. decile spread ----------------------------------------------------
    frame["decile"] = pd.qcut(frame["grade"], 10, labels=False, duplicates="drop")
    top = frame[frame["decile"] == frame["decile"].max()]["y_runs"].mean()
    bot = frame[frame["decile"] == 0]["y_runs"].mean()
    print(f"\nTop-decile grades  -> {top:.2f} actual runs/start")
    print(f"Bottom-decile      -> {bot:.2f} actual runs/start")
    print(f"Spread             -> {bot - top:+.2f} runs/start")
    print("\nRule of thumb: a spread of ~1.0+ runs/start between best and worst")
    print("deciles means the rankings carry real signal; the per-start MAE will")
    print("always look large because single games are noisy.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=dt.date.today().year)
    ap.add_argument("--start", type=str, default=None,
                    help="YYYY-MM-DD (default: Apr 20 of --year, so early-"
                         "season samples aren't garbage)")
    ap.add_argument("--end", type=str, default=None)
    args = ap.parse_args()

    end = (dt.date.fromisoformat(args.end) if args.end
           else min(dt.date.today() - dt.timedelta(days=1),
                    dt.date(args.year, 11, 5)))
    start = (dt.date.fromisoformat(args.start) if args.start
             else dt.date(args.year, 4, 20))

    print(f"Loading {args.year} Statcast data (cached after first run)...")
    sc = D.fetch_statcast_season(end)
    sc["_gd"] = pd.to_datetime(sc["game_date"]).dt.date
    sc = sc[sc["_gd"] <= end].drop(columns="_gd")

    print("Building point-in-time projections for every start "
          "(slow the first time)...")
    frame = ML.build_training_frame(sc, start)
    if frame.empty:
        print("No starts found in that window.")
        return 1
    frame.to_csv(f"out/backtest_{args.year}.csv", index=False)
    evaluate(frame)
    print(f"\nPer-start detail saved to out/backtest_{args.year}.csv")
    return 0


if __name__ == "__main__":
    import pathlib
    pathlib.Path("out").mkdir(exist_ok=True)
    raise SystemExit(main())
