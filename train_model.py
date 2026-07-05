#!/usr/bin/env python3
"""
Train the optional gradient-boosting calibration layer on this season's
(or several seasons') Statcast data.

    python train_model.py                  # current season
    python train_model.py --year 2025     # a past season

Prints time-series cross-validated MAE for runs and strikeouts per start,
then saves the model to cache/gbm.pkl. predict_today.py picks it up
automatically on the next run.
"""
import argparse
import datetime as dt

from mlb_matchup import data as D
from mlb_matchup import model as ML

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=dt.date.today().year)
    args = ap.parse_args()
    end = min(dt.date.today(), dt.date(args.year, 11, 5))
    sc = D.fetch_statcast_season(end)
    # skip the first 3 weeks: point-in-time samples too small to be useful
    frame = ML.build_training_frame(sc, dt.date(args.year, 4, 20))
    print(f"Built {len(frame)} training starts")
    scores = ML.train(frame)
    print(f"CV MAE  runs/start: {scores['y_runs']:.2f}   K/start: {scores['y_k']:.2f}")
    print("Saved model to cache/gbm.pkl")
