"""
Optional machine-learning layer.

The odds-ratio engine is fully interpretable and works out of the box.
This module adds a gradient-boosting model trained on historical starts
that learns how engine features map to *actual* game outcomes (runs
allowed and strikeouts per start), which calibrates the projections and
can capture interactions the analytic engine misses.

Training rows are built start-by-start using ONLY data available before
that start (point-in-time aggregation) to avoid leakage.
"""

from __future__ import annotations

import datetime as dt
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from . import data as D
from .engine import project_matchup

FEATURES = [
    "exp_xwoba_allowed", "exp_k_pct", "exp_runs_per_9",   # engine outputs
    "p_xwoba", "p_k_pct", "p_bb_pct", "p_whiff", "p_gb", "p_velo_trend",
    "l_xwoba", "l_k_pct", "l_barrel",                      # lineup aggregates
    "park_factor", "is_home",
]

MODEL_PATH = Path("cache/gbm.pkl")


def build_training_frame(sc: pd.DataFrame, min_date: dt.date) -> pd.DataFrame:
    """
    Walk every start in the Statcast data chronologically and build
    point-in-time features + realized outcomes. Slow (minutes) but only
    needs to run when retraining.
    """
    sc = sc.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"]).dt.date
    rows = []
    dates = sorted(d for d in sc["game_date"].unique() if d >= min_date)
    for day in dates:
        past = sc[sc["game_date"] < day]
        if past.empty:
            continue
        today_pitches = sc[sc["game_date"] == day]
        bat_tbl = D.aggregate_batters(past, day)
        pit_tbl = D.aggregate_pitchers(past, day)
        for (gid, half), grp in today_pitches.groupby(["game_pk", "inning_topbot"]):
            starter = grp.sort_values("at_bat_number").iloc[0]["pitcher"]
            sp = grp[grp["pitcher"] == starter]
            outcome_pa = sp[sp["events"].notna() & sp["events"].isin(D.PA_END_EVENTS)]
            if len(outcome_pa) < 12:      # skip openers / early blowups
                continue
            if starter not in pit_tbl.index:
                continue
            lineup_ids = (
                sp.sort_values("at_bat_number")["batter"].drop_duplicates().head(9).tolist()
            )
            lineup = [
                D.batter_profile(bat_tbl.loc[b], str(b))
                for b in lineup_ids if b in bat_tbl.index
            ]
            if len(lineup) < 6:
                continue
            pitcher = D.pitcher_profile(pit_tbl.loc[starter], str(starter))
            home_team = sp.iloc[0]["home_team"]
            is_home = half == "Top"       # home pitcher pitches the top half
            pf = D.PARK_FACTORS.get(home_team, 1.0)
            proj = project_matchup(pitcher, lineup, pf, is_home)
            lin = bat_tbl.loc[[b for b in lineup_ids if b in bat_tbl.index]]
            runs = float(sp["post_bat_score"].max() - sp["bat_score"].min())
            rows.append({
                "date": day, "pitcher": starter,
                "exp_xwoba_allowed": proj.exp_xwoba_allowed,
                "exp_k_pct": proj.exp_k_pct,
                "exp_runs_per_9": proj.exp_runs_per_9,
                "p_xwoba": pitcher.xwoba_allowed, "p_k_pct": pitcher.k_pct,
                "p_bb_pct": pitcher.bb_pct, "p_whiff": pitcher.whiff_pct,
                "p_gb": pitcher.gb_pct, "p_velo_trend": pitcher.velo_trend,
                "l_xwoba": lin["xwoba"].mean(), "l_k_pct": lin["k_pct"].mean(),
                "l_barrel": lin["barrel_pct"].mean(),
                "park_factor": pf, "is_home": int(is_home),
                "y_runs": runs,
                "y_k": float(outcome_pa["events"].str.startswith("strikeout").sum()),
            })
    return pd.DataFrame(rows)


def train(frame: pd.DataFrame) -> dict:
    frame = frame.sort_values("date")
    X = frame[FEATURES].fillna(frame[FEATURES].median(numeric_only=True))
    models, scores = {}, {}
    for target in ("y_runs", "y_k"):
        y = frame[target]
        gbm = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, early_stopping=True,
        )
        cv = TimeSeriesSplit(n_splits=5)
        scores[target] = -cross_val_score(
            gbm, X, y, cv=cv, scoring="neg_mean_absolute_error"
        ).mean()
        gbm.fit(X, y)
        models[target] = gbm
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)
    return scores


def predict(feature_row: dict) -> dict | None:
    """Blend in ML predictions if a trained model exists; else None."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    X = pd.DataFrame([feature_row]).reindex(columns=FEATURES).fillna(0)
    return {
        "ml_exp_runs": float(models["y_runs"].predict(X)[0]),
        "ml_exp_k": float(models["y_k"].predict(X)[0]),
    }
