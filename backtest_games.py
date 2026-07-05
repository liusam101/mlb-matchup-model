#!/usr/bin/env python3
"""
Backtest the GAME WIN PROBABILITY model against past weeks or seasons.

    python backtest_games.py --year 2026                        # season to date
    python backtest_games.py --year 2026 --start 2026-06-01     # recent weeks
    python backtest_games.py --year 2025                        # a full past season

For every game in the window it rebuilds the projection exactly as it would
have looked pregame — player profiles and bullpen availability from ONLY the
days before, the actual starters and lineups (both known pregame), park and
home field — then compares the win probability to who actually won.

Reported:
  1. Accuracy picking winners, vs. the "always pick home team" baseline.
  2. Brier score (mean squared error of the probabilities; lower = better).
     Coin-flip = 0.250. Vegas closing lines run ~0.240-0.245. Beating the
     home-team baseline and landing near that band is a strong result.
  3. Calibration: when the model says 60%, does that side win ~60%?
  4. Bullpen check: does the model's edge grow in games where one pen was
     meaningfully more available than the other?
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_matchup import data as D
from mlb_matchup.engine import project_game, project_matchup

# Statcast team codes occasionally differ from our park-factor keys
TEAM_ALIAS = {"ARI": "AZ", "CHW": "CWS", "WSN": "WSH", "SDP": "SD",
              "SFG": "SF", "TBR": "TB", "KCR": "KC", "OAK": "ATH"}


def _norm(team: str) -> str:
    return TEAM_ALIAS.get(team, team)


def replay(sc: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    sc = sc.copy()
    sc["gdate"] = pd.to_datetime(sc["game_date"]).dt.date
    rows = []
    dates = sorted(d for d in sc["gdate"].unique() if start <= d <= end)
    for i, day in enumerate(dates, 1):
        past = sc[sc["gdate"] < day]
        if past["gdate"].nunique() < 14:      # need a real sample behind us
            continue
        today = sc[sc["gdate"] == day]
        bat_tbl = D.aggregate_batters(past, day)
        pit_tbl = D.aggregate_pitchers(past, day)
        pens = D.aggregate_bullpens(past, day)
        print(f"  [{i}/{len(dates)}] {day}: {today['game_pk'].nunique()} games")

        for gpk, game in today.groupby("game_pk"):
            top = game[game["inning_topbot"] == "Top"].sort_values(
                ["at_bat_number", "pitch_number"])
            bot = game[game["inning_topbot"] == "Bot"].sort_values(
                ["at_bat_number", "pitch_number"])
            if top.empty or bot.empty:
                continue
            home_team = _norm(game["home_team"].iloc[0])
            away_team = _norm(game["away_team"].iloc[0])

            home_sp = top["pitcher"].iloc[0]      # home pitches the top half
            away_sp = bot["pitcher"].iloc[0]
            if home_sp not in pit_tbl.index or away_sp not in pit_tbl.index:
                continue
            home_p = D.pitcher_profile(pit_tbl.loc[home_sp], str(home_sp))
            away_p = D.pitcher_profile(pit_tbl.loc[away_sp], str(away_sp))

            def lineup_from(half: pd.DataFrame) -> list:
                ids = half["batter"].drop_duplicates().head(9)
                return [D.batter_profile(bat_tbl.loc[b], str(b))
                        for b in ids if b in bat_tbl.index]

            away_lineup = lineup_from(top)        # away bats in the top half
            home_lineup = lineup_from(bot)
            if len(away_lineup) < 7 or len(home_lineup) < 7:
                continue

            pf = D.PARK_FACTORS.get(home_team, 1.0)
            home_res = project_matchup(home_p, away_lineup, pf, is_home=True)
            away_res = project_matchup(away_p, home_lineup, pf, is_home=False)
            from mlb_matchup.engine import BullpenProfile
            hp = pens.get(home_team, BullpenProfile(team=home_team))
            ap = pens.get(away_team, BullpenProfile(team=away_team))
            gp = project_game(home_res, away_res, hp, ap,
                              home_lineup, away_lineup, pf)

            # actual result from the final post-scores
            h_final = float(game["post_home_score"].max())
            a_final = float(game["post_away_score"].max())
            if h_final == a_final:
                continue
            rows.append({
                "date": day, "home": home_team, "away": away_team,
                "home_wp": gp.home_wp,
                "home_won": int(h_final > a_final),
                "home_runs": h_final, "away_runs": a_final,
                "exp_home_runs": gp.exp_home_runs,
                "exp_away_runs": gp.exp_away_runs,
                "pen_avail_gap": round(hp.avail_frac - ap.avail_frac, 2),
            })
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame) -> None:
    n = len(df)
    p, y = df["home_wp"].clip(0.01, 0.99), df["home_won"]

    picks_right = ((p > 0.5) == (y == 1)).mean()
    home_base = y.mean()
    brier = ((p - y) ** 2).mean()
    brier_base = ((home_base - y) ** 2).mean()   # constant home-rate forecast
    logloss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    print("\n" + "=" * 66)
    print(f"GAME BACKTEST: {n} games, {df['date'].min()} -> {df['date'].max()}")
    print("=" * 66)
    print(f"\nPick accuracy:          {picks_right:.1%}")
    print(f"Always-pick-home:       {home_base:.1%}"
          f"   (model {'BEATS' if picks_right > home_base else 'LOSES TO'} it)")
    print(f"Brier score:            {brier:.4f}  (coin flip 0.2500; "
          f"constant-home baseline {brier_base:.4f}; Vegas ~0.24)")
    print(f"Log loss:               {logloss:.4f}")

    bins = [0, .40, .45, .50, .55, .60, .65, 1.0]
    df = df.copy()
    df["bucket"] = pd.cut(df["home_wp"], bins)
    cal = df.groupby("bucket", observed=True).agg(
        games=("home_won", "size"),
        predicted=("home_wp", "mean"),
        actual=("home_won", "mean"),
    ).round(3)
    print("\nCalibration (predicted home WP vs. how often home actually won):")
    print(cal.to_string())

    # does the bullpen-availability signal show up in results?
    fresh_edge = df[df["pen_avail_gap"] >= 0.25]
    tired_edge = df[df["pen_avail_gap"] <= -0.25]
    if len(fresh_edge) >= 25 and len(tired_edge) >= 25:
        print(f"\nBullpen availability check:")
        print(f"  home pen much fresher ({len(fresh_edge)} games): "
              f"home won {fresh_edge['home_won'].mean():.1%}")
        print(f"  home pen much more taxed ({len(tired_edge)} games): "
              f"home won {tired_edge['home_won'].mean():.1%}")

    print("\nHow to read this: per-game baseball is close to a coin flip by")
    print("design (the best teams lose 60+ times). Success = beating the")
    print("always-home baseline on Brier, calibration rows that line up, and")
    print("55%+ buckets that actually win 55%+.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=dt.date.today().year)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    args = ap.parse_args()

    end = (dt.date.fromisoformat(args.end) if args.end
           else min(dt.date.today() - dt.timedelta(days=1),
                    dt.date(args.year, 11, 5)))
    start = (dt.date.fromisoformat(args.start) if args.start
             else dt.date(args.year, 4, 20))

    print(f"Loading {args.year} Statcast data (cached after first run)...")
    sc = D.fetch_statcast_season(end)

    print("Replaying games with point-in-time projections...")
    df = replay(sc, start, end)
    if df.empty:
        print("No games found in that window.")
        return 1
    Path("out").mkdir(exist_ok=True)
    df.to_csv(f"out/backtest_games_{args.year}.csv", index=False)
    evaluate(df)
    print(f"\nPer-game detail saved to out/backtest_games_{args.year}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
