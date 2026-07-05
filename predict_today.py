#!/usr/bin/env python3
"""
Daily slate runner.

    python predict_today.py                # today's full MLB slate (live data)
    python predict_today.py --date 2026-07-05
    python predict_today.py --demo         # offline demo with synthetic data

Output, best -> worst:
  1. GAME BOARD   — win probability for every game, combining both starter
                    matchups, both lineups, and availability-adjusted bullpens
  2. PITCHER BOARD — every starter graded vs. the lineup he faces
  3. HITTER BOARD  — best individual hitter-vs-starter edges

CSVs land in ./out.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from mlb_matchup.engine import (
    BatterProfile, BullpenProfile, PitcherProfile,
    project_game, project_matchup, LEAGUE,
)
from mlb_matchup import model as ML

OUT = Path("out")
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------
def run_live(date: dt.date) -> tuple[list, list]:
    from mlb_matchup import data as D
    import statsapi

    print(f"Fetching Statcast data through {date} (first run of the season "
          f"downloads ~1-2 GB and takes a while; later runs are incremental)...")
    sc = D.fetch_statcast_season(date)
    bat_tbl = D.aggregate_batters(sc, date)
    pit_tbl = D.aggregate_pitchers(sc, date)
    pens = D.aggregate_bullpens(sc, date)

    def pid_for(name: str) -> int | None:
        try:
            hits = statsapi.lookup_player(name)
            return hits[0]["id"] if hits else None
        except Exception:
            return None

    def build_side(prob_name: str | None, opp_lineup_ids: list) -> tuple:
        """Returns (PitcherProfile|None, lineup list, lineup_is_real)."""
        pitcher = None
        if prob_name:
            pid = pid_for(prob_name)
            if pid is not None and pid in pit_tbl.index:
                pitcher = D.pitcher_profile(pit_tbl.loc[pid], prob_name)
        lineup, real = [], True
        for bid, bname in opp_lineup_ids[:9]:
            if bid in bat_tbl.index:
                lineup.append(D.batter_profile(bat_tbl.loc[bid], bname))
        if len(lineup) < 6:
            real = False
            lineup = [BatterProfile(name=f"avg #{i+1}", bats="R") for i in range(9)]
        return pitcher, lineup, real

    matchups, games = [], []
    for g in D.todays_games(date):
        lineups = D.game_lineups(g["game_id"])
        pf = D.PARK_FACTORS.get(g["home_abbr"], 1.0)

        home_p, away_lineup, away_real = build_side(g["home_probable"], lineups.get("away", []))
        away_p, home_lineup, home_real = build_side(g["away_probable"], lineups.get("home", []))

        home_res = away_res = None
        if home_p:
            home_res = project_matchup(home_p, away_lineup, pf, is_home=True)
            home_res.opponent = g["away"]
            home_res.team = g["home"]
            if not away_real:
                print(f"  [note] {g['away']} lineup not posted; using league-average")
            matchups.append((home_res, home_p, away_lineup, pf, True))
        else:
            print(f"  [skip] no data yet for {g['home']} starter "
                  f"({g['home_probable'] or 'TBD'})")
        if away_p:
            away_res = project_matchup(away_p, home_lineup, pf, is_home=False)
            away_res.opponent = g["home"]
            away_res.team = g["away"]
            if not home_real:
                print(f"  [note] {g['home']} lineup not posted; using league-average")
            matchups.append((away_res, away_p, home_lineup, pf, False))
        else:
            print(f"  [skip] no data yet for {g['away']} starter "
                  f"({g['away_probable'] or 'TBD'})")

        # game-level projection needs both sides
        if home_res and away_res:
            home_pen = pens.get(g["home_abbr"], BullpenProfile(team=g["home_abbr"]))
            away_pen = pens.get(g["away_abbr"], BullpenProfile(team=g["away_abbr"]))
            gp = project_game(home_res, away_res, home_pen, away_pen,
                              home_lineup, away_lineup, pf)
            gp.home_team, gp.away_team = g["home"], g["away"]
            gp.lineups_confirmed = home_real and away_real
            games.append(gp)
        else:
            print(f"  [note] no win prob for {g['away']} @ {g['home']} "
                  f"(need both starters)")

    # optional ML calibration on the pitcher board
    results = []
    for res, pitcher, lineup, pf, is_home in matchups:
        ml = ML.predict({
            "exp_xwoba_allowed": res.exp_xwoba_allowed,
            "exp_k_pct": res.exp_k_pct,
            "exp_runs_per_9": res.exp_runs_per_9,
            "p_xwoba": pitcher.xwoba_allowed, "p_k_pct": pitcher.k_pct,
            "p_bb_pct": pitcher.bb_pct, "p_whiff": pitcher.whiff_pct,
            "p_gb": pitcher.gb_pct, "p_velo_trend": pitcher.velo_trend,
            "l_xwoba": sum(b.xwoba for b in lineup) / len(lineup),
            "l_k_pct": sum(b.k_pct for b in lineup) / len(lineup),
            "l_barrel": sum(b.barrel_pct for b in lineup) / len(lineup),
            "park_factor": pf, "is_home": int(is_home),
        })
        if ml:
            res.ml = ml
        results.append(res)
    return results, games


# ---------------------------------------------------------------------------
# Demo mode (offline, synthetic) — proves the pipeline end to end
# ---------------------------------------------------------------------------
def run_demo() -> tuple[list, list]:
    ace = PitcherProfile(
        name="Ace Righty (demo)", throws="R", tbf=520, xwoba_allowed=0.268,
        k_pct=0.31, bb_pct=0.06, whiff_pct=0.33, gb_pct=0.48,
        xwoba_vs_l=0.285, tbf_vs_l=240, xwoba_vs_r=0.252, tbf_vs_r=280,
        recent_xwoba=0.240, recent_tbf=110, velo_trend=+0.4,
    )
    wildcard = PitcherProfile(
        name="Wild Lefty (demo)", throws="L", tbf=430, xwoba_allowed=0.330,
        k_pct=0.26, bb_pct=0.13, whiff_pct=0.30, gb_pct=0.38,
        xwoba_vs_l=0.290, tbf_vs_l=120, xwoba_vs_r=0.345, tbf_vs_r=310,
        recent_xwoba=0.365, recent_tbf=90, velo_trend=-1.1,
    )

    def mk(name, bats, quality, chase=None, k=None, pa=330):
        base = {"elite": 0.390, "good": 0.345, "avg": 0.312, "weak": 0.275}[quality]
        return BatterProfile(
            name=name, bats=bats, pa=pa, xwoba=base,
            k_pct=k or LEAGUE["k_pct"], chase_pct=chase or LEAGUE["chase_pct"],
            xwoba_vs_l=base + (0.012 if bats == "R" else -0.012), pa_vs_l=pa // 3,
            xwoba_vs_r=base + (0.012 if bats == "L" else -0.012), pa_vs_r=2 * pa // 3,
            recent_xwoba=base + 0.02, recent_pa=60,
        )

    good_lineup = [
        mk("G1 Star (L)", "L", "elite"), mk("G2 (L)", "L", "good"),
        mk("G3 (S)", "S", "good"), mk("G4 (L)", "L", "good"),
        mk("G5 (R)", "R", "avg"), mk("G6 (L)", "L", "avg", chase=0.34, k=0.29),
        mk("G7 (L)", "L", "avg"), mk("G8 (R)", "R", "weak", chase=0.36, k=0.31),
        mk("G9 (L)", "L", "weak"),
    ]
    weak_lineup = [
        mk("W1 (R)", "R", "avg", chase=0.33, k=0.27), mk("W2 (R)", "R", "avg"),
        mk("W3 (L)", "L", "good"), mk("W4 (R)", "R", "weak", chase=0.35, k=0.30),
        mk("W5 (R)", "R", "weak", chase=0.34, k=0.31), mk("W6 (S)", "S", "weak"),
        mk("W7 (R)", "R", "weak", chase=0.37, k=0.33), mk("W8 (R)", "R", "weak"),
        mk("W9 (R)", "R", "weak", chase=0.38, k=0.34),
    ]

    fresh_pen = BullpenProfile(team="FRESH", xwoba_allowed=0.295, k_pct=0.27,
                               tbf=900, avail_frac=1.0)
    gassed_pen = BullpenProfile(team="GASSED", xwoba_allowed=0.322, k_pct=0.22,
                                tbf=900, avail_frac=0.45, n_unavailable=3,
                                unavailable=["Closer", "Setup1", "Setup2"])

    home_res = project_matchup(ace, good_lineup, 1.0, is_home=True)
    home_res.opponent = "Good lineup (demo)"
    away_res = project_matchup(wildcard, weak_lineup, 1.0, is_home=False)
    away_res.opponent = "Weak lineup (demo)"

    g1 = project_game(home_res, away_res, fresh_pen, gassed_pen,
                      home_lineup=weak_lineup, away_lineup=good_lineup)
    g1.home_team, g1.away_team = "Aces w/ fresh pen", "Wild w/ gassed pen"

    # same game but pens swapped, to show availability moving the number
    g2 = project_game(home_res, away_res, gassed_pen, fresh_pen,
                      home_lineup=weak_lineup, away_lineup=good_lineup)
    g2.home_team, g2.away_team = "Aces w/ GASSED pen", "Wild w/ fresh pen"

    return [home_res, away_res], [g1, g2]


# ---------------------------------------------------------------------------
def report(results: list, games: list) -> None:
    if games:
        print("\n" + "=" * 78)
        print(f"{'GAME BOARD — WIN PROBABILITIES':^78}")
        print("=" * 78)
        print(f"{'Matchup':<44}{'Score':>12}{'Home WP':>10}")
        print("-" * 78)
        rows = []
        for gp in sorted(games, key=lambda x: abs(x.home_wp - 0.5), reverse=True):
            flag = "" if getattr(gp, "lineups_confirmed", True) else " *"
            matchup = f"{gp.away_team} @ {gp.home_team}{flag}"
            score = f"{gp.exp_away_runs:.1f}-{gp.exp_home_runs:.1f}"
            print(f"{matchup[:43]:<44}{score:>12}{gp.home_wp:>9.1%}")
            for n in gp.notes:
                print(f"     - {n}")
            rows.append({
                "away": gp.away_team, "home": gp.home_team,
                "away_pitcher": gp.away_pitcher, "home_pitcher": gp.home_pitcher,
                "exp_away_runs": gp.exp_away_runs, "exp_home_runs": gp.exp_home_runs,
                "home_wp": gp.home_wp,
                "lineups_confirmed": getattr(gp, "lineups_confirmed", True),
                "notes": "; ".join(gp.notes),
            })
        pd.DataFrame(rows).to_csv(OUT / "game_board.csv", index=False)
        print("  (* = a lineup wasn't posted yet; re-run near first pitch)")

    results.sort(key=lambda r: r.matchup_grade, reverse=True)
    rows = []
    print("\n" + "=" * 78)
    print(f"{'PITCHER MATCHUP BOARD':^78}")
    print("=" * 78)
    print(f"{'Rk':<3}{'Pitcher':<26}{'Opponent':<30}{'Grade':>6}{'xwOBA':>7}{'R/9':>6}")
    print("-" * 78)
    for i, r in enumerate(results, 1):
        print(f"{i:<3}{r.pitcher[:25]:<26}{str(r.opponent)[:29]:<30}"
              f"{r.matchup_grade:>6.1f}{r.exp_xwoba_allowed:>7.3f}{r.exp_runs_per_9:>6.2f}")
        rows.append({
            "rank": i, "pitcher": r.pitcher, "opponent": r.opponent,
            "grade": r.matchup_grade, "exp_xwoba": r.exp_xwoba_allowed,
            "exp_k_pct": r.exp_k_pct, "exp_runs_per_9": r.exp_runs_per_9,
            "park_factor": r.park_factor,
            **(getattr(r, "ml", {}) or {}),
        })
    pd.DataFrame(rows).to_csv(OUT / "pitcher_board.csv", index=False)

    print("\nBest individual HITTER spots today (biggest edge over the starter):")
    hitters = []
    for r in results:
        for line in r.batter_lines:
            hitters.append({**line, "vs_pitcher": r.pitcher})
    hb = pd.DataFrame(hitters).sort_values("matchup_xwoba", ascending=False)
    print(hb.head(10).to_string(index=False))
    hb.to_csv(OUT / "hitter_board.csv", index=False)
    print(f"\nCSVs written to {OUT.resolve()}")
    print("Grades: 50 = league-average spot, 60+ = strong, 40- = avoid.")
    print("Win probs include starters, lineups, park, home field, and "
          "bullpen availability (25+ pitches yesterday or back-to-back days "
          "= unavailable).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--demo", action="store_true", help="offline synthetic demo")
    args = ap.parse_args()
    if args.demo:
        results, games = run_demo()
        report(results, games)
        sys.exit(0)
    date = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    results, games = run_live(date)
    report(results, games)
