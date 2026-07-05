"""
Data layer. Pulls everything the engine needs for a given date:

  - Schedule + probable pitchers + lineups: MLB Stats API (statsapi package)
  - Batter/pitcher skill stats (xwOBA, K%, BB%, barrel%, whiff%, chase%,
    splits, recent form, velocity): Statcast via pybaseball, aggregated
    from pitch-level data and cached locally as parquet.

Everything is cached under ./cache so repeated daily runs are fast and
you aren't hammering Baseball Savant.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import BatterProfile, PitcherProfile, LEAGUE

CACHE = Path(os.environ.get("MLB_MODEL_CACHE", "cache"))
CACHE.mkdir(exist_ok=True)

# Baseball Savant park factors (3-yr rolling, runs; 100 = neutral).
# Update once a season from savant's park factors leaderboard.
PARK_FACTORS = {
    "COL": 1.28, "BOS": 1.08, "CIN": 1.07, "KC": 1.05, "AZ": 1.04,
    "PHI": 1.03, "TEX": 1.02, "ATL": 1.02, "LAA": 1.01, "TOR": 1.01,
    "BAL": 1.00, "MIN": 1.00, "WSH": 1.00, "HOU": 1.00, "CHC": 0.99,
    "PIT": 0.99, "STL": 0.99, "MIA": 0.98, "NYY": 0.98, "ATH": 0.98,
    "DET": 0.97, "CLE": 0.97, "LAD": 0.97, "MIL": 0.97, "NYM": 0.96,
    "SD": 0.96, "CWS": 0.96, "TB": 0.95, "SF": 0.94, "SEA": 0.92,
}


# ---------------------------------------------------------------------------
# Statcast aggregation
# ---------------------------------------------------------------------------
def _season_start(today: dt.date) -> dt.date:
    return dt.date(today.year, 3, 15)


def fetch_statcast_season(today: dt.date, force: bool = False) -> pd.DataFrame:
    """Download (or load cached) pitch-level Statcast data for the season."""
    from pybaseball import statcast

    path = CACHE / f"statcast_{today.year}.parquet"
    start = _season_start(today)
    if path.exists() and not force:
        df = pd.read_parquet(path)
        have_through = pd.to_datetime(df["game_date"]).max().date()
        if have_through >= today - dt.timedelta(days=1):
            return df
        # incremental top-up
        new = statcast(str(have_through + dt.timedelta(days=1)), str(today))
        df = pd.concat([df, new], ignore_index=True)
    else:
        df = statcast(str(start), str(today))
    df.to_parquet(path, index=False)
    return df


PA_END_EVENTS = {
    "single", "double", "triple", "home_run", "walk", "strikeout",
    "field_out", "force_out", "grounded_into_double_play", "hit_by_pitch",
    "sac_fly", "sac_bunt", "field_error", "fielders_choice",
    "fielders_choice_out", "double_play", "strikeout_double_play",
    "other_out", "triple_play", "catcher_interf",
}


def _rate(series_num, series_den):
    den = series_den if np.ndim(series_den) == 0 else series_den
    return np.where(den > 0, series_num / den, np.nan)


def aggregate_batters(sc: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """Per-batter skill table from pitch-level Statcast."""
    sc = sc.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"])
    pa = sc[sc["events"].notna() & sc["events"].isin(PA_END_EVENTS)]
    recent_cut = pd.Timestamp(today - dt.timedelta(days=21))

    def agg(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        g = frame.groupby("batter")
        out = pd.DataFrame({
            f"{prefix}pa": g.size(),
            f"{prefix}xwoba": g["estimated_woba_using_speedangle"].mean(),
            f"{prefix}k_pct": g["events"].apply(lambda s: s.str.startswith("strikeout").mean()),
            f"{prefix}bb_pct": g["events"].apply(lambda s: (s == "walk").mean()),
        })
        return out

    base = agg(pa)

    # batted-ball & plate-discipline metrics need all pitches
    swings = sc[sc["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
        "hit_into_play",
    ])]
    whiffs = swings["description"].str.startswith("swinging").groupby(swings["batter"]).mean()
    out_zone = sc[(sc["zone"] > 9)]
    chase = out_zone["description"].isin(
        ["swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"]
    ).groupby(out_zone["batter"]).mean()
    bbe = sc[sc["type"] == "X"]
    barrel = (bbe["launch_speed_angle"] == 6).groupby(bbe["batter"]).mean()
    hard = (bbe["launch_speed"] >= 95).groupby(bbe["batter"]).mean()

    vs_l = agg(pa[pa["p_throws"] == "L"], "vl_")
    vs_r = agg(pa[pa["p_throws"] == "R"], "vr_")
    recent = agg(pa[pa["game_date"] >= recent_cut], "recent_")

    df = base.join([vs_l, vs_r, recent], how="left")
    df["whiff_pct"] = whiffs
    df["chase_pct"] = chase
    df["barrel_pct"] = barrel
    df["hard_hit_pct"] = hard
    df["bats"] = pa.groupby("batter")["stand"].agg(
        lambda s: "S" if s.nunique() > 1 else s.iloc[0]
    )
    return df


def aggregate_pitchers(sc: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """Per-pitcher skill table, including a fastball velocity trend."""
    sc = sc.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"])
    pa = sc[sc["events"].notna() & sc["events"].isin(PA_END_EVENTS)]
    recent_cut = pd.Timestamp(today - dt.timedelta(days=30))

    def agg(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        g = frame.groupby("pitcher")
        return pd.DataFrame({
            f"{prefix}tbf": g.size(),
            f"{prefix}xwoba": g["estimated_woba_using_speedangle"].mean(),
            f"{prefix}k_pct": g["events"].apply(lambda s: s.str.startswith("strikeout").mean()),
            f"{prefix}bb_pct": g["events"].apply(lambda s: (s == "walk").mean()),
        })

    base = agg(pa)
    vs_l = agg(pa[pa["stand"] == "L"], "vl_")
    vs_r = agg(pa[pa["stand"] == "R"], "vr_")
    recent = agg(pa[pa["game_date"] >= recent_cut], "recent_")

    swings = sc[sc["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
        "hit_into_play",
    ])]
    whiffs = swings["description"].str.startswith("swinging").groupby(swings["pitcher"]).mean()
    bbe = sc[sc["type"] == "X"]
    barrel = (bbe["launch_speed_angle"] == 6).groupby(bbe["pitcher"]).mean()
    gb = (bbe["bb_type"] == "ground_ball").groupby(bbe["pitcher"]).mean()

    fb = sc[sc["pitch_type"].isin(["FF", "SI", "FC"])]
    season_velo = fb.groupby("pitcher")["release_speed"].mean()
    recent_velo = fb[fb["game_date"] >= recent_cut].groupby("pitcher")["release_speed"].mean()

    df = base.join([vs_l, vs_r, recent], how="left")
    df["whiff_pct"] = whiffs
    df["barrel_pct"] = barrel
    df["gb_pct"] = gb
    df["velo_trend"] = (recent_velo - season_velo).fillna(0.0)
    df["throws"] = pa.groupby("pitcher")["p_throws"].first()
    return df


# ---------------------------------------------------------------------------
# Schedule / lineups (MLB Stats API via `statsapi`)
# ---------------------------------------------------------------------------
def todays_games(date: dt.date) -> list[dict]:
    import statsapi

    sched = statsapi.schedule(date=date.strftime("%m/%d/%Y"))
    games = []
    for g in sched:
        games.append({
            "game_id": g["game_id"],
            "home": g["home_name"], "away": g["away_name"],
            "home_abbr": _abbr(g["home_name"]), "away_abbr": _abbr(g["away_name"]),
            "home_probable": g.get("home_probable_pitcher") or None,
            "away_probable": g.get("away_probable_pitcher") or None,
        })
    return games


def game_lineups(game_id: int) -> dict:
    """Posted lineups if available (usually 1-4h before first pitch);
    falls back to empty lists (caller then uses recent-lineup fallback)."""
    import statsapi

    box = statsapi.boxscore_data(game_id)
    out = {}
    for side in ("home", "away"):
        ids = box.get(side, {}).get("battingOrder", []) or []
        names = []
        for pid in ids:
            info = box[side]["players"].get(f"ID{pid}", {})
            names.append((pid, info.get("person", {}).get("fullName", str(pid))))
        out[side] = names
    return out


_ABBR = {
    "Arizona Diamondbacks": "AZ", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "ATH", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}


def _abbr(name: str) -> str:
    return _ABBR.get(name, name[:3].upper())


# ---------------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------------
def batter_profile(row: pd.Series, name: str) -> BatterProfile:
    g = lambda k, d: row[k] if k in row and pd.notna(row[k]) else d
    return BatterProfile(
        name=name,
        bats=g("bats", "R"),
        pa=int(g("pa", 0)),
        xwoba=g("xwoba", LEAGUE["xwoba"]),
        k_pct=g("k_pct", LEAGUE["k_pct"]),
        bb_pct=g("bb_pct", LEAGUE["bb_pct"]),
        barrel_pct=g("barrel_pct", LEAGUE["barrel_pct"]),
        hard_hit_pct=g("hard_hit_pct", LEAGUE["hard_hit_pct"]),
        chase_pct=g("chase_pct", LEAGUE["chase_pct"]),
        whiff_pct=g("whiff_pct", LEAGUE["whiff_pct"]),
        xwoba_vs_l=g("vl_xwoba", None), pa_vs_l=int(g("vl_pa", 0)),
        xwoba_vs_r=g("vr_xwoba", None), pa_vs_r=int(g("vr_pa", 0)),
        recent_xwoba=g("recent_xwoba", None), recent_pa=int(g("recent_pa", 0)),
    )


def pitcher_profile(row: pd.Series, name: str) -> PitcherProfile:
    g = lambda k, d: row[k] if k in row and pd.notna(row[k]) else d
    return PitcherProfile(
        name=name,
        throws=g("throws", "R"),
        tbf=int(g("tbf", 0)),
        xwoba_allowed=g("xwoba", LEAGUE["xwoba"]),
        k_pct=g("k_pct", LEAGUE["k_pct"]),
        bb_pct=g("bb_pct", LEAGUE["bb_pct"]),
        barrel_pct_allowed=g("barrel_pct", LEAGUE["barrel_pct"]),
        whiff_pct=g("whiff_pct", LEAGUE["whiff_pct"]),
        gb_pct=g("gb_pct", LEAGUE["gb_pct"]),
        xwoba_vs_l=g("vl_xwoba", None), tbf_vs_l=int(g("vl_tbf", 0)),
        xwoba_vs_r=g("vr_xwoba", None), tbf_vs_r=int(g("vr_tbf", 0)),
        recent_xwoba=g("recent_xwoba", None), recent_tbf=int(g("recent_tbf", 0)),
        velo_trend=float(g("velo_trend", 0.0)),
    )


# ---------------------------------------------------------------------------
# Bullpen aggregation with day-of availability
# ---------------------------------------------------------------------------
def aggregate_bullpens(sc: pd.DataFrame, today: dt.date,
                       window_days: int = 45) -> dict:
    """
    Per-team bullpen profile for `today`, availability-adjusted.

    Relievers = every pitcher in the last `window_days` who wasn't the first
    pitcher of his team's game (i.e., not the starter/opener that day).
    Availability rules (from actual pitch counts in the Statcast data):
      - threw 25+ pitches yesterday            -> unavailable (weight 0)
      - pitched on BOTH of the last two days   -> unavailable (weight 0)
      - threw 15-24 pitches yesterday          -> compromised (weight 0.5)
    Each reliever's skill is then weighted by (recent usage share x
    availability), so losing your two highest-leverage arms hurts the
    profile the way it hurts a real bullpen.

    Returns {team_abbr: engine.BullpenProfile}.
    """
    from .engine import BullpenProfile, shrink, LEAGUE

    sc = sc.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"]).dt.date
    sc["pitching_team"] = np.where(
        sc["inning_topbot"] == "Top", sc["home_team"], sc["away_team"]
    )
    recent = sc[sc["game_date"] >= today - dt.timedelta(days=window_days)]
    if recent.empty:
        return {}

    # first pitcher of each team-game = starter; everyone else = relief
    order = recent.sort_values("at_bat_number")
    starters = (
        order.groupby(["game_pk", "pitching_team"])["pitcher"]
        .first().rename("starter_id").reset_index()
    )
    rec = recent.merge(starters, on=["game_pk", "pitching_team"], how="left")
    rp = rec[rec["pitcher"] != rec["starter_id"]]

    yday = today - dt.timedelta(days=1)
    d2 = today - dt.timedelta(days=2)
    # id -> name map (Statcast's player_name column is the pitcher's name)
    if "player_name" in rec.columns:
        name_map = rec.groupby("pitcher")["player_name"].first().to_dict()
    else:
        name_map = {}
    # workload from ALL appearances (incl. spot starts / opener duty), so a
    # reliever who opened yesterday still shows up as unavailable today
    all_counts = rec.groupby(["pitcher", "game_date"]).size()

    def pitches_on(day: dt.date, ids) -> pd.Series:
        if day in all_counts.index.get_level_values(1):
            return all_counts.xs(day, level="game_date").reindex(ids).fillna(0)
        return pd.Series(0.0, index=ids)

    pens: dict = {}
    for team, grp in rp.groupby("pitching_team"):
        pa = grp[grp["events"].notna() & grp["events"].isin(PA_END_EVENTS)]
        if pa.empty:
            continue
        g = pa.groupby("pitcher")
        stats = pd.DataFrame({
            "tbf": g.size(),
            "xwoba": g["estimated_woba_using_speedangle"].mean(),
            "k_pct": g["events"].apply(lambda s: s.str.startswith("strikeout").mean()),
        })
        stats["p_yday"] = pitches_on(yday, stats.index)
        stats["p_d2"] = pitches_on(d2, stats.index)

        stats["avail"] = np.select(
            [
                (stats["p_yday"] >= 25) | ((stats["p_yday"] > 0) & (stats["p_d2"] > 0)),
                stats["p_yday"] >= 15,
            ],
            [0.0, 0.5],
            default=1.0,
        )
        usage = stats["tbf"] / stats["tbf"].sum()
        w = usage * stats["avail"]
        avail_frac = float(w.sum())            # 1.0 = fully rested pen
        if w.sum() == 0:
            w = usage                          # everyone gassed: use them anyway
        xw_raw = float((stats["xwoba"].fillna(LEAGUE["xwoba"]) * w).sum() / w.sum())
        k_raw = float((stats["k_pct"].fillna(LEAGUE["k_pct"]) * w).sum() / w.sum())
        down = stats[stats["avail"] == 0.0]

        pens[team] = BullpenProfile(
            team=team,
            xwoba_allowed=xw_raw,
            k_pct=k_raw,
            tbf=int(stats["tbf"].sum()),
            avail_frac=round(avail_frac, 2),
            n_unavailable=int(len(down)),
            unavailable=[name_map.get(i, i) for i in down.index],
        )
    return pens
