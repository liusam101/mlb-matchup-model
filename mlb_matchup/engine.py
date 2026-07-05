"""
Matchup engine: the statistical core of the model.

Philosophy
----------
Single-game baseball is mostly noise. The way to find real signal is to:
  1. Use *skill estimators* instead of outcome stats (xwOBA not AVG,
     K-BB% not ERA, barrel% not hits allowed).
  2. Shrink everything toward league average based on sample size
     (empirical Bayes) so a hot 40-PA stretch doesn't dominate.
  3. Combine pitcher-vs-batter with the odds-ratio method (a
     sabermetric standard for combining a pitcher's rate allowed with
     a batter's rate produced relative to league average).
  4. Adjust for context: platoon handedness, park, home/away.
  5. Blend season-long skill with recent form using exponential decay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# League baselines (updated automatically from data when available; these are
# fallbacks roughly matching recent MLB environments).
# ---------------------------------------------------------------------------
LEAGUE = {
    "woba": 0.312,
    "xwoba": 0.312,
    "k_pct": 0.222,
    "bb_pct": 0.082,
    "barrel_pct": 0.077,   # per batted ball event
    "hard_hit_pct": 0.395,
    "whiff_pct": 0.247,
    "chase_pct": 0.286,
    "gb_pct": 0.435,
}

# Shrinkage priors: the number of "phantom" league-average PA added to each
# observed sample. Chosen near the stabilization points from research on
# when stats become reliable (K% stabilizes fast, wOBA-type stats slowly).
SHRINK_PA = {
    "xwoba": 300,
    "k_pct": 60,
    "bb_pct": 120,
    "barrel_pct": 200,
    "whiff_pct": 70,
    "chase_pct": 70,
    "hard_hit_pct": 160,
}

# Generic platoon prior: how much better hitters are with the platoon
# advantage, expressed as additive xwOBA. Splits are shrunk hard because
# individual platoon splits are notoriously noisy.
PLATOON_XWOBA_EDGE = 0.020
PLATOON_SHRINK_PA = 1000


def shrink(observed: float, n: int, league_mean: float, prior_n: int) -> float:
    """Empirical-Bayes shrinkage: blend observed rate with league mean."""
    if n is None or n <= 0 or observed is None or math.isnan(observed):
        return league_mean
    return (observed * n + league_mean * prior_n) / (n + prior_n)


def odds_ratio(batter_rate: float, pitcher_rate: float, league_rate: float) -> float:
    """
    Odds-ratio method: expected rate when a batter with rate B faces a
    pitcher who allows rate P in a league with rate L.
    """

    def odds(p: float) -> float:
        p = min(max(p, 1e-4), 1 - 1e-4)
        return p / (1 - p)

    x = odds(batter_rate) * odds(pitcher_rate) / odds(league_rate)
    return x / (1 + x)


# ---------------------------------------------------------------------------
# Player profiles
# ---------------------------------------------------------------------------
@dataclass
class BatterProfile:
    name: str
    bats: str                      # 'L', 'R', or 'S'
    pa: int = 0
    xwoba: float = LEAGUE["xwoba"]
    k_pct: float = LEAGUE["k_pct"]
    bb_pct: float = LEAGUE["bb_pct"]
    barrel_pct: float = LEAGUE["barrel_pct"]
    hard_hit_pct: float = LEAGUE["hard_hit_pct"]
    chase_pct: float = LEAGUE["chase_pct"]
    whiff_pct: float = LEAGUE["whiff_pct"]
    # splits: observed xwOBA and PA vs each pitcher hand
    xwoba_vs_l: float | None = None
    pa_vs_l: int = 0
    xwoba_vs_r: float | None = None
    pa_vs_r: int = 0
    recent_xwoba: float | None = None   # last ~15 games, exp-weighted
    recent_pa: int = 0

    def effective_hand(self, pitcher_throws: str) -> str:
        if self.bats == "S":
            return "L" if pitcher_throws == "R" else "R"
        return self.bats

    def skill_xwoba(self, pitcher_throws: str) -> float:
        """Shrunk overall xwOBA, adjusted for platoon split and recent form."""
        base = shrink(self.xwoba, self.pa, LEAGUE["xwoba"], SHRINK_PA["xwoba"])

        # Platoon adjustment: shrink the player's own split hard toward the
        # generic platoon effect, then apply as a delta from their base.
        hand = self.effective_hand(pitcher_throws)
        advantage = hand != pitcher_throws
        generic = base + (PLATOON_XWOBA_EDGE / 2 if advantage else -PLATOON_XWOBA_EDGE / 2)
        split_obs = self.xwoba_vs_l if pitcher_throws == "L" else self.xwoba_vs_r
        split_pa = self.pa_vs_l if pitcher_throws == "L" else self.pa_vs_r
        if split_obs is not None and split_pa > 0:
            adjusted = shrink(split_obs, split_pa, generic, PLATOON_SHRINK_PA)
        else:
            adjusted = generic

        # Recent form: small weight, heavily shrunk (hot streaks are ~80% noise)
        if self.recent_xwoba is not None and self.recent_pa > 0:
            form = shrink(self.recent_xwoba, self.recent_pa, adjusted, 250)
            adjusted = 0.85 * adjusted + 0.15 * form
        return adjusted


@dataclass
class PitcherProfile:
    name: str
    throws: str                    # 'L' or 'R'
    tbf: int = 0                   # total batters faced
    xwoba_allowed: float = LEAGUE["xwoba"]
    k_pct: float = LEAGUE["k_pct"]
    bb_pct: float = LEAGUE["bb_pct"]
    barrel_pct_allowed: float = LEAGUE["barrel_pct"]
    whiff_pct: float = LEAGUE["whiff_pct"]
    gb_pct: float = LEAGUE["gb_pct"]
    xwoba_vs_l: float | None = None
    tbf_vs_l: int = 0
    xwoba_vs_r: float | None = None
    tbf_vs_r: int = 0
    recent_xwoba: float | None = None   # last ~5 starts, exp-weighted
    recent_tbf: int = 0
    velo_trend: float = 0.0        # recent FB velo minus season avg (mph)

    def skill_xwoba_allowed(self, batter_hand: str) -> float:
        base = shrink(self.xwoba_allowed, self.tbf, LEAGUE["xwoba"], SHRINK_PA["xwoba"])
        advantage = batter_hand != self.throws  # batter has platoon advantage
        generic = base + (PLATOON_XWOBA_EDGE / 2 if advantage else -PLATOON_XWOBA_EDGE / 2)
        split_obs = self.xwoba_vs_l if batter_hand == "L" else self.xwoba_vs_r
        split_tbf = self.tbf_vs_l if batter_hand == "L" else self.tbf_vs_r
        if split_obs is not None and split_tbf > 0:
            adjusted = shrink(split_obs, split_tbf, generic, PLATOON_SHRINK_PA)
        else:
            adjusted = generic
        if self.recent_xwoba is not None and self.recent_tbf > 0:
            form = shrink(self.recent_xwoba, self.recent_tbf, adjusted, 250)
            adjusted = 0.85 * adjusted + 0.15 * form
        # Velocity decline is one of the few leading indicators that matters
        # in-season: penalize ~6 pts of xwOBA per mph lost, credit gains
        # slightly less (gains are often just weather/adrenaline).
        if self.velo_trend < 0:
            adjusted += 0.006 * abs(self.velo_trend)   # losing velo -> worse
        else:
            adjusted -= 0.004 * self.velo_trend        # gaining velo -> better
        return adjusted

    def skill_k_pct(self, batter: "BatterProfile") -> float:
        p = shrink(self.k_pct, self.tbf, LEAGUE["k_pct"], SHRINK_PA["k_pct"])
        b = shrink(batter.k_pct, batter.pa, LEAGUE["k_pct"], SHRINK_PA["k_pct"])
        # whiff/chase interaction: whiff-prone + chase-prone hitters get an
        # extra bump against high-whiff pitchers.
        interaction = (
            shrink(self.whiff_pct, self.tbf, LEAGUE["whiff_pct"], SHRINK_PA["whiff_pct"])
            - LEAGUE["whiff_pct"]
        ) * (
            shrink(batter.chase_pct, batter.pa, LEAGUE["chase_pct"], SHRINK_PA["chase_pct"])
            - LEAGUE["chase_pct"]
        ) * 4.0
        return odds_ratio(b, p, LEAGUE["k_pct"]) + interaction


# ---------------------------------------------------------------------------
# Game-level projection
# ---------------------------------------------------------------------------
@dataclass
class MatchupResult:
    pitcher: str
    opponent: str
    park_factor: float
    exp_xwoba_allowed: float       # lineup-weighted expected xwOBA against
    exp_k_pct: float
    exp_runs_per_9: float
    matchup_grade: float           # 0-100, higher = better for the pitcher
    batter_lines: list = field(default_factory=list)


# Leadoff spots bat more often; weights approximate PA share by lineup slot.
LINEUP_WEIGHTS = [0.125, 0.122, 0.119, 0.116, 0.113, 0.110, 0.107, 0.104, 0.084]

# xwOBA -> runs/9 conversion. Runs scale roughly with wOBA^2 around the
# league mean; the linearization below is accurate in the normal range.
def xwoba_to_runs_per_9(xwoba: float, league_rpg: float = 4.4) -> float:
    return league_rpg * (xwoba / LEAGUE["xwoba"]) ** 2


def project_matchup(
    pitcher: PitcherProfile,
    lineup: list[BatterProfile],
    park_factor: float = 1.0,
    is_home: bool = True,
) -> MatchupResult:
    """Project a starting pitcher against a specific lineup."""
    weights = LINEUP_WEIGHTS[: len(lineup)]
    wsum = sum(weights)
    exp_xwoba, exp_k = 0.0, 0.0
    lines = []
    for w, batter in zip(weights, lineup):
        hand = batter.effective_hand(pitcher.throws)
        b_x = batter.skill_xwoba(pitcher.throws)
        p_x = pitcher.skill_xwoba_allowed(hand)
        # Odds-ratio on the "reaches-positively" scale via xwOBA treated as a
        # pseudo-rate (standard practice for wOBA-scale combinations).
        m_x = odds_ratio(b_x, p_x, LEAGUE["xwoba"])
        m_k = pitcher.skill_k_pct(batter)
        exp_xwoba += w * m_x
        exp_k += w * m_k
        lines.append(
            {
                "batter": batter.name,
                "hand": hand,
                "batter_xwoba": round(b_x, 3),
                "pitcher_xwoba_allowed": round(p_x, 3),
                "matchup_xwoba": round(m_x, 3),
                "matchup_k_pct": round(m_k, 3),
                "edge": round(LEAGUE["xwoba"] - m_x, 3),  # + favors pitcher
            }
        )
    exp_xwoba /= wsum
    exp_k /= wsum

    # Context: park and home-field (pitchers perform ~2% better at home).
    exp_xwoba *= park_factor ** 0.5           # park affects run env ~ sqrt on wOBA scale
    exp_xwoba *= 0.995 if is_home else 1.005

    rp9 = xwoba_to_runs_per_9(exp_xwoba)

    # Grade: 50 = league average; each .010 of xwOBA suppression ~ 5 pts.
    grade = 50 + (LEAGUE["xwoba"] - exp_xwoba) * 500 + (exp_k - LEAGUE["k_pct"]) * 100
    grade = max(0.0, min(100.0, grade))

    lines.sort(key=lambda r: r["matchup_xwoba"], reverse=True)
    return MatchupResult(
        pitcher=pitcher.name,
        opponent=", ".join(b.name for b in lineup[:3]) + ", ...",
        park_factor=park_factor,
        exp_xwoba_allowed=round(exp_xwoba, 3),
        exp_k_pct=round(exp_k, 3),
        exp_runs_per_9=round(rp9, 2),
        matchup_grade=round(grade, 1),
        batter_lines=lines,
    )


# ---------------------------------------------------------------------------
# Bullpens and full-game win probability
# ---------------------------------------------------------------------------
@dataclass
class BullpenProfile:
    """Availability-weighted bullpen skill for one team on one day.

    xwoba_allowed is already weighted by each reliever's recent usage share
    times his availability today (0 if he threw 25+ pitches yesterday or
    pitched both of the last two days; 0.5 if 15-24 pitches yesterday).
    """
    team: str = ""
    xwoba_allowed: float = LEAGUE["xwoba"]
    k_pct: float = LEAGUE["k_pct"]
    tbf: int = 0
    avail_frac: float = 1.0          # share of usage-weighted pen available
    n_unavailable: int = 0
    unavailable: list = field(default_factory=list)

    def effective_xwoba(self) -> float:
        xw = shrink(self.xwoba_allowed, self.tbf, LEAGUE["xwoba"], 250)
        # A taxed pen doesn't just lose quality (already priced into the
        # availability weighting) — the manager also loses leverage matching
        # and must stretch mop-up arms. Small extra penalty when most of the
        # usage-weighted pen is down.
        if self.avail_frac < 0.55:
            xw += 0.012
        elif self.avail_frac < 0.75:
            xw += 0.006
        return xw


@dataclass
class GameProjection:
    home_team: str
    away_team: str
    home_pitcher: str
    away_pitcher: str
    exp_home_runs: float
    exp_away_runs: float
    home_wp: float
    notes: list = field(default_factory=list)


def expected_starter_ip(exp_xwoba_allowed: float) -> float:
    """Better matchups -> deeper starts. Centered near the modern ~5.1 IP."""
    return float(min(7.0, max(3.7, 5.1 + (LEAGUE["xwoba"] - exp_xwoba_allowed) * 20)))


def lineup_strength(lineup: list["BatterProfile"]) -> float:
    """Shrunk overall xwOBA of a lineup (hand-neutral, for facing a pen)."""
    if not lineup:
        return LEAGUE["xwoba"]
    vals = [shrink(b.xwoba, b.pa, LEAGUE["xwoba"], SHRINK_PA["xwoba"]) for b in lineup]
    w = LINEUP_WEIGHTS[: len(vals)]
    return sum(v * x for v, x in zip(vals, w)) / sum(w)


def project_game(
    home_res: MatchupResult,
    away_res: MatchupResult,
    home_pen: BullpenProfile,
    away_pen: BullpenProfile,
    home_lineup: list[BatterProfile],
    away_lineup: list[BatterProfile],
    park_factor: float = 1.0,
) -> GameProjection:
    """
    Full-game projection: starter innings at the matchup-projected rate,
    remaining innings at the availability-adjusted bullpen rate vs. the
    opposing lineup, combined into expected runs and a win probability
    (pythagenpat + home-field advantage).

    home_res / away_res are each starter's project_matchup result vs. the
    OPPOSING lineup (park/home effects already applied there).
    """
    notes: list[str] = []

    def runs_against(starter_res: MatchupResult, pen: BullpenProfile,
                     opp_lineup: list[BatterProfile]) -> float:
        ip = expected_starter_ip(starter_res.exp_xwoba_allowed)
        starter_runs = starter_res.exp_runs_per_9 * ip / 9.0
        opp_xw = lineup_strength(opp_lineup)
        pen_xw = odds_ratio(opp_xw, pen.effective_xwoba(), LEAGUE["xwoba"])
        pen_xw *= park_factor ** 0.5
        pen_runs = xwoba_to_runs_per_9(pen_xw) * (9.0 - ip) / 9.0
        return starter_runs + pen_runs

    exp_away_runs = runs_against(home_res, home_pen, away_lineup)  # vs home staff
    exp_home_runs = runs_against(away_res, away_pen, home_lineup)  # vs away staff

    for pen, side in ((home_pen, "home"), (away_pen, "away")):
        if pen.n_unavailable:
            names = ", ".join(str(x) for x in pen.unavailable[:4])
            notes.append(f"{side} pen down {pen.n_unavailable}: {names}")

    # Pythagenpat win expectancy, then home-field bump (~54/46 baseline).
    rpg = exp_home_runs + exp_away_runs
    expo = max(1.2, rpg ** 0.287)
    wp = exp_home_runs ** expo / (exp_home_runs ** expo + exp_away_runs ** expo)
    odds = (wp / (1 - wp)) * 1.16          # home-field advantage on odds scale
    wp = odds / (1 + odds)

    return GameProjection(
        home_team="", away_team="",
        home_pitcher=home_res.pitcher, away_pitcher=away_res.pitcher,
        exp_home_runs=round(exp_home_runs, 2),
        exp_away_runs=round(exp_away_runs, 2),
        home_wp=round(wp, 3),
        notes=notes,
    )
