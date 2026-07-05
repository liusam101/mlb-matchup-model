"""Offline sanity tests for the matchup engine (no network needed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mlb_matchup.engine import (
    BatterProfile, PitcherProfile, project_matchup, shrink, odds_ratio, LEAGUE,
)


def test_shrinkage_pulls_small_samples_to_league():
    hot_small = shrink(0.450, 30, LEAGUE["xwoba"], 300)
    hot_large = shrink(0.450, 600, LEAGUE["xwoba"], 300)
    assert hot_small < hot_large
    assert abs(hot_small - LEAGUE["xwoba"]) < 0.02   # 30 PA barely moves it


def test_odds_ratio_symmetry():
    # avg batter vs avg pitcher = league average
    assert abs(odds_ratio(0.312, 0.312, 0.312) - 0.312) < 1e-9
    # good batter vs bad pitcher > either alone
    assert odds_ratio(0.360, 0.340, 0.312) > 0.360


def test_platoon_advantage_direction():
    b = BatterProfile(name="L bat", bats="L", pa=500, xwoba=0.330)
    assert b.skill_xwoba("R") > b.skill_xwoba("L")


def test_better_pitcher_grades_higher():
    lineup = [BatterProfile(name=f"b{i}", bats="R", pa=400) for i in range(9)]
    ace = PitcherProfile(name="ace", throws="R", tbf=500,
                         xwoba_allowed=0.260, k_pct=0.32)
    scrub = PitcherProfile(name="scrub", throws="R", tbf=500,
                           xwoba_allowed=0.360, k_pct=0.16)
    ra = project_matchup(ace, lineup)
    rs = project_matchup(scrub, lineup)
    assert ra.matchup_grade > rs.matchup_grade
    assert ra.exp_runs_per_9 < rs.exp_runs_per_9


def test_park_factor_moves_runs():
    lineup = [BatterProfile(name=f"b{i}", bats="R", pa=400) for i in range(9)]
    p = PitcherProfile(name="p", throws="R", tbf=500)
    coors = project_matchup(p, lineup, park_factor=1.28)
    tmobile = project_matchup(p, lineup, park_factor=0.92)
    assert coors.exp_runs_per_9 > tmobile.exp_runs_per_9





def test_game_projection_favors_better_side():
    from mlb_matchup.engine import BullpenProfile, project_game
    good = [BatterProfile(name=f"g{i}", bats="R", pa=400, xwoba=0.345) for i in range(9)]
    weak = [BatterProfile(name=f"w{i}", bats="R", pa=400, xwoba=0.285) for i in range(9)]
    ace = PitcherProfile(name="ace", throws="R", tbf=500, xwoba_allowed=0.265, k_pct=0.31)
    scrub = PitcherProfile(name="scrub", throws="R", tbf=500, xwoba_allowed=0.355, k_pct=0.17)
    pen = BullpenProfile(tbf=800)
    # home team: ace + good lineup; away: scrub + weak lineup
    home_res = project_matchup(ace, weak, is_home=True)      # ace faces weak lineup
    away_res = project_matchup(scrub, good, is_home=False)   # scrub faces good lineup
    gp = project_game(home_res, away_res, pen, pen, good, weak)
    assert gp.home_wp > 0.62, f"stacked home side should be a big favorite, got {gp.home_wp}"


def test_gassed_bullpen_lowers_win_prob():
    from mlb_matchup.engine import BullpenProfile, project_game
    lineup = [BatterProfile(name=f"b{i}", bats="R", pa=400) for i in range(9)]
    p = PitcherProfile(name="p", throws="R", tbf=500)
    res_h = project_matchup(p, lineup, is_home=True)
    res_a = project_matchup(p, lineup, is_home=False)
    fresh = BullpenProfile(tbf=800, xwoba_allowed=0.300, avail_frac=1.0)
    gassed = BullpenProfile(tbf=800, xwoba_allowed=0.300, avail_frac=0.40, n_unavailable=3)
    wp_fresh = project_game(res_h, res_a, fresh, fresh, lineup, lineup).home_wp
    wp_gassed = project_game(res_h, res_a, gassed, fresh, lineup, lineup).home_wp
    assert wp_gassed < wp_fresh, "home team with gassed pen should win less often"


def test_home_field_advantage_in_even_game():
    from mlb_matchup.engine import BullpenProfile, project_game
    lineup = [BatterProfile(name=f"b{i}", bats="R", pa=400) for i in range(9)]
    p = PitcherProfile(name="p", throws="R", tbf=500)
    res_h = project_matchup(p, lineup, is_home=True)
    res_a = project_matchup(p, lineup, is_home=False)
    pen = BullpenProfile(tbf=800)
    wp = project_game(res_h, res_a, pen, pen, lineup, lineup).home_wp
    assert 0.51 < wp < 0.58, f"even game should sit near 54% home, got {wp}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All engine tests passed.")
