# MLB Daily Matchup Model

Projects every MLB starting pitcher against the actual lineup he'll face
each day — and every hitter against the starter he'll face — using
predictive skill metrics, not surface stats. Free data only (Baseball
Savant Statcast via `pybaseball` + the MLB Stats API); no API keys.

## Quick start

```bash
pip install -r requirements.txt

python predict_today.py --demo          # offline demo, proves the pipeline
python predict_today.py                 # today's live slate
python predict_today.py --date 2026-07-05
python tests/test_engine.py             # sanity tests (offline)
```

Output: a **game board** (win probability for every game), a ranked pitcher
matchup board (grade, expected xwOBA allowed, expected runs/9), the best
individual hitter-vs-starter edges of the day, and CSVs in `out/`.

## Game win probabilities

Each game's win probability combines: both starters' matchup projections vs.
the actual opposing lineups, expected starter innings (better matchups go
deeper), park, home field (~54/46 baseline), and — the differentiator —
**bullpen quality adjusted for who's actually available today**:

- A reliever who threw **25+ pitches yesterday** or pitched **both of the
  last two days** is treated as unavailable (weight 0).
- **15-24 pitches yesterday** = compromised (half weight).
- Each reliever is weighted by his recent usage share × availability, so
  losing the closer and setup man hurts far more than losing a mop-up arm,
  and a pen with most of its usage-weighted innings down takes an extra
  leverage penalty.
- Unavailable names print under each game line so you can sanity-check.

Remaining innings after the projected starter exit are charged at the
availability-adjusted pen rate against the opposing lineup's strength.
Win expectancy comes from pythagenpat on the two expected run totals.
Note: availability is inferred from pitch counts in the data — it can't see
IL moves or announced closer rest, so treat the notes as a strong prior,
not gospel.

**Heads up:** the first live run downloads the season's pitch-level Statcast
data (~1-2 GB, takes a while). It's cached in `cache/` and topped up
incrementally, so every later run is fast.

## How it works

1. **Skill estimation, not outcome stats.** Every player is profiled from
   pitch-level Statcast data: xwOBA (not AVG/ERA), K% and BB%, barrel% and
   hard-hit%, whiff% and chase%, ground-ball rate. These regress to future
   performance far better than the traditional stats.
2. **Empirical-Bayes shrinkage.** Every rate is blended toward league
   average based on sample size, using priors near each stat's research
   stabilization point (K% stabilizes in ~60 PA; xwOBA needs ~300). A hot
   40-PA stretch can't hijack a projection.
3. **Platoon-aware, both directions.** Each hitter's xwOBA vs. the
   starter's hand meets the starter's xwOBA allowed vs. that hitter's side.
   Individual platoon splits are shrunk hard toward the generic platoon
   effect (they're notoriously noisy), switch-hitters are handled, and the
   lineup is weighted by PA share per slot (leadoff ≈ 25% more PAs than
   the 9-hole).
4. **Odds-ratio combination.** Batter rate × pitcher rate relative to
   league rate — the sabermetric standard for merging the two sides into
   one expected rate per matchup.
5. **Form and health.** Recent xwOBA (last ~5 starts / ~3 weeks) gets a
   small, heavily-shrunk weight, and fastball velocity trend is a
   first-class input: a starter losing velo is penalized (~6 pts of xwOBA
   per mph), one of the few real leading indicators in-season.
6. **Context.** Park run factors (edit yearly in `mlb_matchup/data.py`)
   and home/away. Lineups are used once posted; before that, the model
   falls back to a league-average lineup and flags it — re-run closer to
   first pitch.
7. **K-specific interaction.** Expected strikeout rate per matchup includes
   a whiff×chase interaction: chase-prone hitters get an extra bump against
   high-whiff pitchers.

Every game's output includes per-batter lines (batter xwOBA vs. hand,
pitcher xwOBA allowed vs. side, combined matchup xwOBA, matchup K%), so you
can see exactly *why* a spot grades the way it does.

## Backtesting against past seasons

```bash
python backtest.py --year 2025
python backtest.py --year 2026                       # season to date
python backtest.py --year 2025 --start 2025-06-01 --end 2025-09-28
```

Replays every start in the window with point-in-time features (only data
available before that day), compares projections to the runs/strikeouts
each starter actually recorded, and reports: MAE vs. naive baselines,
Spearman rank correlation, a calibration table by grade bucket, and the
top-vs-bottom decile spread. Per-start detail lands in `out/backtest_<year>.csv`.

How to judge the results: per-start MAE will always look large (single
games are noisy) — what matters is beating the always-predict-the-mean
baseline, a calibration table where actual runs fall as grade rises, and
a decile spread of roughly a run per start or more.

## Optional ML calibration layer

```bash
python train_model.py                # current season
python train_model.py --year 2025   # a past season
```

Replays the season start-by-start with **point-in-time features** (only
data available before each start — no leakage), labels each start with the
runs and strikeouts the pitcher actually recorded, and fits shallow
gradient-boosting models with time-series cross-validation. The trained
model (`cache/gbm.pkl`) is picked up automatically by `predict_today.py`
and reported alongside the analytic grades.

## Reading the grades

50 = league-average spot. 60+ = strong. 40- = avoid. The demo illustrates
the spread: an ace facing a whiff-prone weak lineup in a pitcher's park
grades ~95; a wild, velo-declining lefty at a Coors-like park grades ~23.

## Honest limitations

- **Single games are mostly noise.** The model separates a 3.8 from a 5.5
  expected-runs spot; it cannot promise an ace won't get shelled tonight.
  Judge it on calibration over weeks. Expect CV MAE around ~1.9-2.1
  runs/start — that's the irreducible noise of baseball, not a bug.
- **Career batter-vs-pitcher history ("4-for-9 lifetime") is deliberately
  excluded.** Those samples are the classic trap of naive matchup analysis;
  platoon splits + skill metrics carry the actual signal.
- Bullpens, weather, umpires, and catcher framing aren't modeled. Clean
  extension points: add a bullpen table in `data.py`, or a
  temperature/wind multiplier next to the park factor in `engine.py`.
- Early-season and just-called-up players ride league-average priors until
  their samples grow; the shrinkage makes this graceful, not silent —
  check `pa`/`tbf` in the profiles if a grade looks off.
- Be polite to Baseball Savant: the cache exists so you only download each
  day's pitches once.

## Layout

```
predict_today.py         daily entry point (live or --demo)
train_model.py           optional: fit the GBM calibration layer
mlb_matchup/engine.py    statistical core: profiles, shrinkage, odds-ratio,
                         platoon logic, game projection (pure, no I/O)
mlb_matchup/data.py      Statcast download/cache + aggregation, schedule,
                         lineups, park factors
mlb_matchup/model.py     point-in-time training frame + GBM + predict()
tests/test_engine.py     offline sanity tests
```
