"""
Step 11b - Does the injury layer actually help? (run SECOND)
-------------------------------------------------------------
The experiment:
  1. Build team ratings using ONLY matches played before the 2022
     World Cup kicked off (2022-11-19). No peeking at the future.
  2. Predict all 64 real WC2022 matches two ways:
        Model A - ignore injuries entirely (the Step 7 engine)
        Model B - apply the real reported absences through the
                  Step 10 condition layer
  3. Score both with RPS against what actually happened.

If Model B's RPS is lower, the injury data genuinely improved the
predictions. If not, our importance weights need rethinking - and
that is a perfectly valid scientific result too.

Needs wc2022_injuries.json from step11_fetch.py in the same folder.
(Without it, the script still runs and just scores Model A.)

HONESTY NOTES:
  - 64 matches is a small sample; treat the verdict as a strong
    hint, not proof.
  - Every fetched absence is weighted as tier "key", role "both"
    (the API can't tell us who the stars were).

Run:  venv/bin/python step11_backtest.py
"""

import json
import math
import os
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
TOURNAMENT_START = "2022-11-19"
RATINGS_SINCE = "2018-01-01"
HALF_LIFE_DAYS = 365
MIN_GAMES, ITERATIONS, MAX_GOALS, RHO = 10, 25, 6, -0.13

INJURY_FILE = "wc2022_injuries.json"
# Every reported absence is treated as a "key" player affecting "both"
KEY_HIT = 0.10


# ---------------------------------------------------------------
# Ratings as of the day before the tournament (Step 7 engine)
# ---------------------------------------------------------------
def build_pre_tournament_ratings(df):
    train = df[(df["date"] >= RATINGS_SINCE) & (df["date"] < TOURNAMENT_START)]
    counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
    universe = set(counts[counts >= MIN_GAMES].index)
    m = train[train["home_team"].isin(universe) & train["away_team"].isin(universe)].copy()

    ref = pd.Timestamp(TOURNAMENT_START)
    m["age"] = (ref - m["date"]).dt.days
    m["weight"] = 0.5 ** (m["age"] / HALF_LIFE_DAYS)

    team_matches = {t: [] for t in universe}
    for row in m.itertuples(index=False):
        team_matches[row.home_team].append((row.away_team, row.home_score, row.away_score, row.weight))
        team_matches[row.away_team].append((row.home_team, row.away_score, row.home_score, row.weight))

    ws = {t: sum(w * gs for _, gs, gc, w in team_matches[t]) for t in universe}
    wc_ = {t: sum(w * gc for _, gs, gc, w in team_matches[t]) for t in universe}
    total_w = m["weight"].sum() * 2
    mu = (m["weight"] * (m["home_score"] + m["away_score"])).sum() / total_w

    attack = {t: 1.0 for t in universe}; defense = {t: 1.0 for t in universe}
    for _ in range(ITERATIONS):
        na, nd = {}, {}
        for t in universe:
            od = sum(w * defense[o] for o, gs, gc, w in team_matches[t])
            oa = sum(w * attack[o] for o, gs, gc, w in team_matches[t])
            na[t] = ws[t] / (mu * od) if od else 1.0
            nd[t] = wc_[t] / (mu * oa) if oa else 1.0
        am = sum(na.values()) / len(na); dm = sum(nd.values()) / len(nd)
        attack = {t: v / am for t, v in na.items()}
        defense = {t: v / dm for t, v in nd.items()}
    return attack, defense, mu


# ---------------------------------------------------------------
# Prediction (Dixon-Coles, neutral venue - Qatar hosted, and Qatar
# matches keep neutral treatment for simplicity)
# ---------------------------------------------------------------
def poisson_probability(k, expected):
    return (expected ** k) * math.exp(-expected) / math.factorial(k)


def dixon_coles_tau(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def outcome_probs(exp_a, exp_b):
    pa = [poisson_probability(g, exp_a) for g in range(MAX_GOALS + 1)]
    pb = [poisson_probability(g, exp_b) for g in range(MAX_GOALS + 1)]
    w = d = l = total = 0.0
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            p = pa[x] * pb[y] * dixon_coles_tau(x, y, exp_a, exp_b, RHO)
            total += p
            if x > y: w += p
            elif x == y: d += p
            else: l += p
    return [w / total, d / total, l / total]


def condition_mults(n_absent):
    """n key players out -> attack shrinks, defense leaks (split roles)."""
    att = (1 - KEY_HIT * 0.5) ** n_absent
    deff = (1 + KEY_HIT * 0.5) ** n_absent
    return att, deff


def ranked_probability_score(pred, outcome):
    cp = [pred[0], pred[0] + pred[1]]
    co = [outcome[0], outcome[0] + outcome[1]]
    return ((cp[0] - co[0]) ** 2 + (cp[1] - co[1]) ** 2) / 2


# ---------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------
def main():
    print("Downloading match data...")
    df = pd.read_csv(DATA_URL).dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])

    wc = df[(df["tournament"] == "FIFA World Cup") &
            (df["date"] >= "2022-11-01") & (df["date"] <= "2022-12-31")]
    print(f"Found {len(wc)} World Cup 2022 matches to predict.")

    attack, defense, mu = build_pre_tournament_ratings(df)

    injuries = {}
    if os.path.exists(INJURY_FILE):
        with open(INJURY_FILE) as f:
            injuries = json.load(f)
        print(f"Loaded {INJURY_FILE}: {len(injuries)} team-match injury entries.")
    else:
        print(f"({INJURY_FILE} not found - scoring Model A only.)")

    unmatched = set()
    n = 0
    rps_a = rps_b = 0.0
    acc_a = acc_b = 0
    injury_matches = 0

    for row in wc.itertuples(index=False):
        h, a = row.home_team, row.away_team
        if h not in attack or a not in attack:
            continue
        date = str(row.date)[:10]
        n += 1

        exp_h = attack[h] * defense[a] * mu
        exp_a_ = attack[a] * defense[h] * mu

        # Model A: no injuries
        pred_a = outcome_probs(exp_h, exp_a_)

        # Model B: apply reported absences
        miss_h = injuries.get(f"{h}|{date}", [])
        miss_a = injuries.get(f"{a}|{date}", [])
        if miss_h or miss_a:
            injury_matches += 1
        h_att, h_def = condition_mults(len(miss_h))
        a_att, a_def = condition_mults(len(miss_a))
        pred_b = outcome_probs(exp_h * h_att * a_def, exp_a_ * a_att * h_def)

        outcome = ([1, 0, 0] if row.home_score > row.away_score
                   else [0, 1, 0] if row.home_score == row.away_score
                   else [0, 0, 1])
        truth = outcome.index(1)

        rps_a += ranked_probability_score(pred_a, outcome)
        rps_b += ranked_probability_score(pred_b, outcome)
        if pred_a.index(max(pred_a)) == truth: acc_a += 1
        if pred_b.index(max(pred_b)) == truth: acc_b += 1

    # check for team names in the injury file that never matched
    wc_teams = set(wc["home_team"]) | set(wc["away_team"])
    for key in injuries:
        team = key.split("|")[0]
        if team not in wc_teams:
            unmatched.add(team)

    print(f"\nScored {n} matches. Injury data covered {injury_matches} of them.")
    print(f"\n  {'':24}{'RPS':>10}{'Accuracy':>12}")
    print(f"  {'Model A (no injuries)':<24}{rps_a / n:>10.4f}{acc_a / n * 100:>11.1f}%")
    if injuries:
        print(f"  {'Model B (with injuries)':<24}{rps_b / n:>10.4f}{acc_b / n * 100:>11.1f}%")
        diff = (rps_a - rps_b) / rps_a * 100
        if rps_b < rps_a:
            print(f"\n  VERDICT: injuries IMPROVED predictions (RPS {diff:.1f}% better).")
        elif rps_b > rps_a:
            print(f"\n  VERDICT: injuries made predictions WORSE (RPS {-diff:.1f}% worse).")
        else:
            print("\n  VERDICT: no measurable difference.")
        print("  (64 matches is a small sample - treat this as a hint, not proof.)")
    if unmatched:
        print(f"\n  WARNING - injury team names that matched no WC team: {sorted(unmatched)}")
        print("  Add these to NAME_FIX in step11_fetch.py and re-run both scripts.")


if __name__ == "__main__":
    main()
