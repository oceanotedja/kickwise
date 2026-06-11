"""
World Cup Match Predictor - Step 5 (Backtesting / Evaluation)
--------------------------------------------------------------
A model that LOOKS confident is worthless until you measure it.
This script tests how good your predictor really is.

HOW BACKTESTING WORKS:
  1. Train the ratings on OLDER matches only (before a cutoff date).
  2. Predict NEWER matches the model has never seen.
  3. Compare the predictions to what actually happened.
This avoids "cheating": the model never sees the answer in advance.

THE SCORES WE REPORT:
  - Accuracy: % of matches where the most likely outcome was right.
              (Crude - it ignores how confident we were.)
  - RPS (Ranked Probability Score): the standard football metric.
              Rewards putting probability close to the truth. LOWER is better.
  - Log loss: punishes confident wrong predictions hard. LOWER is better.

We also score a simple BASELINE (just predict the average outcome
every time). If our model can't beat that, it isn't learning anything.

Run:  venv/bin/python predictor_step5.py
"""

import math
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

TRAIN_START = "2021-06-01"   # build ratings from matches in this window
CUTOFF      = "2025-06-01"   # everything before this = training, after = testing
TEST_END    = "2026-06-01"

MIN_GAMES, ITERATIONS, MAX_GOALS, RHO = 10, 25, 6, -0.13


# ---------------------------------------------------------------
# Ratings + prediction (same engine as Step 4)
# ---------------------------------------------------------------
def build_ratings(d):
    counts = pd.concat([d["home_team"], d["away_team"]]).value_counts()
    universe = set(counts[counts >= MIN_GAMES].index)
    m = d[d["home_team"].isin(universe) & d["away_team"].isin(universe)]
    mu = (m["home_score"].sum() + m["away_score"].sum()) / (len(m) * 2)

    stats = {}
    for t in universe:
        h = m[m["home_team"] == t]; a = m[m["away_team"] == t]
        stats[t] = {
            "scored": h["home_score"].sum() + a["away_score"].sum(),
            "conceded": h["away_score"].sum() + a["home_score"].sum(),
            "opponents": list(h["away_team"]) + list(a["home_team"]),
        }

    attack = {t: 1.0 for t in universe}; defense = {t: 1.0 for t in universe}
    for _ in range(ITERATIONS):
        na, nd = {}, {}
        for t in universe:
            od = sum(defense[o] for o in stats[t]["opponents"])
            oa = sum(attack[o] for o in stats[t]["opponents"])
            na[t] = stats[t]["scored"] / (mu * od) if od else 1.0
            nd[t] = stats[t]["conceded"] / (mu * oa) if oa else 1.0
        am = sum(na.values()) / len(na); dm = sum(nd.values()) / len(nd)
        attack = {t: v / am for t, v in na.items()}
        defense = {t: v / dm for t, v in nd.items()}
    return attack, defense, mu


def poisson_probability(k, expected):
    return (expected ** k) * math.exp(-expected) / math.factorial(k)


def dixon_coles_tau(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def predict(team_a, team_b, attack, defense, mu):
    """Returns [P(A win), P(draw), P(B win)]."""
    ea = attack[team_a] * defense[team_b] * mu
    eb = attack[team_b] * defense[team_a] * mu
    pa = [poisson_probability(g, ea) for g in range(MAX_GOALS + 1)]
    pb = [poisson_probability(g, eb) for g in range(MAX_GOALS + 1)]
    p_win = p_draw = p_loss = total = 0.0
    grid = {}
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            p = pa[x] * pb[y] * dixon_coles_tau(x, y, ea, eb, RHO)
            grid[(x, y)] = p; total += p
    for (x, y), p in grid.items():
        p /= total
        if x > y: p_win += p
        elif x == y: p_draw += p
        else: p_loss += p
    return [p_win, p_draw, p_loss]


# ---------------------------------------------------------------
# Scoring metrics
# ---------------------------------------------------------------
def ranked_probability_score(pred, outcome):
    """Ordered outcomes [win, draw, loss]. Lower = better."""
    cp = [pred[0], pred[0] + pred[1]]
    co = [outcome[0], outcome[0] + outcome[1]]
    return ((cp[0] - co[0]) ** 2 + (cp[1] - co[1]) ** 2) / 2


def outcome_vector(home_goals, away_goals):
    if home_goals > away_goals: return [1, 0, 0]   # home win
    if home_goals == away_goals: return [0, 1, 0]  # draw
    return [0, 0, 1]                               # away win


# ---------------------------------------------------------------
# Run the backtest
# ---------------------------------------------------------------
def main():
    print("Downloading match data...")
    df = pd.read_csv(DATA_URL).dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])

    train = df[(df["date"] >= TRAIN_START) & (df["date"] < CUTOFF)]
    test = df[(df["date"] >= CUTOFF) & (df["date"] <= TEST_END)]

    attack, defense, mu = build_ratings(train)

    # Baseline: how often each outcome happened in training
    base = [
        (train["home_score"] > train["away_score"]).mean(),
        (train["home_score"] == train["away_score"]).mean(),
        (train["home_score"] < train["away_score"]).mean(),
    ]

    n = correct = correct_base = 0
    rps_model = rps_base = logloss = 0.0

    for _, row in test.iterrows():
        a, b = row["home_team"], row["away_team"]
        if a not in attack or b not in attack:
            continue   # skip matches with an unrated team
        pred = predict(a, b, attack, defense, mu)
        outcome = outcome_vector(row["home_score"], row["away_score"])
        truth = outcome.index(1)
        n += 1

        if pred.index(max(pred)) == truth: correct += 1
        if base.index(max(base)) == truth: correct_base += 1
        rps_model += ranked_probability_score(pred, outcome)
        rps_base += ranked_probability_score(base, outcome)
        logloss += -math.log(max(pred[truth], 1e-9))

    print(f"\nTrained on matches {TRAIN_START} to {CUTOFF}")
    print(f"Tested on {n} unseen matches ({CUTOFF} to {TEST_END})")
    print("\n  YOUR MODEL")
    print(f"    Accuracy:  {correct / n * 100:5.1f}%")
    print(f"    RPS:       {rps_model / n:.4f}   (lower is better)")
    print(f"    Log loss:  {logloss / n:.4f}")
    print("\n  BASELINE (always guess the average outcome)")
    print(f"    Accuracy:  {correct_base / n * 100:5.1f}%")
    print(f"    RPS:       {rps_base / n:.4f}")

    improvement = (rps_base - rps_model) / rps_base * 100
    print(f"\n  -> Your model's RPS is {improvement:.0f}% better than the baseline.")


if __name__ == "__main__":
    main()
