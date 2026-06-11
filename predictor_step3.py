"""
World Cup Match Predictor - Step 3 (Dixon-Coles)
-------------------------------------------------
Same real data as Step 2, but with a smarter scoreline model.

THE PROBLEM WITH STEP 2:
The basic model assumed each team's goals are completely
independent. Real football isn't like that - low scores (0-0,
1-1) happen more often than independent Poisson predicts, and
narrow 1-0 / 0-1 wins happen a little less often.

THE FIX (Dixon-Coles, 1997):
Multiply four low-score cells (0-0, 1-0, 0-1, 1-1) by a small
correction factor, then re-balance everything so it still adds
up to 100%. This is the standard model used by football analysts.

Needs pandas (you already installed it). Run:
    venv/bin/python predictor_step3.py
"""

import math
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SINCE = "2022-01-01"
MAX_GOALS = 6

# Dixon-Coles correction strength. A small negative number that
# nudges low scores toward what real football data shows. -0.13 is
# the classic value from the original paper. (Closer to 0 = weaker
# correction; you can tune this later.)
RHO = -0.13


# ---------------------------------------------------------------
# 1. Load data and build team strengths (same as Step 2)
# ---------------------------------------------------------------
def load_data():
    print("Downloading match data... (takes a few seconds)")
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= SINCE]


def build_team_table(df):
    table = {}
    all_teams = set(df["home_team"]) | set(df["away_team"])
    for team in all_teams:
        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]
        games = len(home) + len(away)
        if games < 5:
            continue
        scored = home["home_score"].sum() + away["away_score"].sum()
        conceded = home["away_score"].sum() + away["home_score"].sum()
        table[team] = {"games": games, "scored": scored / games, "conceded": conceded / games}
    return table


def league_average(df):
    total_goals = df["home_score"].sum() + df["away_score"].sum()
    return total_goals / (len(df) * 2)


# ---------------------------------------------------------------
# 2. The prediction math (now with Dixon-Coles)
# ---------------------------------------------------------------
def poisson_probability(k, expected):
    return (expected ** k) * math.exp(-expected) / math.factorial(k)


def expected_goals(attacker, defender, teams, league_avg):
    return teams[attacker]["scored"] * (teams[defender]["conceded"] / league_avg)


def dixon_coles_tau(x, y, lam, mu, rho):
    """The correction factor for low scores. Returns 1 (no change)
    for every scoreline except the four low ones."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def predict(team_a, team_b, teams, league_avg):
    exp_a = expected_goals(team_a, team_b, teams, league_avg)
    exp_b = expected_goals(team_b, team_a, teams, league_avg)

    prob_a = [poisson_probability(g, exp_a) for g in range(MAX_GOALS + 1)]
    prob_b = [poisson_probability(g, exp_b) for g in range(MAX_GOALS + 1)]

    # Build the full scoreline grid, applying the Dixon-Coles correction
    grid = {}
    total = 0.0
    for a_goals in range(MAX_GOALS + 1):
        for b_goals in range(MAX_GOALS + 1):
            p = prob_a[a_goals] * prob_b[b_goals]
            p *= dixon_coles_tau(a_goals, b_goals, exp_a, exp_b, RHO)
            grid[(a_goals, b_goals)] = p
            total += p

    # Re-normalize so all probabilities add back up to 1 (100%)
    for key in grid:
        grid[key] /= total

    # Tally outcomes from the corrected grid
    p_a_win = p_draw = p_b_win = 0.0
    for (a_goals, b_goals), p in grid.items():
        if a_goals > b_goals:
            p_a_win += p
        elif a_goals == b_goals:
            p_draw += p
        else:
            p_b_win += p

    # Most likely scorelines (sorted high to low)
    top_scores = sorted(grid.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "team_a": team_a, "team_b": team_b, "exp_a": exp_a, "exp_b": exp_b,
        "p_a_win": p_a_win, "p_draw": p_draw, "p_b_win": p_b_win,
        "top_scores": top_scores,
    }


def show(r):
    a, b = r["team_a"], r["team_b"]
    print(f"\n  {a}  vs  {b}")
    print("  " + "-" * 34)
    print(f"  Expected goals:  {a} {r['exp_a']:.2f}  |  {b} {r['exp_b']:.2f}")
    print(f"  {a} win:  {r['p_a_win'] * 100:5.1f}%")
    print(f"  Draw:    {r['p_draw'] * 100:5.1f}%")
    print(f"  {b} win:  {r['p_b_win'] * 100:5.1f}%")
    print("  Most likely scorelines:")
    for (a_goals, b_goals), p in r["top_scores"]:
        print(f"     {a} {a_goals} - {b_goals} {b}   {p * 100:4.1f}%")


# ---------------------------------------------------------------
# 3. Run it
# ---------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    teams = build_team_table(df)
    league_avg = league_average(df)
    print(f"Loaded {len(teams)} teams. League average goals/game: {league_avg:.2f}")

    team_a, team_b = "Brazil", "England"
    if team_a not in teams or team_b not in teams:
        print("\nCouldn't find one of those teams. Example valid names:")
        print(", ".join(sorted(list(teams))[:20]), "...")
    else:
        show(predict(team_a, team_b, teams, league_avg))
