"""
World Cup Match Predictor - Step 2 (Real Data)
-----------------------------------------------
Same Poisson model as Step 1, but the team numbers are now
calculated from REAL international match results instead of
being made up.

Data source: a free, public dataset of every international
football match (github.com/martj42/international_results).
No sign-up or API key needed.

You DO need the pandas library now. Install it once by typing
this in your terminal:
    pip install pandas
(on Mac, you may need:  pip3 install pandas)

Then run:
    python predictor_step2.py
"""

import math
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Only use matches from this date onward, so the numbers reflect
# each team's CURRENT strength rather than ancient history.
SINCE = "2022-01-01"

MAX_GOALS = 6   # how many goals per team we calculate (0 to 6)


# ---------------------------------------------------------------
# 1. Load the data and work out each team's attack/defense numbers
# ---------------------------------------------------------------
def load_data():
    print("Downloading match data... (takes a few seconds)")
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=["home_score", "away_score"])   # drop unplayed matches
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= SINCE]
    return df


def build_team_table(df):
    """For every team, compute average goals scored and conceded per game."""
    table = {}
    all_teams = set(df["home_team"]) | set(df["away_team"])
    for team in all_teams:
        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]
        games = len(home) + len(away)
        if games < 5:        # skip teams with too little data to be reliable
            continue
        scored = home["home_score"].sum() + away["away_score"].sum()
        conceded = home["away_score"].sum() + away["home_score"].sum()
        table[team] = {
            "games": games,
            "scored": scored / games,
            "conceded": conceded / games,
        }
    return table


def league_average(df):
    """Average goals scored by a team in a single game, across all teams."""
    total_goals = df["home_score"].sum() + df["away_score"].sum()
    total_team_games = len(df) * 2
    return total_goals / total_team_games


# ---------------------------------------------------------------
# 2. The prediction math (unchanged from Step 1)
# ---------------------------------------------------------------
def poisson_probability(k, expected):
    return (expected ** k) * math.exp(-expected) / math.factorial(k)


def expected_goals(attacker, defender, teams, league_avg):
    attack = teams[attacker]["scored"]
    defense = teams[defender]["conceded"]
    return attack * (defense / league_avg)


def predict(team_a, team_b, teams, league_avg):
    exp_a = expected_goals(team_a, team_b, teams, league_avg)
    exp_b = expected_goals(team_b, team_a, teams, league_avg)

    prob_a = [poisson_probability(g, exp_a) for g in range(MAX_GOALS + 1)]
    prob_b = [poisson_probability(g, exp_b) for g in range(MAX_GOALS + 1)]

    p_a_win = p_draw = p_b_win = 0.0
    best_score, best_score_prob = (0, 0), 0.0
    for a_goals in range(MAX_GOALS + 1):
        for b_goals in range(MAX_GOALS + 1):
            p = prob_a[a_goals] * prob_b[b_goals]
            if p > best_score_prob:
                best_score_prob, best_score = p, (a_goals, b_goals)
            if a_goals > b_goals:
                p_a_win += p
            elif a_goals == b_goals:
                p_draw += p
            else:
                p_b_win += p

    return {
        "team_a": team_a, "team_b": team_b, "exp_a": exp_a, "exp_b": exp_b,
        "p_a_win": p_a_win, "p_draw": p_draw, "p_b_win": p_b_win,
        "best_score": best_score, "best_score_prob": best_score_prob,
    }


def show(r):
    a, b = r["team_a"], r["team_b"]
    print(f"\n  {a}  vs  {b}")
    print("  " + "-" * 34)
    print(f"  Expected goals:  {a} {r['exp_a']:.2f}  |  {b} {r['exp_b']:.2f}")
    print(f"  {a} win:  {r['p_a_win'] * 100:5.1f}%")
    print(f"  Draw:    {r['p_draw'] * 100:5.1f}%")
    print(f"  {b} win:  {r['p_b_win'] * 100:5.1f}%")
    s = r["best_score"]
    print(f"  Most likely score:  {a} {s[0]} - {s[1]} {b}  ({r['best_score_prob'] * 100:.1f}%)")


# ---------------------------------------------------------------
# 3. Run it
# ---------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    teams = build_team_table(df)
    league_avg = league_average(df)
    print(f"Loaded {len(teams)} teams. League average goals/game: {league_avg:.2f}")

    # Change these to any two teams. If a name isn't found, we'll tell you.
    team_a, team_b = "Brazil", "England"
    if team_a not in teams or team_b not in teams:
        print(f"\nCouldn't find one of those teams. Example valid names:")
        print(", ".join(sorted(list(teams))[:20]), "...")
    else:
        show(predict(team_a, team_b, teams, league_avg))
