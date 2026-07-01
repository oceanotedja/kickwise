"""
World Cup Match Predictor - Step 1 (Minimum Viable Version)
------------------------------------------------------------
Predicts Win / Draw / Loss probabilities and the most likely
scoreline for a match between two teams, using a Poisson model.

This version uses ONLY Python's standard library, so you don't
need to install anything. Just run it:   python predictor.py

The team numbers below are illustrative starter values (average
goals scored and conceded per match). In a later step we'll
replace them with real data from the internet.
"""

import math

# ---------------------------------------------------------------
# 1. Our tiny "database": each team's average goals scored and
#    conceded per match. (Made-up starter numbers for now.)
# ---------------------------------------------------------------
teams = {
    "Brazil":    {"scored": 2.1, "conceded": 0.8},
    "France":    {"scored": 2.0, "conceded": 0.9},
    "Argentina": {"scored": 1.9, "conceded": 0.9},
    "England":   {"scored": 1.8, "conceded": 1.0},
    "Germany":   {"scored": 1.7, "conceded": 1.1},
    "USA":       {"scored": 1.5, "conceded": 1.2},
    "Japan":     {"scored": 1.4, "conceded": 1.2},
    "Morocco":   {"scored": 1.3, "conceded": 1.0},
}

# League-wide average goals per team per match. Used to scale how
# strong a team's attack is versus how leaky an opponent's defense is.
LEAGUE_AVG_GOALS = 1.7

# How many goals we bother calculating per team (0 to 6 is plenty).
MAX_GOALS = 6


def poisson_probability(k, expected):
    """Chance of scoring exactly k goals when the expected (average)
    number of goals is `expected`. This is the Poisson formula:
    (lambda^k * e^-lambda) / k!"""
    return (expected ** k) * math.exp(-expected) / math.factorial(k)


def expected_goals(attacking_team, defending_team):
    """How many goals we expect `attacking_team` to score against
    `defending_team`: their attack, adjusted by the opponent's
    defense, scaled by the league average."""
    attack = teams[attacking_team]["scored"]
    defense = teams[defending_team]["conceded"]
    return attack * (defense / LEAGUE_AVG_GOALS)


def predict(team_a, team_b):
    # Expected goals for each side
    exp_a = expected_goals(team_a, team_b)
    exp_b = expected_goals(team_b, team_a)

    # Probability each team scores 0, 1, 2, ... goals
    prob_a = [poisson_probability(g, exp_a) for g in range(MAX_GOALS + 1)]
    prob_b = [poisson_probability(g, exp_b) for g in range(MAX_GOALS + 1)]

    # Go through every possible scoreline and tally the outcomes
    p_a_win = p_draw = p_b_win = 0.0
    best_score = (0, 0)
    best_score_prob = 0.0

    for a_goals in range(MAX_GOALS + 1):
        for b_goals in range(MAX_GOALS + 1):
            p = prob_a[a_goals] * prob_b[b_goals]   # chance of this exact score
            if p > best_score_prob:
                best_score_prob = p
                best_score = (a_goals, b_goals)
            if a_goals > b_goals:
                p_a_win += p
            elif a_goals == b_goals:
                p_draw += p
            else:
                p_b_win += p

    return {
        "team_a": team_a, "team_b": team_b,
        "exp_a": exp_a, "exp_b": exp_b,
        "p_a_win": p_a_win, "p_draw": p_draw, "p_b_win": p_b_win,
        "best_score": best_score, "best_score_prob": best_score_prob,
    }


def show(result):
    a, b = result["team_a"], result["team_b"]
    print(f"\n  {a}  vs  {b}")
    print("  " + "-" * 30)
    print(f"  Expected goals:  {a} {result['exp_a']:.2f}  |  {b} {result['exp_b']:.2f}")
    print(f"  {a} win:  {result['p_a_win'] * 100:5.1f}%")
    print(f"  Draw:    {result['p_draw'] * 100:5.1f}%")
    print(f"  {b} win:  {result['p_b_win'] * 100:5.1f}%")
    s = result["best_score"]
    print(f"  Most likely score:  {a} {s[0]} - {s[1]} {b}  ({result['best_score_prob'] * 100:.1f}%)")


if __name__ == "__main__":
    # Change these two names to any teams listed in the `teams` dict above
    match = predict("Brazil", "England")
    show(match)


