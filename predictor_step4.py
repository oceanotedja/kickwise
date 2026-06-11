"""
World Cup Match Predictor - Step 4 (Strength of Opposition)
------------------------------------------------------------
Steps 2-3 used each team's RAW goal averages. Problem: a team
that scores a lot against weak opponents looked just as strong
as one that scores against elite teams.

THE FIX:
Give every team an ATTACK rating and a DEFENSE rating that are
adjusted for who they actually played. We find these with a
simple repeated calculation:

  - A team's attack = goals they scored, divided by how strong
    their opponents' defenses were.
  - A team's defense = goals they conceded, divided by how
    strong their opponents' attacks were.

We start everyone at 1.0 (average) and repeat the calculation
~25 times until the numbers settle. A rating above 1.0 = better
than average attack; a defense rating below 1.0 = harder to
score against.

Then expected goals for a match = attack(A) x defense(B) x league_avg,
fed into the same Dixon-Coles model from Step 3.

Run:  venv/bin/python predictor_step4.py
"""

import math
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SINCE = "2022-01-01"
MIN_GAMES = 10        # only rate teams with at least this many matches
ITERATIONS = 25       # how many times to repeat the ratings calculation
MAX_GOALS = 6
RHO = -0.13           # Dixon-Coles low-score correction (from Step 3)


# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
def load_data():
    print("Downloading match data... (takes a few seconds)")
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= SINCE]


# ---------------------------------------------------------------
# 2. Build opponent-adjusted attack & defense ratings
# ---------------------------------------------------------------
def build_ratings(df):
    # Universe = teams with enough games. We only use matches where
    # BOTH teams are rated, so every opponent has a rating.
    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    universe = set(counts[counts >= MIN_GAMES].index)
    m = df[df["home_team"].isin(universe) & df["away_team"].isin(universe)]

    # League baseline: average goals scored by a team in one game.
    mu = (m["home_score"].sum() + m["away_score"].sum()) / (len(m) * 2)

    # For each team, gather total goals scored/conceded and the list
    # of opponents they faced (one entry per match).
    stats = {}
    for team in universe:
        home = m[m["home_team"] == team]
        away = m[m["away_team"] == team]
        stats[team] = {
            "scored": home["home_score"].sum() + away["away_score"].sum(),
            "conceded": home["away_score"].sum() + away["home_score"].sum(),
            "opponents": list(home["away_team"]) + list(away["home_team"]),
        }

    # Start everyone at average (1.0), then refine repeatedly.
    attack = {t: 1.0 for t in universe}
    defense = {t: 1.0 for t in universe}
    for _ in range(ITERATIONS):
        new_attack, new_defense = {}, {}
        for t in universe:
            opp_defense_sum = sum(defense[o] for o in stats[t]["opponents"])
            opp_attack_sum = sum(attack[o] for o in stats[t]["opponents"])
            new_attack[t] = stats[t]["scored"] / (mu * opp_defense_sum) if opp_defense_sum else 1.0
            new_defense[t] = stats[t]["conceded"] / (mu * opp_attack_sum) if opp_attack_sum else 1.0
        # Re-center so the average rating stays at 1.0
        a_mean = sum(new_attack.values()) / len(new_attack)
        d_mean = sum(new_defense.values()) / len(new_defense)
        attack = {t: v / a_mean for t, v in new_attack.items()}
        defense = {t: v / d_mean for t, v in new_defense.items()}

    return attack, defense, mu


# ---------------------------------------------------------------
# 3. Prediction math (Dixon-Coles, from Step 3)
# ---------------------------------------------------------------
def poisson_probability(k, expected):
    return (expected ** k) * math.exp(-expected) / math.factorial(k)


def dixon_coles_tau(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def predict(team_a, team_b, attack, defense, league_avg):
    # Expected goals now account for both teams' strength
    exp_a = attack[team_a] * defense[team_b] * league_avg
    exp_b = attack[team_b] * defense[team_a] * league_avg

    prob_a = [poisson_probability(g, exp_a) for g in range(MAX_GOALS + 1)]
    prob_b = [poisson_probability(g, exp_b) for g in range(MAX_GOALS + 1)]

    grid, total = {}, 0.0
    for a_goals in range(MAX_GOALS + 1):
        for b_goals in range(MAX_GOALS + 1):
            p = prob_a[a_goals] * prob_b[b_goals]
            p *= dixon_coles_tau(a_goals, b_goals, exp_a, exp_b, RHO)
            grid[(a_goals, b_goals)] = p
            total += p
    for key in grid:
        grid[key] /= total

    p_a_win = p_draw = p_b_win = 0.0
    for (a_goals, b_goals), p in grid.items():
        if a_goals > b_goals:
            p_a_win += p
        elif a_goals == b_goals:
            p_draw += p
        else:
            p_b_win += p

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
# 4. Run it
# ---------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    attack, defense, league_avg = build_ratings(df)
    print(f"Rated {len(attack)} teams. League average goals/game: {league_avg:.2f}")

    team_a, team_b = "Brazil", "England"
    if team_a not in attack or team_b not in attack:
        print("\nCouldn't find one of those teams. Example valid names:")
        print(", ".join(sorted(attack)[:20]), "...")
    else:
        show(predict(team_a, team_b, attack, defense, league_avg))
