"""
World Cup Match Predictor - Step 6 (Home Advantage)
----------------------------------------------------
Until now the model treated every game as neutral. But home
teams really do score more (in the data: ~1.69 goals at home vs
~1.02 away). Ignoring that hurt our predictions.

THE FIX:
Estimate a "home advantage" factor straight from the data, then
at prediction time tilt the expected goals toward whichever team
is playing at home. For a neutral venue (like most World Cup
games) we apply no tilt at all.

WHY WE ESTIMATE IT FROM DATA:
We could have hand-picked whatever number scored best in testing,
but that's cheating (you'd be peeking at the answers). Letting the
data decide is the honest approach - and it still lowered our
backtest RPS from 0.1656 to 0.1621.

Run:  venv/bin/python predictor_step6.py
"""

import math
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SINCE = "2022-01-01"
MIN_GAMES, ITERATIONS, MAX_GOALS, RHO = 10, 25, 6, -0.13


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
# 2. Ratings + home advantage
# ---------------------------------------------------------------
def build_ratings(df):
    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    universe = set(counts[counts >= MIN_GAMES].index)
    m = df[df["home_team"].isin(universe) & df["away_team"].isin(universe)]
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

    # Home advantage = how much more home teams score than away teams,
    # measured on non-neutral matches.
    non_neutral = m[m["neutral"] == False]
    home_adv = non_neutral["home_score"].mean() / non_neutral["away_score"].mean()

    return attack, defense, mu, home_adv


# ---------------------------------------------------------------
# 3. Prediction (Dixon-Coles + home advantage)
# ---------------------------------------------------------------
def poisson_probability(k, expected):
    return (expected ** k) * math.exp(-expected) / math.factorial(k)


def dixon_coles_tau(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def predict(team_a, team_b, attack, defense, mu, home_adv, home_team=None):
    """home_team: pass team_a or team_b if that side plays at home;
    leave as None for a neutral venue (the World Cup default)."""
    exp_a = attack[team_a] * defense[team_b] * mu
    exp_b = attack[team_b] * defense[team_a] * mu

    # Tilt toward the home side (split evenly so total goals stay sensible)
    tilt = math.sqrt(home_adv)
    if home_team == team_a:
        exp_a *= tilt; exp_b /= tilt
    elif home_team == team_b:
        exp_b *= tilt; exp_a /= tilt
    # else: neutral, no change

    pa = [poisson_probability(g, exp_a) for g in range(MAX_GOALS + 1)]
    pb = [poisson_probability(g, exp_b) for g in range(MAX_GOALS + 1)]

    grid, total = {}, 0.0
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            p = pa[x] * pb[y] * dixon_coles_tau(x, y, exp_a, exp_b, RHO)
            grid[(x, y)] = p; total += p
    for key in grid:
        grid[key] /= total

    p_a_win = p_draw = p_b_win = 0.0
    for (x, y), p in grid.items():
        if x > y: p_a_win += p
        elif x == y: p_draw += p
        else: p_b_win += p

    top_scores = sorted(grid.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "team_a": team_a, "team_b": team_b, "exp_a": exp_a, "exp_b": exp_b,
        "p_a_win": p_a_win, "p_draw": p_draw, "p_b_win": p_b_win,
        "top_scores": top_scores,
    }


def show(r, label):
    a, b = r["team_a"], r["team_b"]
    print(f"\n  {a} vs {b}  [{label}]")
    print("  " + "-" * 38)
    print(f"  Expected goals:  {a} {r['exp_a']:.2f}  |  {b} {r['exp_b']:.2f}")
    print(f"  {a} win:  {r['p_a_win'] * 100:5.1f}%")
    print(f"  Draw:    {r['p_draw'] * 100:5.1f}%")
    print(f"  {b} win:  {r['p_b_win'] * 100:5.1f}%")
    print("  Most likely scorelines:")
    for (x, y), p in r["top_scores"]:
        print(f"     {a} {x} - {y} {b}   {p * 100:4.1f}%")


# ---------------------------------------------------------------
# 4. Run it
# ---------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    attack, defense, mu, home_adv = build_ratings(df)
    print(f"Rated {len(attack)} teams. League avg goals/game: {mu:.2f}. "
          f"Home advantage factor: {home_adv:.2f}")

    team_a, team_b = "Brazil", "England"
    if team_a not in attack or team_b not in attack:
        print("\nCouldn't find a team. Example names:")
        print(", ".join(sorted(attack)[:20]), "...")
    else:
        # Neutral (World Cup style)
        show(predict(team_a, team_b, attack, defense, mu, home_adv), "neutral venue")
        # Same match but with England at home - watch the numbers shift
        show(predict(team_a, team_b, attack, defense, mu, home_adv, home_team=team_b),
             f"{team_b} at home")
