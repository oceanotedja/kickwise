"""
World Cup Match Predictor - Step 7 (Time-Weighting / Recent Form)
------------------------------------------------------------------
Until now every match counted equally - a result from 2019 had
the same say as one from last week. But a team's recent form
tells you more about how good they are RIGHT NOW.

THE FIX (exponential time decay, the classic Dixon-Coles trick):
Give each match a weight that fades the older it is. We use a
"half-life" of 18 months: a match 18 months old counts half as
much as a brand-new one, a 3-year-old match a quarter as much,
and so on. Recent results dominate; ancient ones barely register.

WHY 18 MONTHS:
Testing showed the sweet spot is roughly 1-1.5 years. Forget too
slowly and stale results drag you down; forget too fast and you
throw away useful data. 18 months is a sensible middle ground
(about one international cycle), not a value cherry-picked to win.

Because decay handles old data automatically, we can now safely
load MORE history (back to 2019) - the weights sort it out.

Run:  venv/bin/python predictor_step7.py
"""

import math
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SINCE = "2019-01-01"
HALF_LIFE_DAYS = 548          # 18 months: a match this old counts half as much
MIN_GAMES, ITERATIONS, MAX_GOALS, RHO = 8, 30, 6, -0.13


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
# 2. Time-weighted, opponent-adjusted ratings
# ---------------------------------------------------------------
def build_ratings(df):
    # Recency weight for each match: newest = 1.0, fading with age.
    reference = df["date"].max()
    decay = math.log(2) / HALF_LIFE_DAYS
    df = df.copy()
    df["w"] = ((reference - df["date"]).dt.days * -decay).map(math.exp)

    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    universe = set(counts[counts >= MIN_GAMES].index)
    m = df[df["home_team"].isin(universe) & df["away_team"].isin(universe)]

    # Pull the columns we need into a fast plain-Python list of matches.
    matches = list(zip(m["home_team"], m["away_team"],
                       m["home_score"], m["away_score"], m["w"]))

    # Weighted league average goals per team per game.
    w_goals = sum((hg + ag) * w for _, _, hg, ag, w in matches)
    w_games = sum(2 * w for *_, w in matches)
    mu = w_goals / w_games

    # Weighted goals scored / conceded per team (these don't change between iterations).
    w_scored = {t: 0.0 for t in universe}
    w_conceded = {t: 0.0 for t in universe}
    for h, a, hg, ag, w in matches:
        w_scored[h] += hg * w; w_conceded[h] += ag * w
        w_scored[a] += ag * w; w_conceded[a] += hg * w

    # Iterate to convergence (same idea as Step 4, now weighted).
    attack = {t: 1.0 for t in universe}; defense = {t: 1.0 for t in universe}
    for _ in range(ITERATIONS):
        den_att = {t: 0.0 for t in universe}
        den_def = {t: 0.0 for t in universe}
        for h, a, hg, ag, w in matches:
            den_att[h] += defense[a] * w; den_def[h] += attack[a] * w
            den_att[a] += defense[h] * w; den_def[a] += attack[h] * w
        na = {t: w_scored[t] / (mu * den_att[t]) if den_att[t] else 1.0 for t in universe}
        nd = {t: w_conceded[t] / (mu * den_def[t]) if den_def[t] else 1.0 for t in universe}
        am = sum(na.values()) / len(na); dm = sum(nd.values()) / len(nd)
        attack = {t: v / am for t, v in na.items()}
        defense = {t: v / dm for t, v in nd.items()}

    # Home advantage (weighted) from non-neutral matches.
    nn = m[m["neutral"] == False]
    home_goals = (nn["home_score"] * nn["w"]).sum() / nn["w"].sum()
    away_goals = (nn["away_score"] * nn["w"]).sum() / nn["w"].sum()
    home_adv = home_goals / away_goals

    return attack, defense, mu, home_adv


# ---------------------------------------------------------------
# 3. Prediction (Dixon-Coles + home advantage, from Step 6)
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
    exp_a = attack[team_a] * defense[team_b] * mu
    exp_b = attack[team_b] * defense[team_a] * mu
    tilt = math.sqrt(home_adv)
    if home_team == team_a:
        exp_a *= tilt; exp_b /= tilt
    elif home_team == team_b:
        exp_b *= tilt; exp_a /= tilt

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
          f"Home advantage: {home_adv:.2f}. Form half-life: {HALF_LIFE_DAYS} days.")

    team_a, team_b = "Brazil", "England"
    if team_a not in attack or team_b not in attack:
        print("\nCouldn't find a team. Example names:")
        print(", ".join(sorted(attack)[:20]), "...")
    else:
        show(predict(team_a, team_b, attack, defense, mu, home_adv), "neutral venue")
        show(predict(team_a, team_b, attack, defense, mu, home_adv, home_team=team_b),
             f"{team_b} at home")
