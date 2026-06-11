"""
World Cup Match Predictor - Step 10 (Player Condition)
-------------------------------------------------------
The feature you wanted from day one: injuries, suspensions,
fatigue. This adds a CONDITION LAYER on top of the Step 7 engine.

HOW IT WORKS:
Team ratings are built from results, so they assume a roughly
full-strength squad. When players are missing, we shrink the
team's attack and/or inflate its defense (concede more) before
the scoreline math runs.

Each absent player gets an importance tier:
    "star"  - the talisman; ~18% of the team's threat
    "key"   - a clear starter; ~10%
    "squad" - rotation player; ~4%
and a role: "attack", "defense", or "both" (e.g. a midfielder).
A rest-day deficit also costs ~2% of expected goals per day
(capped at 3 days).

These weights are reasoned heuristics, not fitted constants -
historical injury data isn't freely available to backtest them.
That's normal: even professional models hand-tune this part.

TWO WAYS TO FEED IT:
  1. MANUAL (works right now): edit SQUAD_NEWS below using real
     World Cup team news from the press.
  2. AUTOMATIC (optional): get a free API key at api-football.com
     (100 requests/day free), paste it into API_KEY below, and
     fetch_injuries() pulls the official injury list for the
     entire World Cup in a single request.

Run:  venv/bin/python predictor_step10.py
"""

import math
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SINCE = "2019-01-01"
HALF_LIFE_DAYS = 365
MIN_GAMES, ITERATIONS, MAX_GOALS, RHO = 10, 25, 6, -0.13

# ----------------------------------------------------------------
# YOUR SQUAD NEWS - edit this with real team news.
# Format: team -> list of (player_name, tier, role)
# tier: "star" | "key" | "squad"     role: "attack" | "defense" | "both"
# ----------------------------------------------------------------
SQUAD_NEWS = {
    # Example entries - replace with the latest real news:
    "England": [
        ("Star striker (example)", "star", "attack"),
    ],
    "France": [
        ("First-choice CB (example)", "key", "defense"),
    ],
}

# Rest-day deficit per team (negative = fewer rest days than opponent).
# Mostly matters in knockout rounds; leave empty if unknown.
REST_DAYS = {
    # "France": -2,
}

# Optional: paste your free key from https://www.api-football.com
API_KEY = "8cff0f9812c1efb9e40bc575089ff090"

IMPORTANCE = {"star": 0.18, "key": 0.10, "squad": 0.04}


# ----------------------------------------------------------------
# 1. Engine (Step 7, unchanged)
# ----------------------------------------------------------------
def load_data():
    print("Downloading match data... (takes a few seconds)")
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= SINCE]


def build_ratings(df):
    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    universe = set(counts[counts >= MIN_GAMES].index)
    m = df[df["home_team"].isin(universe) & df["away_team"].isin(universe)].copy()

    ref_date = m["date"].max()
    m["age"] = (ref_date - m["date"]).dt.days
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

    nn = m[m["neutral"] == False]
    home_adv = nn["home_score"].mean() / nn["away_score"].mean()
    return attack, defense, mu, home_adv


# ----------------------------------------------------------------
# 2. THE NEW PART: condition multipliers
# ----------------------------------------------------------------
def condition_multipliers(team):
    """Turn a team's squad news + rest into (attack_mult, defense_mult).
    attack_mult < 1 means we expect them to score less;
    defense_mult > 1 means we expect them to concede more."""
    att_mult, def_mult = 1.0, 1.0
    for player, tier, role in SQUAD_NEWS.get(team, []):
        hit = IMPORTANCE[tier]
        share = 0.5 if role == "both" else 1.0   # a "both" player splits the hit
        if role in ("attack", "both"):
            att_mult *= (1 - hit * share)
        if role in ("defense", "both"):
            def_mult *= (1 + hit * share)
    rest = max(min(REST_DAYS.get(team, 0), 3), -3)
    att_mult *= (1 + 0.02 * rest)
    return att_mult, def_mult


# ----------------------------------------------------------------
# 3. Prediction with condition applied
# ----------------------------------------------------------------
def poisson_probability(k, expected):
    return (expected ** k) * math.exp(-expected) / math.factorial(k)


def dixon_coles_tau(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def predict(team_a, team_b, attack, defense, mu, home_adv,
            home_team=None, use_condition=True):
    exp_a = attack[team_a] * defense[team_b] * mu
    exp_b = attack[team_b] * defense[team_a] * mu

    if use_condition:
        a_att, a_def = condition_multipliers(team_a)
        b_att, b_def = condition_multipliers(team_b)
        # My attack is shrunk by my absences AND boosted by the
        # opponent's weakened defense (their def_mult > 1).
        exp_a *= a_att * b_def
        exp_b *= b_att * a_def

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
    top = sorted(grid.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return exp_a, exp_b, p_a_win, p_draw, p_b_win, top


# ----------------------------------------------------------------
# 4. Optional: auto-fetch real injuries from API-Football
# ----------------------------------------------------------------
def fetch_injuries():
    """Pulls the official World Cup injury list (one request) and
    fills SQUAD_NEWS automatically. Every fetched absence is marked
    tier='key', role='both' - upgrade your stars manually after."""
    if not API_KEY:
        print("(No API key set - using manual SQUAD_NEWS instead.)")
        return
    import requests
    r = requests.get(
        "https://v3.football.api-sports.io/injuries",
        headers={"x-apisports-key": API_KEY},
        params={"league": 1, "season": 2026},   # league 1 = FIFA World Cup
        timeout=30,
    )
    r.raise_for_status()
    for item in r.json().get("response", []):
        team = item["team"]["name"]
        player = item["player"]["name"]
        reason = item["player"].get("reason", "unavailable")
        SQUAD_NEWS.setdefault(team, [])
        if not any(p == player for p, _, _ in SQUAD_NEWS[team]):
            SQUAD_NEWS[team].append((f"{player} ({reason})", "key", "both"))
    print(f"Fetched injuries for {len(SQUAD_NEWS)} teams from API-Football.")


# ----------------------------------------------------------------
# 5. Demo: same match with and without the squad news
# ----------------------------------------------------------------
def show(label, r, a, b):
    exp_a, exp_b, pa, pdr, pb, top = r
    print(f"\n  [{label}]")
    print(f"  Expected goals:  {a} {exp_a:.2f}  |  {b} {exp_b:.2f}")
    print(f"  {a} win {pa*100:5.1f}%   Draw {pdr*100:5.1f}%   {b} win {pb*100:5.1f}%")
    print(f"  Top scores: " + ",  ".join(f"{x}-{y} ({p*100:.1f}%)" for (x, y), p in top))


if __name__ == "__main__":
    df = load_data()
    attack, defense, mu, home_adv = build_ratings(df)
    fetch_injuries()

    team_a, team_b = "England", "France"
    print(f"\n  {team_a} vs {team_b} (neutral venue)")
    print("  " + "=" * 50)
    show("Full strength", predict(team_a, team_b, attack, defense, mu, home_adv,
                                  use_condition=False), team_a, team_b)
    show("With current squad news", predict(team_a, team_b, attack, defense, mu,
                                            home_adv, use_condition=True), team_a, team_b)

    print("\n  Squad news applied:")
    for t in (team_a, team_b):
        for player, tier, role in SQUAD_NEWS.get(t, []):
            print(f"    {t}: {player}  [{tier}, {role}]")
        if REST_DAYS.get(t):
            print(f"    {t}: rest-day difference {REST_DAYS[t]:+d}")
