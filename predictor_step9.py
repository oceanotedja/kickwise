"""
World Cup Match Predictor - Step 9 (Tournament Simulation)
-----------------------------------------------------------
The showpiece: simulate the ENTIRE 2026 World Cup thousands of
times and estimate every team's chance of winning the trophy.

HOW IT WORKS:
  1. Build team ratings with the full Step 7 engine
     (opponent-adjusted, time-weighted, Dixon-Coles).
  2. Read the REAL 2026 group-stage fixtures from the dataset
     and infer the 12 groups automatically.
  3. Play one whole tournament: for every match, randomly draw
     a scoreline from the model's probability grid. Compute group
     tables, advance top 2 + the 8 best third-place teams, then
     play the knockout rounds to a champion.
  4. Repeat 10,000 times. Count how often each team wins it all,
     reaches the final, the semis, etc.

HOST ADVANTAGE: USA, Mexico and Canada get the home tilt from
Step 6 in their matches - they really are playing at home.

SIMPLIFICATIONS (honesty corner):
  - The real round-of-32 bracket has fixed slots; we approximate
    it by seeding qualifiers on group-stage performance (1 vs 32,
    2 vs 31, ...). Close enough for good estimates.
  - Drawn knockout games are settled by each side's relative win
    strength (a stand-in for extra time + penalties).

Run:  venv/bin/python predictor_step9.py     (takes ~1-2 minutes)
"""

import math
import random
import bisect
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SINCE = "2019-01-01"
HALF_LIFE_DAYS = 365
MIN_GAMES, ITERATIONS, MAX_GOALS, RHO = 10, 25, 6, -0.13
N_SIMS = 10000
HOSTS = {"United States", "Mexico", "Canada"}


# ---------------------------------------------------------------
# 1. Ratings engine (Step 7)
# ---------------------------------------------------------------
def load_all():
    print("Downloading match data...")
    df = pd.read_csv(DATA_URL)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_ratings(df):
    hist = df.dropna(subset=["home_score", "away_score"])
    hist = hist[hist["date"] >= SINCE]
    counts = pd.concat([hist["home_team"], hist["away_team"]]).value_counts()
    universe = set(counts[counts >= MIN_GAMES].index)
    m = hist[hist["home_team"].isin(universe) & hist["away_team"].isin(universe)].copy()

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


# ---------------------------------------------------------------
# 2. Match grid: probability of every scoreline (cached per pair)
# ---------------------------------------------------------------
def poisson_probability(k, expected):
    return (expected ** k) * math.exp(-expected) / math.factorial(k)


def dixon_coles_tau(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


GRID_CACHE = {}

def match_grid(a, b, attack, defense, mu, home_adv):
    """Returns (scorelines, cumulative probs, P(a wins | not draw))
    for fast random sampling. Host nations get the home tilt."""
    key = (a, b)
    if key in GRID_CACHE:
        return GRID_CACHE[key]

    ea = attack[a] * defense[b] * mu
    eb = attack[b] * defense[a] * mu
    tilt = math.sqrt(home_adv)
    if a in HOSTS and b not in HOSTS:
        ea *= tilt; eb /= tilt
    elif b in HOSTS and a not in HOSTS:
        eb *= tilt; ea /= tilt

    pa = [poisson_probability(g, ea) for g in range(MAX_GOALS + 1)]
    pb = [poisson_probability(g, eb) for g in range(MAX_GOALS + 1)]

    scores, probs = [], []
    p_win = p_loss = 0.0
    total = 0.0
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            p = pa[x] * pb[y] * dixon_coles_tau(x, y, ea, eb, RHO)
            scores.append((x, y)); probs.append(p); total += p
            if x > y: p_win += p
            elif x < y: p_loss += p

    cumulative, run = [], 0.0
    for p in probs:
        run += p / total
        cumulative.append(run)
    shootout_a = p_win / (p_win + p_loss)   # a's edge if the game must produce a winner

    GRID_CACHE[key] = (scores, cumulative, shootout_a)
    return GRID_CACHE[key]


def sample_score(a, b, attack, defense, mu, home_adv):
    scores, cumulative, _ = match_grid(a, b, attack, defense, mu, home_adv)
    return scores[bisect.bisect_left(cumulative, random.random())]


# ---------------------------------------------------------------
# 3. Tournament structure from the real fixtures
# ---------------------------------------------------------------
def get_world_cup(df):
    gs = df[(df["tournament"] == "FIFA World Cup") &
            (df["date"] >= "2026-06-01") & (df["date"] <= "2026-06-27")]
    fixtures = list(zip(gs["home_team"], gs["away_team"]))

    # Infer the 12 groups: teams that play each other are in the same group.
    adj = {}
    for h, a in fixtures:
        adj.setdefault(h, set()).add(a)
        adj.setdefault(a, set()).add(h)
    seen, groups = set(), []
    for t in adj:
        if t in seen:
            continue
        comp, stack = {t}, [t]
        while stack:
            for v in adj[stack.pop()]:
                if v not in comp:
                    comp.add(v); stack.append(v)
        seen |= comp; groups.append(sorted(comp))
    return fixtures, groups


# ---------------------------------------------------------------
# 4. Simulate one full tournament
# ---------------------------------------------------------------
def simulate_once(fixtures, groups, attack, defense, mu, home_adv):
    # --- group stage: play the real fixtures ---
    pts = {t: 0 for g in groups for t in g}
    gd = {t: 0 for t in pts}
    gf = {t: 0 for t in pts}
    for h, a in fixtures:
        hs, as_ = sample_score(h, a, attack, defense, mu, home_adv)
        gd[h] += hs - as_; gd[a] += as_ - hs
        gf[h] += hs; gf[a] += as_
        if hs > as_: pts[h] += 3
        elif hs < as_: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1

    def table_key(t):
        return (pts[t], gd[t], gf[t], random.random())   # random settles full ties

    qualified, thirds = [], []
    for g in groups:
        order = sorted(g, key=table_key, reverse=True)
        qualified += order[:2]
        thirds.append(order[2])
    thirds = sorted(thirds, key=table_key, reverse=True)
    qualified += thirds[:8]                      # 24 + 8 = 32 teams

    # --- knockouts: seed by group-stage record, 1v32, 2v31, ... ---
    field = sorted(qualified, key=table_key, reverse=True)
    results = {t: "groups" if t not in field else "R32" for t in pts}

    round_names = ["R16", "QF", "SF", "Final", "Champion"]
    rnd = 0
    while len(field) > 1:
        nxt = []
        n = len(field)
        for i in range(n // 2):
            a, b = field[i], field[n - 1 - i]
            hs, as_ = sample_score(a, b, attack, defense, mu, home_adv)
            if hs == as_:   # extra time + penalties proxy
                _, _, edge_a = match_grid(a, b, attack, defense, mu, home_adv)
                winner = a if random.random() < edge_a else b
            else:
                winner = a if hs > as_ else b
            nxt.append(winner)
        for t in nxt:
            results[t] = round_names[min(rnd, len(round_names) - 1)]
        field = nxt
        rnd += 1
    results[field[0]] = "Champion"
    return results


# ---------------------------------------------------------------
# 5. Run many tournaments and tally
# ---------------------------------------------------------------
def main():
    df = load_all()
    attack, defense, mu, home_adv = build_ratings(df)
    fixtures, groups = get_world_cup(df)
    print(f"Engine ready: {len(attack)} teams rated, host tilt x{home_adv:.2f}")
    print(f"Real fixtures: {len(fixtures)} group games, {len(groups)} groups")
    print(f"Simulating the 2026 World Cup {N_SIMS:,} times...\n")

    order = ["Champion", "Final", "SF", "QF", "R16", "R32", "groups"]
    reach = {t: {stage: 0 for stage in order} for g in groups for t in g}

    for i in range(N_SIMS):
        results = simulate_once(fixtures, groups, attack, defense, mu, home_adv)
        for t, stage in results.items():
            idx = order.index(stage)
            for s in order[idx:]:          # reaching SF also means reaching QF etc.
                reach[t][s] += 1
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1:,} tournaments simulated")

    print(f"\n{'TEAM':<16}{'WIN IT':>8}{'FINAL':>8}{'SEMI':>8}{'QF':>8}{'KO':>8}")
    print("-" * 56)
    ranked = sorted(reach, key=lambda t: reach[t]["Champion"], reverse=True)
    for t in ranked[:20]:
        r = reach[t]
        print(f"{t:<16}"
              f"{r['Champion'] / N_SIMS * 100:>7.1f}%"
              f"{r['Final'] / N_SIMS * 100:>7.1f}%"
              f"{r['SF'] / N_SIMS * 100:>7.1f}%"
              f"{r['QF'] / N_SIMS * 100:>7.1f}%"
              f"{r['R32'] / N_SIMS * 100:>7.1f}%")
    print("\n(KO = reaching the knockout round of 32)")


if __name__ == "__main__":
    main()
