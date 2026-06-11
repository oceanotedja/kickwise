"""
World Cup Match Predictor - Tournament Simulator v2 (Step 12)
-------------------------------------------------------------
The 2026 World Cup Monte Carlo simulator, now powered by the
Step 12 engine (Elo + tournament-weighted Dixon-Coles blend).

Same simulation structure as Step 9, better match predictions.

Run:  venv/bin/python tournament_sim.py   (takes ~2 minutes)
"""

import math
import random
import bisect
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SINCE = "2019-01-01"
HALF_LIFE_DAYS = 365
MIN_GAMES, ITERATIONS, MAX_GOALS, RHO = 10, 25, 6, -0.13
ELO_WEIGHT = 0.4
ELO_START  = 1500
ELO_K      = 32
DRAW_BASE  = 0.22
N_SIMS     = 10000
HOSTS      = {"United States", "Mexico", "Canada"}

TOURN_WEIGHTS = {
    "FIFA World Cup": 1.5, "UEFA Euro": 1.4, "Copa América": 1.4,
    "Africa Cup of Nations": 1.3, "AFC Asian Cup": 1.3,
    "CONCACAF Gold Cup": 1.2, "FIFA World Cup qualification": 1.1,
    "UEFA Nations League": 1.1, "Friendly": 0.4,
}

def tournament_weight(name):
    for key, val in TOURN_WEIGHTS.items():
        if key.lower() in name.lower(): return val
    return 1.0


# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
def load_all():
    print("Downloading match data...")
    df = pd.read_csv(DATA_URL)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


# ---------------------------------------------------------------
# 2. Build Elo
# ---------------------------------------------------------------
def build_elo(df):
    played = df.dropna(subset=["home_score", "away_score"])
    elo = {}
    for row in played.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = elo.get(h, ELO_START); ra = elo.get(a, ELO_START)
        exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
        sh = 1 if row.home_score > row.away_score else (0.5 if row.home_score == row.away_score else 0)
        K = ELO_K * tournament_weight(row.tournament)
        elo[h] = rh + K * (sh - exp_h)
        elo[a] = ra + K * ((1 - sh) - (1 - exp_h))
    return elo


# ---------------------------------------------------------------
# 3. Build DC ratings
# ---------------------------------------------------------------
def build_dc_ratings(df):
    played = df.dropna(subset=["home_score", "away_score"])
    d = played[played["date"] >= SINCE].copy()
    counts = pd.concat([d["home_team"], d["away_team"]]).value_counts()
    universe = set(counts[counts >= MIN_GAMES].index)
    m = d[d["home_team"].isin(universe) & d["away_team"].isin(universe)].copy()
    ref_date = m["date"].max()
    m["age"] = (ref_date - m["date"]).dt.days
    m["weight"] = 0.5 ** (m["age"] / HALF_LIFE_DAYS)
    m["weight"] *= m["tournament"].apply(tournament_weight)
    team_matches = {t: [] for t in universe}
    for row in m.itertuples(index=False):
        team_matches[row.home_team].append((row.away_team, row.home_score, row.away_score, row.weight))
        team_matches[row.away_team].append((row.home_team, row.away_score, row.home_score, row.weight))
    ws = {t: sum(w * gs for _, gs, gc, w in team_matches[t]) for t in universe}
    wc_ = {t: sum(w * gc for _, gs, gc, w in team_matches[t]) for t in universe}
    mu = (m["weight"] * (m["home_score"] + m["away_score"])).sum() / (m["weight"].sum() * 2)
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
# 4. Match sampling with Elo+DC blend (cached per pair)
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

def match_grid(a, b, attack, defense, mu, home_adv, elo):
    key = (a, b)
    if key in GRID_CACHE: return GRID_CACHE[key]

    exp_a = attack[a] * defense[b] * mu
    exp_b = attack[b] * defense[a] * mu
    tilt = math.sqrt(home_adv)
    if a in HOSTS and b not in HOSTS: exp_a *= tilt; exp_b /= tilt
    elif b in HOSTS and a not in HOSTS: exp_b *= tilt; exp_a /= tilt

    pa = [poisson_probability(g, exp_a) for g in range(MAX_GOALS + 1)]
    pb = [poisson_probability(g, exp_b) for g in range(MAX_GOALS + 1)]
    scores, dc_probs = [], []
    dc_win = dc_loss = total = 0.0
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            p = pa[x] * pb[y] * dixon_coles_tau(x, y, exp_a, exp_b, RHO)
            scores.append((x, y)); dc_probs.append(p); total += p
            if x > y: dc_win += p
            elif x < y: dc_loss += p
    dc_probs = [p / total for p in dc_probs]
    dc_win /= total; dc_loss /= total; dc_draw = 1 - dc_win - dc_loss

    ra = elo.get(a, ELO_START); rb = elo.get(b, ELO_START)
    exp_h = 1 / (1 + 10 ** ((rb - ra) / 400))
    el = [max(exp_h - DRAW_BASE/2, 0.01), DRAW_BASE, max((1-exp_h) - DRAW_BASE/2, 0.01)]

    dc = [dc_win, dc_draw, dc_loss]
    blended = [dc[i]*(1-ELO_WEIGHT) + el[i]*ELO_WEIGHT for i in range(3)]
    s = sum(blended); blended = [x/s for x in blended]

    # Scale DC scoreline grid to match the blended outcome probabilities
    scale_win = blended[0] / dc_win if dc_win > 0 else 1
    scale_draw = blended[1] / dc_draw if dc_draw > 0 else 1
    scale_loss = blended[2] / dc_loss if dc_loss > 0 else 1
    adjusted = []
    for i, (x, y) in enumerate(scores):
        if x > y: adjusted.append(dc_probs[i] * scale_win)
        elif x == y: adjusted.append(dc_probs[i] * scale_draw)
        else: adjusted.append(dc_probs[i] * scale_loss)
    s2 = sum(adjusted)
    adjusted = [p / s2 for p in adjusted]
    cumulative = []
    run = 0.0
    for p in adjusted: run += p; cumulative.append(run)

    edge_a = blended[0] / (blended[0] + blended[2])
    GRID_CACHE[key] = (scores, cumulative, edge_a)
    return GRID_CACHE[key]


def sample_score(a, b, attack, defense, mu, home_adv, elo):
    scores, cumulative, _ = match_grid(a, b, attack, defense, mu, home_adv, elo)
    return scores[bisect.bisect_left(cumulative, random.random())]


# ---------------------------------------------------------------
# 5. Tournament structure
# ---------------------------------------------------------------
def get_world_cup(df):
    gs = df[(df["tournament"] == "FIFA World Cup") &
            (df["date"] >= "2026-06-01") & (df["date"] <= "2026-06-27")]
    fixtures = list(zip(gs["home_team"], gs["away_team"]))
    adj = {}
    for h, a in fixtures:
        adj.setdefault(h, set()).add(a); adj.setdefault(a, set()).add(h)
    seen, groups = set(), []
    for t in adj:
        if t in seen: continue
        comp, stack = {t}, [t]
        while stack:
            for v in adj[stack.pop()]:
                if v not in comp: comp.add(v); stack.append(v)
        seen |= comp; groups.append(sorted(comp))
    return fixtures, groups


# ---------------------------------------------------------------
# 6. Simulate one full tournament
# ---------------------------------------------------------------
def simulate_once(fixtures, groups, attack, defense, mu, home_adv, elo):
    pts = {t: 0 for g in groups for t in g}
    gd = {t: 0 for t in pts}; gf = {t: 0 for t in pts}

    for h, a in fixtures:
        hs, as_ = sample_score(h, a, attack, defense, mu, home_adv, elo)
        gd[h] += hs - as_; gd[a] += as_ - hs
        gf[h] += hs; gf[a] += as_
        if hs > as_: pts[h] += 3
        elif hs < as_: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1

    def tkey(t): return (pts[t], gd[t], gf[t], random.random())

    qualified, thirds = [], []
    for g in groups:
        order = sorted(g, key=tkey, reverse=True)
        qualified += order[:2]; thirds.append(order[2])
    thirds = sorted(thirds, key=tkey, reverse=True)
    qualified += thirds[:8]

    field = sorted(qualified, key=tkey, reverse=True)
    round_names = ["R16", "QF", "SF", "Final", "Champion"]
    results = {t: "groups" for t in pts}
    rnd = 0
    while len(field) > 1:
        nxt = []
        n = len(field)
        for i in range(n // 2):
            a, b = field[i], field[n - 1 - i]
            hs, as_ = sample_score(a, b, attack, defense, mu, home_adv, elo)
            if hs == as_:
                _, _, edge_a = match_grid(a, b, attack, defense, mu, home_adv, elo)
                winner = a if random.random() < edge_a else b
            else:
                winner = a if hs > as_ else b
            nxt.append(winner)
        for t in nxt:
            results[t] = round_names[min(rnd, len(round_names) - 1)]
        field = nxt; rnd += 1
    results[field[0]] = "Champion"
    return results


# ---------------------------------------------------------------
# 7. Run
# ---------------------------------------------------------------
def main():
    df = load_all()
    elo = build_elo(df)
    attack, defense, mu, home_adv = build_dc_ratings(df)
    fixtures, groups = get_world_cup(df)
    print(f"Engine: {len(attack)} DC-rated teams, {len(elo)} Elo-rated teams")
    print(f"Fixtures: {len(fixtures)} group games, {len(groups)} groups")
    print(f"Simulating {N_SIMS:,} tournaments...\n")

    order = ["Champion", "Final", "SF", "QF", "R16", "R32", "groups"]
    reach = {t: {s: 0 for s in order} for g in groups for t in g}

    for i in range(N_SIMS):
        results = simulate_once(fixtures, groups, attack, defense, mu, home_adv, elo)
        for t, stage in results.items():
            idx = order.index(stage)
            for s in order[idx:]: reach[t][s] += 1
        if (i + 1) % 2000 == 0:
            print(f"  ...{i+1:,} simulated")

    print(f"\n{'TEAM':<18}{'WIN':>8}{'FINAL':>8}{'SEMI':>8}{'QF':>8}{'KO':>8}")
    print("-" * 58)
    ranked = sorted(reach, key=lambda t: reach[t]["Champion"], reverse=True)
    for t in ranked[:20]:
        r = reach[t]
        print(f"{t:<18}"
              f"{r['Champion']/N_SIMS*100:>7.1f}%"
              f"{r['Final']/N_SIMS*100:>7.1f}%"
              f"{r['SF']/N_SIMS*100:>7.1f}%"
              f"{r['QF']/N_SIMS*100:>7.1f}%"
              f"{r['R16']/N_SIMS*100:>7.1f}%")


if __name__ == "__main__":
    main()
