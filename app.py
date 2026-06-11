"""
World Cup Match Predictor - App v2 (Step 12 Engine)
----------------------------------------------------
The Streamlit web app, now powered by the Step 12 engine:
  - Tournament-weighted Dixon-Coles ratings
  - Self-built Elo ratings
  - 40/60 Elo/DC blend  ->  62.9% accuracy, RPS 0.159

Run:  venv/bin/streamlit run app.py
"""

import math
import pandas as pd
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SINCE = "2019-01-01"
HALF_LIFE_DAYS = 365
MIN_GAMES, ITERATIONS, MAX_GOALS, RHO = 10, 25, 6, -0.13
ELO_WEIGHT = 0.4
ELO_START  = 1500
ELO_K      = 32
DRAW_BASE  = 0.22

TOURN_WEIGHTS = {
    "FIFA World Cup": 1.5, "UEFA Euro": 1.4, "Copa America": 1.4,
    "Africa Cup of Nations": 1.3, "AFC Asian Cup": 1.3,
    "CONCACAF Gold Cup": 1.2, "FIFA World Cup qualification": 1.1,
    "UEFA Nations League": 1.1, "Friendly": 0.4,
}

def tournament_weight(name):
    for key, val in TOURN_WEIGHTS.items():
        if key.lower() in name.lower(): return val
    return 1.0

@st.cache_data(ttl=6*3600, show_spinner="Downloading match data...")
def load_data():
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

@st.cache_data(ttl=6*3600, show_spinner="Building Elo ratings...")
def build_elo(_df):
    elo = {}
    for row in _df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = elo.get(h, ELO_START); ra = elo.get(a, ELO_START)
        exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
        sh = 1 if row.home_score > row.away_score else (0.5 if row.home_score == row.away_score else 0)
        K = ELO_K * tournament_weight(row.tournament)
        elo[h] = rh + K * (sh - exp_h)
        elo[a] = ra + K * ((1 - sh) - (1 - exp_h))
    return elo

@st.cache_data(ttl=6*3600, show_spinner="Building team strength ratings...")
def build_dc_ratings(_df):
    d = _df[_df["date"] >= SINCE].copy()
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

def poisson_probability(k, expected):
    return (expected ** k) * math.exp(-expected) / math.factorial(k)

def dixon_coles_tau(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0

def predict(team_a, team_b, attack, defense, mu, home_adv, elo, home_team=None):
    exp_a = attack[team_a] * defense[team_b] * mu
    exp_b = attack[team_b] * defense[team_a] * mu
    tilt = math.sqrt(home_adv)
    if home_team == team_a: exp_a *= tilt; exp_b /= tilt
    elif home_team == team_b: exp_b *= tilt; exp_a /= tilt
    pa = [poisson_probability(g, exp_a) for g in range(MAX_GOALS + 1)]
    pb = [poisson_probability(g, exp_b) for g in range(MAX_GOALS + 1)]
    grid, total = {}, 0.0
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            p = pa[x] * pb[y] * dixon_coles_tau(x, y, exp_a, exp_b, RHO)
            grid[(x, y)] = p; total += p
    for key in grid: grid[key] /= total
    dc = [sum(p for (x,y),p in grid.items() if x>y),
          sum(p for (x,y),p in grid.items() if x==y),
          sum(p for (x,y),p in grid.items() if x<y)]
    ra = elo.get(team_a, ELO_START); rb = elo.get(team_b, ELO_START)
    exp_h = 1 / (1 + 10 ** ((rb - ra) / 400))
    el = [max(exp_h - DRAW_BASE/2, 0.01), DRAW_BASE, max((1-exp_h) - DRAW_BASE/2, 0.01)]
    blended = [dc[i]*(1-ELO_WEIGHT) + el[i]*ELO_WEIGHT for i in range(3)]
    s = sum(blended); blended = [x/s for x in blended]
    return blended[0], blended[1], blended[2], grid, ra, rb, exp_a, exp_b

# ---------------------------------------------------------------
# Page
# ---------------------------------------------------------------
st.set_page_config(page_title="World Cup Predictor", page_icon="⚽", layout="centered")
st.title("⚽ World Cup 2026 Match Predictor")
st.caption("Elo + Dixon-Coles blend · tournament-weighted ratings · backtested 62.9% accuracy")

df = load_data()
elo = build_elo(df)
attack, defense, mu, home_adv = build_dc_ratings(df)
team_list = sorted(attack.keys())

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Team A", team_list, index=team_list.index("Brazil"))
with col2:
    team_b = st.selectbox("Team B", team_list, index=team_list.index("England"))

venue = st.radio(
    "Venue",
    ["Neutral (World Cup style)", f"{team_a} at home", f"{team_b} at home"],
    horizontal=True,
)

if team_a == team_b:
    st.warning("Pick two different teams.")
    st.stop()

home_team = None
if venue == f"{team_a} at home": home_team = team_a
elif venue == f"{team_b} at home": home_team = team_b

p_a, p_d, p_b, grid, elo_a, elo_b, exp_a, exp_b = predict(
    team_a, team_b, attack, defense, mu, home_adv, elo, home_team
)

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Elo " + team_a, f"{elo_a:.0f}")
c2.metric("Elo " + team_b, f"{elo_b:.0f}")
c3.metric("xG " + team_a, f"{exp_a:.2f}")
c4.metric("xG " + team_b, f"{exp_b:.2f}")

st.divider()

c1, c2, c3 = st.columns(3)
c1.metric(f"{team_a} win", f"{p_a*100:.1f}%")
c2.metric("Draw", f"{p_d*100:.1f}%")
c3.metric(f"{team_b} win", f"{p_b*100:.1f}%")

st.progress(p_a, text=f"{team_a} win")
st.progress(p_d, text="Draw")
st.progress(p_b, text=f"{team_b} win")

st.subheader("Most likely scorelines")
top = sorted(grid.items(), key=lambda kv: kv[1], reverse=True)[:5]
for (x, y), p in top:
    bar = "█" * int(p * 200)
    st.write(f"{team_a} **{x} - {y}** {team_b} &nbsp;&nbsp; {p*100:.1f}%  `{bar}`")

st.subheader("Full scoreline probabilities (%)")
heat = pd.DataFrame(
    [[grid[(x, y)] * 100 for y in range(MAX_GOALS + 1)] for x in range(MAX_GOALS + 1)],
    index=[f"{team_a} {x}" for x in range(MAX_GOALS + 1)],
    columns=[f"{team_b} {y}" for y in range(MAX_GOALS + 1)],
)
try:
    st.dataframe(heat.style.background_gradient(cmap="Greens").format("{:.1f}"))
except Exception:
    st.dataframe(heat.style.format("{:.1f}"))

st.caption(
    f"Model: {len(team_list)} teams · Elo+DC blend (40/60) · "
    f"tournament-weighted · RPS 0.159 on 966 unseen matches"
)
