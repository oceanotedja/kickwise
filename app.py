"""
Kickwise - World Cup 2026 Predictor
Mobile-first Streamlit app matching the Figma design.
Run: venv/bin/streamlit run app.py
"""
import math, random, bisect
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SINCE="2019-01-01"; HALF=365; MIN_G=10; IT=25; MG=6; RHO=-0.13
ELO_W=0.4; ELO_S=1500; ELO_K=32; DB=0.22
HOSTS={"United States","Mexico","Canada"}
TW={"FIFA World Cup":1.5,"UEFA Euro":1.4,"Copa América":1.4,
    "Africa Cup of Nations":1.3,"AFC Asian Cup":1.3,
    "CONCACAF Gold Cup":1.2,"FIFA World Cup qualification":1.1,
    "UEFA Nations League":1.1,"Friendly":0.4}

FLAGS = {
    "Argentina":"🇦🇷","Algeria":"🇩🇿","Austria":"🇦🇹","Jordan":"🇯🇴",
    "Australia":"🇦🇺","Paraguay":"🇵🇾","Turkey":"🇹🇷","United States":"🇺🇸",
    "Belgium":"🇧🇪","Egypt":"🇪🇬","Iran":"🇮🇷","New Zealand":"🇳🇿",
    "Bosnia and Herzegovina":"🇧🇦","Canada":"🇨🇦","Qatar":"🇶🇦","Switzerland":"🇨🇭",
    "Brazil":"🇧🇷","Haiti":"🇭🇹","Morocco":"🇲🇦","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Cape Verde":"🇨🇻","Saudi Arabia":"🇸🇦","Spain":"🇪🇸","Uruguay":"🇺🇾",
    "Colombia":"🇨🇴","DR Congo":"🇨🇩","Portugal":"🇵🇹","Uzbekistan":"🇺🇿",
    "Croatia":"🇭🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Ghana":"🇬🇭","Panama":"🇵🇦",
    "Curaçao":"🇨🇼","Ecuador":"🇪🇨","Germany":"🇩🇪","Ivory Coast":"🇨🇮",
    "Czech Republic":"🇨🇿","Mexico":"🇲🇽","South Africa":"🇿🇦","South Korea":"🇰🇷",
    "France":"🇫🇷","Iraq":"🇮🇶","Norway":"🇳🇴","Senegal":"🇸🇳",
    "Japan":"🇯🇵","Netherlands":"🇳🇱","Sweden":"🇸🇪","Tunisia":"🇹🇳",
}

# Official FIFA draw (Dec 5, 2025) - correct group letters
GROUPS = {
    "A":["Mexico","South Africa","South Korea","Czech Republic"],
    "B":["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
    "C":["Brazil","Morocco","Haiti","Scotland"],
    "D":["United States","Paraguay","Australia","Turkey"],
    "E":["Germany","Curaçao","Ivory Coast","Ecuador"],
    "F":["Netherlands","Japan","Sweden","Tunisia"],
    "G":["Belgium","Egypt","Iran","New Zealand"],
    "H":["Spain","Cape Verde","Saudi Arabia","Uruguay"],
    "I":["France","Senegal","Iraq","Norway"],
    "J":["Argentina","Algeria","Austria","Jordan"],
    "K":["Portugal","DR Congo","Uzbekistan","Colombia"],
    "L":["England","Croatia","Ghana","Panama"],
}

# ── MANUAL RESULTS ──
# The public dataset can lag 1-2 days behind live matches.
# Add finished results here: (home, away, home_goals, away_goals, date_YYYY-MM-DD)
# They are injected directly into wc_all so predictions move to results immediately
# and group standings update with 3pts win / 1pt draw / 0pt loss.
MANUAL_RESULTS = [
    ("Mexico", "South Africa", 2, 0, "2026-06-11"),        # Group A
    ("South Korea", "Czech Republic", 2, 1, "2026-06-12"), # Group A
    ("United States", "Paraguay", 4, 1, "2026-06-12"),     # Group D
    ("Canada", "Bosnia and Herzegovina", 1, 1, "2026-06-12"), # Group B
    ("Qatar", "Switzerland", 1, 1, "2026-06-13"),          # Group B
    ("Brazil", "Morocco", 1, 1, "2026-06-13"),             # Group C
    ("Haiti", "Scotland", 0, 1, "2026-06-14"),             # Group C
    ("Australia", "Turkey", 2, 0, "2026-06-14"),           # Group D
    ("Germany", "Curaçao", 7, 1, "2026-06-14"),            # Group E
    ("Netherlands", "Japan", 2, 2, "2026-06-14"),          # Group F
    ("Ivory Coast", "Ecuador", 1, 0, "2026-06-14"),        # Group E
    ("Sweden", "Tunisia", 5, 1, "2026-06-15"),             # Group F
    ("Spain", "Cape Verde", 0, 0, "2026-06-15"),           # Group H
    ("Belgium", "Egypt", 1, 1, "2026-06-15"),              # Group G
    ("Saudi Arabia", "Uruguay", 1, 1, "2026-06-15"),       # Group H
    ("Iran", "New Zealand", 2, 2, "2026-06-16"),           # Group G
    ("France", "Senegal", 3, 1, "2026-06-16"),             # Group I
    ("Iraq", "Norway", 1, 4, "2026-06-16"),                # Group I
    ("Argentina", "Algeria", 3, 0, "2026-06-17"),          # Group J
    ("Austria", "Jordan", 3, 1, "2026-06-17"),             # Group J
    ("Portugal", "DR Congo", 1, 1, "2026-06-17"),          # Group K
    ("England", "Croatia", 4, 2, "2026-06-17"),            # Group L
    ("Ghana", "Panama", 1, 0, "2026-06-17"),               # Group L
    ("Uzbekistan", "Colombia", 1, 3, "2026-06-18"),        # Group K
    ("Czech Republic", "South Africa", 1, 1, "2026-06-18"), # Group A
    ("Switzerland", "Bosnia and Herzegovina", 4, 1, "2026-06-18"), # Group B
    ("Canada", "Qatar", 6, 0, "2026-06-18"),               # Group B
    ("Mexico", "South Korea", 1, 0, "2026-06-19"),         # Group A
    ("United States", "Australia", 2, 0, "2026-06-19"),  # Group D
    ("Scotland", "Morocco", 0, 1, "2026-06-19"),  # Group C
    ("Brazil", "Haiti", 3, 0, "2026-06-20"),  # Group C
    ("Turkey", "Paraguay", 0, 1, "2026-06-20"),  # Group D
    ("Germany", "Ivory Coast", 2, 1, "2026-06-20"),  # Group E
    ("Netherlands", "Sweden", 5, 1, "2026-06-20"),  # Group F

]

# ── KICKOFF TIMES (UTC) ──────────────────────────────────────
# Source: Al Jazeera / Sky Sports (confirmed GMT). Display in BJT (UTC+8).
MATCH_TIMES_UTC = {
    # Group A
    ("Mexico","South Africa"):                  "2026-06-11 19:00",
    ("South Korea","Czech Republic"):           "2026-06-12 02:00",
    ("Czech Republic","South Africa"):          "2026-06-18 16:00",
    ("Mexico","South Korea"):                   "2026-06-19 01:00",
    ("Czech Republic","Mexico"):                "2026-06-25 01:00",
    ("South Africa","South Korea"):             "2026-06-25 01:00",
    # Group B
    ("Canada","Bosnia and Herzegovina"):        "2026-06-12 19:00",
    ("Qatar","Switzerland"):                    "2026-06-13 19:00",
    ("Switzerland","Bosnia and Herzegovina"):   "2026-06-18 19:00",
    ("Canada","Qatar"):                         "2026-06-18 22:00",
    ("Switzerland","Canada"):                   "2026-06-24 19:00",
    ("Bosnia and Herzegovina","Qatar"):         "2026-06-24 19:00",
    # Group C
    ("Brazil","Morocco"):                       "2026-06-13 22:00",
    ("Haiti","Scotland"):                       "2026-06-14 01:00",
    ("Scotland","Morocco"):                     "2026-06-19 22:00",
    ("Brazil","Haiti"):                         "2026-06-20 00:30",
    ("Scotland","Brazil"):                      "2026-06-24 22:00",
    ("Morocco","Haiti"):                        "2026-06-24 22:00",
    # Group D
    ("United States","Paraguay"):               "2026-06-12 01:00",
    ("Australia","Turkey"):                     "2026-06-14 04:00",
    ("United States","Australia"):              "2026-06-19 19:00",
    ("Turkey","Paraguay"):                      "2026-06-20 03:00",
    ("Turkey","United States"):                 "2026-06-26 02:00",
    ("Paraguay","Australia"):                   "2026-06-26 02:00",
    # Group E
    ("Germany","Curaçao"):                      "2026-06-14 17:00",
    ("Ivory Coast","Ecuador"):                  "2026-06-14 23:00",
    ("Germany","Ivory Coast"):                  "2026-06-20 20:00",
    ("Ecuador","Curaçao"):                      "2026-06-21 03:00",
    ("Ecuador","Germany"):                      "2026-06-25 20:00",
    ("Curaçao","Ivory Coast"):                  "2026-06-25 20:00",
    # Group F
    ("Netherlands","Japan"):                    "2026-06-14 20:00",
    ("Sweden","Tunisia"):                       "2026-06-15 02:00",
    ("Netherlands","Sweden"):                   "2026-06-20 17:00",
    ("Tunisia","Japan"):                        "2026-06-21 04:00",
    ("Japan","Sweden"):                         "2026-06-25 23:00",
    ("Tunisia","Netherlands"):                  "2026-06-25 23:00",
    # Group G
    ("Belgium","Egypt"):                        "2026-06-15 19:00",
    ("Iran","New Zealand"):                     "2026-06-16 01:00",
    ("Belgium","Iran"):                         "2026-06-21 19:00",
    ("New Zealand","Egypt"):                    "2026-06-22 01:00",
    ("Egypt","Iran"):                           "2026-06-27 03:00",
    ("New Zealand","Belgium"):                  "2026-06-27 03:00",
    # Group H
    ("Spain","Cape Verde"):                     "2026-06-15 16:00",
    ("Saudi Arabia","Uruguay"):                 "2026-06-15 22:00",
    ("Spain","Saudi Arabia"):                   "2026-06-21 16:00",
    ("Uruguay","Cape Verde"):                   "2026-06-21 22:00",
    ("Cape Verde","Saudi Arabia"):              "2026-06-27 00:00",
    ("Uruguay","Spain"):                        "2026-06-27 00:00",
    # Group I
    ("France","Senegal"):                       "2026-06-16 19:00",
    ("Iraq","Norway"):                          "2026-06-16 22:00",
    ("France","Iraq"):                          "2026-06-22 21:00",
    ("Norway","Senegal"):                       "2026-06-23 00:00",
    ("Norway","France"):                        "2026-06-26 19:00",
    ("Senegal","Iraq"):                         "2026-06-26 19:00",
    # Group J
    ("Argentina","Algeria"):                    "2026-06-17 01:00",
    ("Austria","Jordan"):                       "2026-06-17 04:00",
    ("Argentina","Austria"):                    "2026-06-22 17:00",
    ("Jordan","Algeria"):                       "2026-06-23 03:00",
    ("Algeria","Austria"):                      "2026-06-28 02:00",
    ("Jordan","Argentina"):                     "2026-06-28 02:00",
    # Group K
    ("Portugal","DR Congo"):                    "2026-06-17 17:00",
    ("Uzbekistan","Colombia"):                  "2026-06-18 02:00",
    ("Portugal","Uzbekistan"):                  "2026-06-23 17:00",
    ("Colombia","DR Congo"):                    "2026-06-24 02:00",
    ("Colombia","Portugal"):                    "2026-06-27 23:30",
    ("DR Congo","Uzbekistan"):                  "2026-06-27 23:30",
    # Group L
    ("England","Croatia"):                      "2026-06-17 20:00",
    ("Ghana","Panama"):                         "2026-06-17 23:00",
    ("England","Ghana"):                        "2026-06-23 20:00",
    ("Panama","Croatia"):                       "2026-06-23 23:00",
    ("Panama","England"):                       "2026-06-27 21:00",
    ("Croatia","Ghana"):                        "2026-06-27 21:00",
}

def get_bjt_time(h, a):
    """Return kickoff time in Beijing Time (UTC+8) as 'Mon DD · HH:MM BJT'."""
    utc_str = MATCH_TIMES_UTC.get((h, a))
    if not utc_str:
        return None
    dt_bjt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M") + timedelta(hours=8)
    return dt_bjt.strftime("%b %d · %H:%M BJT")

def compute_standings(group_teams, wc_df):
    """Compute live group standings from played WC matches."""
    table = {t:{"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"GD":0,"PTS":0}
             for t in group_teams}
    played = wc_df.dropna(subset=["home_score","away_score"])
    for row in played.itertuples(index=False):
        h,a = row.home_team, row.away_team
        hs,as_ = int(row.home_score), int(row.away_score)
        if h not in table or a not in table: continue
        table[h]["P"]+=1; table[a]["P"]+=1
        table[h]["GF"]+=hs; table[h]["GA"]+=as_
        table[a]["GF"]+=as_; table[a]["GA"]+=hs
        table[h]["GD"]+=hs-as_; table[a]["GD"]+=as_-hs
        if hs>as_:
            table[h]["W"]+=1; table[a]["L"]+=1; table[h]["PTS"]+=3
        elif hs<as_:
            table[a]["W"]+=1; table[h]["L"]+=1; table[a]["PTS"]+=3
        else:
            table[h]["D"]+=1; table[a]["D"]+=1
            table[h]["PTS"]+=1; table[a]["PTS"]+=1
    ranked = sorted(table.items(),
                    key=lambda x:(-x[1]["PTS"],-x[1]["GD"],-x[1]["GF"],x[0]))
    return ranked

def tw(t):
    for k,v in TW.items():
        if k.lower() in t.lower(): return v
    return 1.0

def flag(t): return FLAGS.get(t,"🏳️")

@st.cache_data(ttl=6*3600, show_spinner=False)
def load_and_build(manual_results):
    # manual_results is passed as an argument so the cache key changes
    # whenever we add a new result — no stale "upcoming" after updates.
    df = pd.read_csv(DATA_URL)
    df["date"] = pd.to_datetime(df["date"])
    # Fill in manual results FIRST so they feed the model too:
    # Elo updates, attack/defense ratings, and the favourites ranking
    # all see these matches as played (only where dataset lags behind).
    for mh, ma, mhs, mas, mdate in manual_results:
        mask = (df["home_team"]==mh)&(df["away_team"]==ma)&(df["home_score"].isna())
        df.loc[mask, "home_score"] = mhs
        df.loc[mask, "away_score"] = mas
    played = df.dropna(subset=["home_score","away_score"]).sort_values("date")
    elo = {}
    for r in played.itertuples(index=False):
        h,a=r.home_team,r.away_team
        rh,ra=elo.get(h,ELO_S),elo.get(a,ELO_S)
        eh=1/(1+10**((ra-rh)/400))
        sh=1 if r.home_score>r.away_score else(0.5 if r.home_score==r.away_score else 0)
        K=ELO_K*tw(r.tournament)
        elo[h]=rh+K*(sh-eh); elo[a]=ra+K*((1-sh)-(1-eh))
    d=played[played["date"]>=SINCE].copy()
    counts=pd.concat([d["home_team"],d["away_team"]]).value_counts()
    uni=set(counts[counts>=MIN_G].index)
    m=d[d["home_team"].isin(uni)&d["away_team"].isin(uni)].copy()
    ref=m["date"].max()
    m["w"]=0.5**((ref-m["date"]).dt.days/HALF)*m["tournament"].apply(tw)
    tm={t:[] for t in uni}
    for r in m.itertuples(index=False):
        tm[r.home_team].append((r.away_team,r.home_score,r.away_score,r.w))
        tm[r.away_team].append((r.home_team,r.away_score,r.home_score,r.w))
    ws={t:sum(w*g for _,g,gc,w in tm[t]) for t in uni}
    wc_={t:sum(w*gc for _,g,gc,w in tm[t]) for t in uni}
    mu=(m["w"]*(m["home_score"]+m["away_score"])).sum()/(m["w"].sum()*2)
    at={t:1.0 for t in uni}; de={t:1.0 for t in uni}
    for _ in range(IT):
        na,nd={},{}
        for t in uni:
            od=sum(w*de[o] for o,g,gc,w in tm[t]); oa=sum(w*at[o] for o,g,gc,w in tm[t])
            na[t]=ws[t]/(mu*od) if od else 1.0; nd[t]=wc_[t]/(mu*oa) if oa else 1.0
        am=sum(na.values())/len(na); dm=sum(nd.values())/len(nd)
        at={t:v/am for t,v in na.items()}; de={t:v/dm for t,v in nd.items()}
    nn=m[m["neutral"]==False]
    ha=nn["home_score"].mean()/nn["away_score"].mean()
    # Build wc_all from MATCH_TIMES_UTC — authoritative fixture list, no CSV
    # lag or team-name mismatches. Adding to MANUAL_RESULTS instantly moves a
    # match out of Upcoming and into Results / Standings.
    fixture_rows = []
    for (h, a), utc_str in MATCH_TIMES_UTC.items():
        fixture_rows.append({
            "date": pd.Timestamp(utc_str.split()[0]),
            "home_team": h, "away_team": a,
            "home_score": float("nan"), "away_score": float("nan"),
            "tournament": "FIFA World Cup", "neutral": True,
        })
    wc_all = pd.DataFrame(fixture_rows)

    for mh, ma, mhs, mas, mdate in manual_results:
        mask = (wc_all["home_team"]==mh) & (wc_all["away_team"]==ma)
        if mask.any():
            wc_all.loc[mask, "home_score"] = float(mhs)
            wc_all.loc[mask, "away_score"] = float(mas)
        else:
            new_row = pd.DataFrame([{
                "date": pd.Timestamp(mdate),
                "home_team": mh, "away_team": ma,
                "home_score": float(mhs), "away_score": float(mas),
                "tournament": "FIFA World Cup", "neutral": True,
            }])
            wc_all = pd.concat([wc_all, new_row], ignore_index=True)

    upcoming = wc_all[wc_all["home_score"].isna()].copy()
    upcoming["date_fmt"] = upcoming["date"].dt.strftime("%b %d")
    return at, de, mu, ha, elo, sorted(uni), upcoming, wc_all

def pois(k,l): return l**k*math.exp(-l)/math.factorial(k)
def tau(x,y,l,m,r):
    if x==0 and y==0: return 1-l*m*r
    if x==0 and y==1: return 1+l*r
    if x==1 and y==0: return 1+m*r
    if x==1 and y==1: return 1-r
    return 1.0

def predict(a,b,at,de,mu,ha,elo,home=None):
    ea=at.get(a,1.0)*de.get(b,1.0)*mu; eb=at.get(b,1.0)*de.get(a,1.0)*mu
    tilt=math.sqrt(ha)
    if home==a: ea*=tilt; eb/=tilt
    elif home==b: eb*=tilt; ea/=tilt
    pa=[pois(g,ea) for g in range(MG+1)]; pb=[pois(g,eb) for g in range(MG+1)]
    grid,tot={},0.0
    for x in range(MG+1):
        for y in range(MG+1):
            p=pa[x]*pb[y]*tau(x,y,ea,eb,RHO); grid[(x,y)]=p; tot+=p
    for k in grid: grid[k]/=tot
    dc=[sum(p for (x,y),p in grid.items() if x>y),
        sum(p for (x,y),p in grid.items() if x==y),
        sum(p for (x,y),p in grid.items() if x<y)]
    ra,rb=elo.get(a,ELO_S),elo.get(b,ELO_S)
    eh=1/(1+10**((rb-ra)/400))
    el=[max(eh-DB/2,0.01),DB,max((1-eh)-DB/2,0.01)]
    bl=[dc[i]*(1-ELO_W)+el[i]*ELO_W for i in range(3)]
    s=sum(bl); bl=[x/s for x in bl]
    top=sorted(grid.items(),key=lambda kv:kv[1],reverse=True)[:5]
    return bl[0],bl[1],bl[2],top,ea,eb,ra,rb

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Kickwise",page_icon="⚡",layout="centered",
                   initial_sidebar_state="collapsed")
# Auto-refresh every 2 hours (7,200,000 ms)
if HAS_AUTOREFRESH:
    st_autorefresh(interval=2*60*60*1000, limit=None, key="kickwise_refresh")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&display=swap');

html,body,[class*="css"]{
  font-family:'Inter',sans-serif!important;
  background:#0a0a0a!important;
  color:#e0e0e0!important;
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 1rem 5rem!important;max-width:430px!important;margin:auto;}
.stRadio>div{display:flex!important;gap:6px!important;flex-wrap:nowrap!important;overflow-x:auto!important;padding:2px 0!important;background:transparent!important;-webkit-overflow-scrolling:touch;}
.stRadio label{background:#1c1c1c!important;border-radius:999px!important;padding:6px 14px!important;font-size:0.78rem!important;font-weight:600!important;color:#888!important;white-space:nowrap!important;cursor:pointer!important;border:none!important;}
.stRadio label:has(input:checked){background:#00FF7F!important;color:#000!important;}
/* Compact single-letter group pills: small circles, all 12 fit one row */
.grp-radio .stRadio>div{gap:5px!important;justify-content:space-between!important;overflow-x:visible!important;}
.grp-radio .stRadio label{padding:0!important;width:26px!important;height:26px!important;min-width:26px!important;border-radius:50%!important;display:flex!important;align-items:center!important;justify-content:center!important;font-size:0.72rem!important;}
.grp-radio .stRadio label>div:first-child{display:none!important;}  /* hide the radio dot */
.stSelectbox>div>div{background:#1c1c1c!important;border:1px solid #2a2a2a!important;border-radius:12px!important;color:#fff!important;}
div[data-testid="stVerticalBlock"]{gap:0!important;}

/* page header */
.kw-eyebrow{font-size:0.7rem;font-weight:700;color:#F5C518;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;}
.kw-title{font-family:'Bebas Neue',sans-serif;font-size:2.6rem;color:#fff;line-height:1;margin:0 0 2px;}
.kw-sub{font-size:0.8rem;color:#555;margin-bottom:1rem;}
.kw-divider{height:1px;background:#1e1e1e;margin:0.75rem 0 1rem;}

/* fixture card */
.fx-card{background:#161616;border:1px solid #222;border-radius:16px;padding:0.85rem 1rem;margin-bottom:8px;display:flex;align-items:center;justify-content:space-between;}
.fx-time{font-size:0.75rem;font-weight:600;color:#aaa;letter-spacing:0.05em;}
.fx-predict{display:flex;align-items:center;gap:5px;background:#1c2e1c;border:1px solid #00FF7F33;border-radius:999px;padding:5px 12px;font-size:0.72rem;font-weight:700;color:#00FF7F;letter-spacing:0.05em;}

/* match detail */
.match-card{background:#111;border:1px solid #1e1e1e;border-radius:20px;padding:1.25rem 1rem;margin:0.5rem 0 1rem;}
.match-teams{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.25rem;}
.match-team{font-size:1rem;font-weight:800;color:#fff;flex:1;line-height:1.2;}
.match-team.right{text-align:right;}
.match-vs{background:#1e1e1e;border-radius:8px;padding:5px 10px;font-size:0.65rem;color:#444;font-weight:700;letter-spacing:0.1em;margin:0 8px;}

/* prob bar */
.prob-wrap{display:flex;height:52px;gap:4px;border-radius:12px;overflow:hidden;margin-bottom:1rem;}
.prob-seg{display:flex;flex-direction:column;align-items:center;justify-content:center;}
.prob-seg.win{background:linear-gradient(160deg,#00FF7F,#00cc60);}
.prob-seg.draw{background:#252525;}
.prob-seg.loss{background:linear-gradient(160deg,#1a56db,#1341b0);}
.prob-pct{font-size:1rem;font-weight:800;color:#fff;}
.prob-lbl{font-size:0.55rem;color:rgba(255,255,255,0.55);text-transform:uppercase;letter-spacing:0.06em;margin-top:1px;}

/* stats row */
.stats-row{display:flex;gap:8px;margin-bottom:0.75rem;}
.stat-box{flex:1;background:#161616;border-radius:12px;padding:10px;text-align:center;}
.stat-val{font-size:1rem;font-weight:800;color:#00FF7F;}
.stat-lbl{font-size:0.6rem;color:#555;text-transform:uppercase;letter-spacing:0.06em;margin-top:2px;}

/* scorelines */
.sl-title{font-size:0.65rem;color:#444;text-transform:uppercase;letter-spacing:0.1em;margin:1rem 0 0.5rem;}
.sl-item{display:flex;align-items:center;background:#161616;border-radius:10px;padding:0.55rem 0.75rem;margin-bottom:6px;}
.sl-score{font-size:0.88rem;font-weight:700;color:#fff;flex:1;}
.sl-bar-wrap{flex:2;height:3px;background:#222;border-radius:2px;margin:0 10px;}
.sl-bar{height:3px;border-radius:2px;background:#00FF7F;}
.sl-pct{font-size:0.82rem;font-weight:700;color:#00FF7F;min-width:2.5rem;text-align:right;}

/* standings */
.grp-row{display:flex;gap:8px;margin-bottom:1rem;flex-wrap:wrap;}
.grp-pill{width:36px;height:36px;border-radius:50%;background:#1c1c1c;display:flex;align-items:center;justify-content:center;font-size:0.85rem;font-weight:700;color:#888;cursor:pointer;}
.grp-pill.active{background:#00FF7F;color:#000;}
.st-header{display:flex;padding:0 0.5rem;margin-bottom:6px;}
.st-header span{font-size:0.6rem;color:#444;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;}
.st-row{display:flex;align-items:center;background:#161616;border-radius:12px;padding:0.65rem 0.75rem;margin-bottom:6px;border-left:3px solid transparent;}
.st-row.qualify{border-left-color:#00FF7F;}
.st-row.third{border-left-color:#F5C518;}
.st-pos{font-size:0.85rem;font-weight:700;color:#888;width:20px;}
.st-flag{font-size:1.2rem;margin:0 8px;}
.st-name{flex:1;}
.st-team-code{font-size:0.85rem;font-weight:700;color:#fff;}
.st-team-full{font-size:0.65rem;color:#555;}
.st-stat{font-size:0.8rem;font-weight:600;color:#aaa;width:22px;text-align:center;}
.st-pts{font-size:0.85rem;font-weight:800;color:#00FF7F;width:22px;text-align:center;}
.st-legend{display:flex;flex-direction:column;gap:6px;margin-top:1rem;}
.st-legend-item{display:flex;align-items:center;gap:8px;font-size:0.72rem;color:#555;}
.st-dot{width:10px;height:10px;border-radius:50%;}

/* my picks */
.picks-stat{flex:1;background:#161616;border-radius:14px;padding:1rem 0.75rem;}
.picks-stat-icon{font-size:1.1rem;margin-bottom:6px;}
.picks-stat-val{font-size:1.4rem;font-weight:800;color:#fff;}
.picks-stat-lbl{font-size:0.6rem;color:#555;text-transform:uppercase;letter-spacing:0.08em;margin-top:2px;}
.champ-card{background:#161616;border-radius:16px;padding:1rem;display:flex;align-items:center;gap:1rem;margin-bottom:1.25rem;}
.champ-name{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:#fff;letter-spacing:1px;}
.champ-sub{font-size:0.72rem;color:#555;margin-top:2px;}
.champ-badge{display:inline-flex;align-items:center;gap:4px;background:#1e1a00;border:1px solid #F5C518;border-radius:999px;padding:4px 10px;font-size:0.7rem;font-weight:700;color:#F5C518;margin-top:6px;}
.contender-row{display:flex;align-items:center;background:#161616;border-radius:12px;padding:0.75rem 1rem;margin-bottom:6px;}
.cont-pos{font-size:0.85rem;font-weight:700;color:#555;width:20px;}
.cont-flag{font-size:1.1rem;margin:0 10px;}
.cont-name{flex:1;font-size:0.9rem;font-weight:700;color:#fff;}
.cont-mult{font-size:0.9rem;font-weight:700;color:#00FF7F;}
.sect-title{font-size:0.65rem;font-weight:700;color:#444;text-transform:uppercase;letter-spacing:0.1em;margin:1rem 0 0.5rem;}

/* bottom nav */
.bottom-nav{position:fixed;bottom:0;left:0;right:0;background:#111;border-top:1px solid #1e1e1e;z-index:999;display:flex;justify-content:space-around;padding:8px 0 12px;max-width:430px;margin:auto;}
.nav-link{display:flex;flex-direction:column;align-items:center;gap:3px;text-decoration:none;flex:1;padding:4px 0;}
.nav-icon{font-size:1.35rem;color:#555;line-height:1;}
.nav-icon.active{color:#00FF7F;}
.nav-lbl{font-size:0.58rem;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:0.06em;}
.nav-lbl.active{color:#00FF7F;}
</style>
""", unsafe_allow_html=True)

# ── Load ────────────────────────────────────────────────────
with st.spinner(""):
    at,de,mu,ha,elo,team_list,upcoming,wc_all = load_and_build(tuple(MANUAL_RESULTS))

# ── Navigation state ────────────────────────────────────────
if "tab" not in st.session_state: st.session_state.tab = "predict"
if "grp" not in st.session_state: st.session_state.grp = "A"
if "pred_match" not in st.session_state: st.session_state.pred_match = None
if "last_qp" not in st.session_state: st.session_state.last_qp = None

# Bottom-nav links set ?tab=... in the URL; apply it once per change
qp_tab = st.query_params.get("tab")
if qp_tab in ("predict", "standings", "picks") and qp_tab != st.session_state.last_qp:
    st.session_state.tab = qp_tab
    st.session_state.last_qp = qp_tab

# nav buttons via query or columns
c1,c2,c3 = st.columns(3)
with c1:
    if st.button("⚡ PREDICT", use_container_width=True,
                 type="primary" if st.session_state.tab=="predict" else "secondary"):
        st.session_state.tab="predict"; st.rerun()
with c2:
    if st.button("📊 STANDINGS", use_container_width=True,
                 type="primary" if st.session_state.tab=="standings" else "secondary"):
        st.session_state.tab="standings"; st.rerun()
with c3:
    if st.button("⭐ FAVOURITES", use_container_width=True,
                 type="primary" if st.session_state.tab=="picks" else "secondary"):
        st.session_state.tab="picks"; st.rerun()

st.markdown('<div class="kw-divider"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PREDICT TAB
# ════════════════════════════════════════════════════════════
if st.session_state.tab == "predict":
    st.markdown("""
    <div class="kw-eyebrow">🏆 World Cup 2026</div>
    <div class="kw-title">PREDICTIONS</div>
    <div class="kw-sub">Powered by Kickwise AI · 62.9% accuracy</div>
    """, unsafe_allow_html=True)

    # Tab: Upcoming vs Results
    view_mode = st.radio("View", ["⚽ Upcoming", "📋 Results"],
                         horizontal=True, label_visibility="collapsed")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if view_mode == "📋 Results":
        # Show all played matches
        played_matches = wc_all.dropna(subset=["home_score","away_score"]).sort_values("date", ascending=False)
        if played_matches.empty:
            st.info("No results yet — check back after matches finish.")
        else:
            for _, row in played_matches.iterrows():
                h, a = row["home_team"], row["away_team"]
                hs, as_ = int(row["home_score"]), int(row["away_score"])
                date_str = pd.Timestamp(row["date"]).strftime("%b %d")
                winner_h = "color:#00FF7F;font-weight:900" if hs > as_ else ("color:#aaa" if hs < as_ else "color:#F5C518")
                winner_a = "color:#00FF7F;font-weight:900" if as_ > hs else ("color:#aaa" if as_ < hs else "color:#F5C518")
                st.markdown(f"""
                <div class="fx-card" style="border-color:#1a3a1a;">
                  <div style="flex:1;">
                    <div class="fx-time">FT · {date_str}</div>
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-top:4px;">
                      <div style="font-size:0.88rem;font-weight:700;{winner_h}">{flag(h)} {h}</div>
                      <div style="font-size:1.1rem;font-weight:900;color:#fff;margin:0 10px;">{hs} – {as_}</div>
                      <div style="font-size:0.88rem;font-weight:700;{winner_a}">{a} {flag(a)}</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
        st.stop()

    # Round filter
    round_sel = st.radio("Round", ["Group Stage","Round of 32","Round of 16","Quarter Finals","Semi Finals","Final"],
                         horizontal=True, label_visibility="collapsed")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Show fixtures list — sorted by kickoff time (earliest first).
    # upcoming rows always have a MATCH_TIMES_UTC entry since wc_all is built from it.
    if not upcoming.empty:
        upcoming_sorted = upcoming.iloc[
            upcoming.apply(
                lambda r: MATCH_TIMES_UTC.get((r["home_team"], r["away_team"]), "9999"),
                axis=1
            ).argsort().values
        ]
        shown = 0
        for _, row in upcoming_sorted.iterrows():
            h, a = row["home_team"], row["away_team"]
            if h not in at or a not in at: continue
            date_str = row["date_fmt"]
            bjt = get_bjt_time(h, a)
            time_str = bjt if bjt else f"{date_str} · BJT"
            btn_key = f"fx_{h}_{a}"
            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown(f"""
                <div class="fx-card" style="margin-bottom:0">
                  <div>
                    <div class="fx-time">{time_str}</div>
                    <div style="font-size:0.88rem;font-weight:700;color:#fff;margin-top:3px;">
                      {flag(h)} {h} <span style="color:#444;font-size:0.75rem;margin:0 4px;">vs</span> {flag(a)} {a}
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with col2:
                if st.button("⚡ PREDICT", key=btn_key, use_container_width=True):
                    st.session_state.pred_match = (h, a)
                    st.session_state.tab = "predict_detail"
                    st.rerun()
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            shown += 1
            if shown >= 20: break
    else:
        st.info("No upcoming fixtures available.")

# ════════════════════════════════════════════════════════════
# PREDICT DETAIL
# ════════════════════════════════════════════════════════════
elif st.session_state.tab == "predict_detail":
    if st.button("← Back"):
        st.session_state.tab = "predict"; st.rerun()
    h, a = st.session_state.pred_match
    pw, pd_, pb, top, ea, eb, ra, rb = predict(h, a, at, de, mu, ha, elo)
    pw_w = max(int(pw*100), 8)
    pd_w = max(int(pd_*100), 8)
    pb_w = max(int(pb*100), 8)
    st.markdown(f"""
    <div class="kw-eyebrow">🏆 World Cup 2026</div>
    <div class="kw-title">MATCH PREVIEW</div>
    <div class="kw-divider"></div>
    <div class="match-card">
      <div class="match-teams">
        <div class="match-team">{flag(h)}<br>{h}</div>
        <div class="match-vs">VS</div>
        <div class="match-team right">{flag(a)}<br>{a}</div>
      </div>
      <div class="prob-wrap">
        <div class="prob-seg win" style="flex:{pw_w}">
          <div class="prob-pct">{pw*100:.0f}%</div>
          <div class="prob-lbl">Win</div>
        </div>
        <div class="prob-seg draw" style="flex:{pd_w}">
          <div class="prob-pct">{pd_*100:.0f}%</div>
          <div class="prob-lbl">Draw</div>
        </div>
        <div class="prob-seg loss" style="flex:{pb_w}">
          <div class="prob-pct">{pb*100:.0f}%</div>
          <div class="prob-lbl">Win</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # Stats
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-box"><div class="stat-val">{ea:.2f}</div><div class="stat-lbl">{h} xG</div></div>
      <div class="stat-box"><div class="stat-val">{ra:.0f}</div><div class="stat-lbl">{h} Elo</div></div>
      <div class="stat-box"><div class="stat-val">{rb:.0f}</div><div class="stat-lbl">{a} Elo</div></div>
      <div class="stat-box"><div class="stat-val">{eb:.2f}</div><div class="stat-lbl">{a} xG</div></div>
    </div>
    """, unsafe_allow_html=True)
    # Scorelines
    st.markdown('<div class="sl-title">Most likely scorelines</div>', unsafe_allow_html=True)
    max_p = top[0][1] if top else 1
    for (x,y),p in top:
        bw = int((p/max_p)*100)
        st.markdown(f"""
        <div class="sl-item">
          <div class="sl-score">{h} {x} – {y} {a}</div>
          <div class="sl-bar-wrap"><div class="sl-bar" style="width:{bw}%"></div></div>
          <div class="sl-pct">{p*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STANDINGS TAB
# ════════════════════════════════════════════════════════════
elif st.session_state.tab == "standings":
    st.markdown("""
    <div class="kw-eyebrow">🏆 World Cup 2026</div>
    <div class="kw-title">STANDINGS</div>
    """, unsafe_allow_html=True)

    # Refresh button
    col_r1, col_r2 = st.columns([3,1])
    with col_r2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear(); st.rerun()

    # Group selector - 12 compact circular pills, all on one row (no scroll)
    groups_list = list("ABCDEFGHIJKL")
    st.markdown('<div class="grp-radio">', unsafe_allow_html=True)
    sel = st.radio("Group", groups_list,
                   index=groups_list.index(st.session_state.grp),
                   horizontal=True, label_visibility="collapsed",
                   key="grp_radio")
    st.markdown('</div>', unsafe_allow_html=True)
    if sel != st.session_state.grp:
        st.session_state.grp = sel
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    grp = st.session_state.grp
    teams_in_grp = GROUPS.get(grp, [])

    # Compute live standings
    ranked = compute_standings(teams_in_grp, wc_all)
    played_count = sum(s["P"] for _,s in ranked) // 2
    matches_total = 6  # each group has 6 matches
    st.markdown(f'<div class="kw-sub">{played_count}/{matches_total} matches played in Group {grp}</div>',
                unsafe_allow_html=True)

    # Column headers
    st.markdown("""
    <div style="display:flex;padding:0 8px;margin-bottom:4px;">
      <span style="flex:3;font-size:0.6rem;color:#444;font-weight:700;text-transform:uppercase;">TEAM</span>
      <span style="width:22px;text-align:center;font-size:0.6rem;color:#444;font-weight:700;">P</span>
      <span style="width:22px;text-align:center;font-size:0.6rem;color:#444;font-weight:700;">W</span>
      <span style="width:22px;text-align:center;font-size:0.6rem;color:#444;font-weight:700;">D</span>
      <span style="width:22px;text-align:center;font-size:0.6rem;color:#444;font-weight:700;">L</span>
      <span style="width:26px;text-align:center;font-size:0.6rem;color:#444;font-weight:700;">GD</span>
      <span style="width:26px;text-align:center;font-size:0.6rem;color:#00FF7F;font-weight:700;">PTS</span>
    </div>
    """, unsafe_allow_html=True)

    for i, (team, s) in enumerate(ranked):
        border = "qualify" if i < 2 else ("third" if i == 2 else "")
        code = team[:3].upper()
        gd_str = f"+{s['GD']}" if s['GD'] >= 0 else str(s['GD'])
        st.markdown(f"""
        <div class="st-row {border}">
          <div class="st-pos">{i+1}</div>
          <div class="st-flag">{flag(team)}</div>
          <div class="st-name">
            <div class="st-team-code">{code}</div>
            <div class="st-team-full">{team}</div>
          </div>
          <div class="st-stat">{s['P']}</div>
          <div class="st-stat">{s['W']}</div>
          <div class="st-stat">{s['D']}</div>
          <div class="st-stat">{s['L']}</div>
          <div class="st-stat">{gd_str}</div>
          <div class="st-pts">{s['PTS']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="st-legend">
      <div class="st-legend-item"><div class="st-dot" style="background:#00FF7F"></div> Advance to Round of 32</div>
      <div class="st-legend-item"><div class="st-dot" style="background:#F5C518"></div> Third-place contention</div>
    </div>
    """, unsafe_allow_html=True)

    # Results in this group so far
    grp_set = set(teams_in_grp)
    grp_played = wc_all.dropna(subset=["home_score","away_score"])
    grp_played = grp_played[grp_played["home_team"].isin(grp_set) &
                            grp_played["away_team"].isin(grp_set)]
    if len(grp_played):
        st.markdown('<div class="sect-title">Results</div>', unsafe_allow_html=True)
        for r in grp_played.itertuples(index=False):
            d = pd.Timestamp(r.date).strftime("%b %d")
            st.markdown(f"""
            <div class="sl-item">
              <div style="font-size:0.65rem;color:#555;width:48px;">{d}</div>
              <div class="sl-score" style="flex:1;">
                {flag(r.home_team)} {r.home_team}
                <span style="color:#00FF7F;margin:0 6px;">{int(r.home_score)} – {int(r.away_score)}</span>
                {r.away_team} {flag(r.away_team)}
              </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MY PICKS TAB
# ════════════════════════════════════════════════════════════
elif st.session_state.tab == "picks":
    st.markdown("""
    <div class="kw-eyebrow">🏆 World Cup 2026</div>
    <div class="kw-title">FAVOURITES</div>
    <div class="kw-sub">Ranked by live Elo rating — updates as results come in</div>
    """, unsafe_allow_html=True)

    # Top contenders by Elo (manual results already feed these ratings)
    wc_elo = {t:elo.get(t,ELO_S) for g in GROUPS.values() for t in g}
    top_cont = sorted(wc_elo.items(), key=lambda x:x[1], reverse=True)[:12]
    for i,(team,rating) in enumerate(top_cont):
        mult = round(max(1.5, 10000/rating), 1)
        st.markdown(f"""
        <div class="contender-row">
          <div class="cont-pos">{i+1}</div>
          <div class="cont-flag">{flag(team)}</div>
          <div class="cont-name">{team}</div>
          <div class="cont-mult">{mult}x</div>
        </div>""", unsafe_allow_html=True)

# ── Bottom nav ───────────────────────────────────────────────
tab = st.session_state.tab
p_active = "active" if tab in ("predict","predict_detail") else ""
s_active = "active" if tab=="standings" else ""
m_active = "active" if tab=="picks" else ""
st.markdown(f"""
<div class="bottom-nav">
  <a class="nav-link" href="?tab=predict" target="_self">
    <div class="nav-icon {p_active}">⚡</div>
    <div class="nav-lbl {p_active}">PREDICT</div>
  </a>
  <a class="nav-link" href="?tab=standings" target="_self">
    <div class="nav-icon {s_active}">📊</div>
    <div class="nav-lbl {s_active}">STANDINGS</div>
  </a>
  <a class="nav-link" href="?tab=picks" target="_self">
    <div class="nav-icon {m_active}">⭐</div>
    <div class="nav-lbl {m_active}">FAVOURITES</div>
  </a>
</div>
""", unsafe_allow_html=True)
