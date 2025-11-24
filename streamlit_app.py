import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from bs4 import BeautifulSoup

# ============================================================
# CONFIG
# ============================================================

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY, "X-API-KEY": API_KEY}

st.set_page_config(page_title="NBA Luxe Predictor", layout="centered")

# ============================================================
# MOBILE LUXE THEME
# ============================================================

LUXE = """
<style>
.stApp {
    background: #07010f;
    color: #f5ecff;
}

.big-card {
    background: rgba(25, 10, 50, 0.85);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid #8f47ff;
    box-shadow: 0 0 25px rgba(143,71,255,0.35);
}

.metric-card {
    background: rgba(50, 20, 90, 0.9);
    padding: 16px;
    border-radius: 16px;
    border: 1px solid #a574ff;
    text-align: center;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 34px;
    font-weight: 800;
    color: white;
}

.metric-label {
    font-size: 14px;
    color: #d1c3ff;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    grid-gap: 12px;
}
</style>
"""
st.markdown(LUXE, unsafe_allow_html=True)


# ============================================================
# API HELPERS
# ============================================================

def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=8)
        return r.json() if r.status_code == 200 else {"data": []}
    except:
        return {"data": []}


@st.cache_data(ttl=300)
def search_players(q):
    if len(q) < 2: return []
    return api_get("players", {"search": q, "per_page": 15}).get("data", [])


@st.cache_data(ttl=300)
def get_player_stats(pid):
    d = api_get("stats", {"player_ids[]": pid, "per_page": 40}).get("data", [])
    return pd.json_normalize(d) if d else pd.DataFrame()


@st.cache_data(ttl=300)
def get_next_opponent(team_id):
    games = api_get("games", {"team_ids[]": team_id, "per_page": 50}).get("data", [])
    if not games: return None
    # Most recent future game (sorted)
    upcoming = sorted(games, key=lambda g: g["date"])
    for g in upcoming:
        if pd.to_datetime(g["date"]) >= pd.Timestamp.now():
            # if home team is our team, opponent is visitor
            if g["home_team"]["id"] == team_id:
                return g["visitor_team"]["abbreviation"]
            else:
                return g["home_team"]["abbreviation"]
    return None


# ============================================================
# DEFENSIVE RATING SCRAPER (Basketball-Reference)
# ============================================================

@st.cache_data(ttl=3600)
def load_def_ratings():
    url = "https://www.basketball-reference.com/leagues/NBA_2025.html"
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": "misc_stats"})
    rows = table.find("tbody").find_all("tr")

    def_map = {}
    for row in rows:
        cells = row.find_all("td")
        if not cells: continue
        team = cells[0].text.strip()
        drtg = cells[13].text.strip()  # Def Rtg column index
        try:
            drtg = float(drtg)
            abbr = team.split()[-1][:3].upper()
            def_map[abbr] = drtg
        except:
            continue

    league_avg = np.mean(list(def_map.values()))
    return def_map, league_avg


# ============================================================
# MODEL PIPELINE
# ============================================================

def prep(df):
    keep = {}
    for s in ["pts","reb","ast","fg3m"]:
        col = [c for c in df.columns if c.endswith(f".{s}")]
        if col: keep[s] = pd.to_numeric(df[col[0]], errors="coerce")

    df2 = pd.DataFrame(keep).dropna()
    for s in keep:
        df2[f"{s}_roll"] = df2[s].rolling(5).mean()
        df2[f"{s}_prev"] = df2[s].shift(1)

    return df2.dropna()


def train_models(df):
    feats = [c for c in df.columns if "roll" in c or "prev" in c]
    models, stds = {}, {}
    X = df[feats].values
    for s in ["pts","reb","ast","fg3m"]:
        if s not in df: continue
        y = df[s].values
        if len(y) < 8: continue
        m = XGBRegressor(
            n_estimators=220, learning_rate=0.05,
            max_depth=3, subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror"
        ).fit(X,y)
        stds[s] = float(np.std(y - m.predict(X)))
        models[s] = (m, feats)
    return models, stds


def predict_next(models, stds, row):
    preds, pred_std = {}, {}
    for s,(model,feats) in models.items():
        X = np.array([row[feats].values], float)
        mu = float(model.predict(X)[0])
        preds[s] = max(0, mu)
        pred_std[s] = max(0.5, stds.get(s,1.0))
    return preds, pred_std


def monte(preds, pred_stds, n=3000):
    sims = {}
    for s in preds:
        sims[s] = np.clip(
            np.random.normal(preds[s], pred_stds[s], n), 0, None
        )
    return pd.DataFrame(sims)


# ============================================================
# UI
# ============================================================

st.markdown("<h2 style='text-align:center;'>🏀 NBA Luxe Predictor</h2>", unsafe_allow_html=True)

query = st.text_input("Search Player", "")
players = search_players(query)

if not players:
    st.info("Type at least 2 letters…")
    st.stop()

names = [f"{p['first_name']} {p['last_name']} ({p['team']['abbreviation']})" for p in players]
idx = st.selectbox("Select Player", range(len(names)), format_func=lambda i: names[i])
player = players[idx]

st.markdown("<div class='big-card'>", unsafe_allow_html=True)
st.image(f"https://balldontlie.io/images/headshots/{player['id']}.png", width=220)
st.markdown(f"<h3>{player['first_name']} {player['last_name']}</h3>", unsafe_allow_html=True)

team_id = player["team"]["id"]
opp_abbr = get_next_opponent(team_id)
if not opp_abbr:
    st.error("Could not detect next opponent.")
    st.stop()

def_rtg, league_avg = load_def_ratings()
opp_drtg = def_rtg.get(opp_abbr, league_avg)

st.markdown(f"**Next Opponent:** {opp_abbr})
**Opponent Defensive Rating:** `{opp_drtg}`  
**League Avg DRTG:** `{league_avg:.1f}`")

# LOAD STATS + MODELING
stats = get_player_stats(player["id"])
df = prep(stats)

if df.empty:
    st.error("Not enough data.")
    st.stop()

models, stds = train_models(df)
latest = df.iloc[-1]

base_preds, base_stds = predict_next(models, stds, latest)

# APPLY DEF ADJUSTMENT
adj_preds = {k: base_preds[k] * (league_avg / opp_drtg) for k in base_preds}

# RUN MONTE USING ADJUSTED MEANS
sims = monte(adj_preds, base_stds)

# METRICS DISPLAY
st.markdown("### 📊 Projections (Base vs Adjusted)")

for stat in ["pts","reb","ast","fg3m"]:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{stat.upper()}</div>
            <div class='metric-value'>Base: {base_preds[stat]:.1f}</div>
            <div class='metric-value'>Adj: {adj_preds[stat]:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# PROP CHECKER
st.subheader("🎲 Prop Probability Checker (Adjusted)")

st_stat = st.selectbox("Stat", ["pts","reb","ast","fg3m"])
line = st.number_input("Line", value=float(round(adj_preds[st_stat],1)))
prob = np.mean(sims[st_stat] > line)

st.markdown(
    f"<h3 style='text-align:center;'>Prob Over: {prob:.1%}</h3>",
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
