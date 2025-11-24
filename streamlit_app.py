import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor

# ============================================================
# CONFIG
# ============================================================

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # your BallDontLie key
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.set_page_config(page_title="NBA Luxe AI Predictor", layout="wide")

# ============================================================
# THEME
# ============================================================

LUXE_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at top, #3b0066 0, #050009 40%, #020006 100%) !important;
    color: #f5f2ff !important;
}
h1, h2, h3, h4 {
    color: #f8f0ff !important;
    font-weight: 700 !important;
}
.luxe-card {
    background: rgba(8, 1, 20, 0.9);
    border-radius: 18px;
    padding: 1.2rem;
    border: 1px solid rgba(155, 89, 182, 0.8);
    box-shadow: 0 0 30px rgba(155, 89, 182, 0.45);
}
.soft-card {
    background: rgba(15, 8, 30, 0.9);
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid rgba(155, 89, 182, 0.4);
}
.stMetricValue {
    color: #fff !important;
    font-weight: 800 !important;
}
</style>
"""
st.markdown(LUXE_CSS, unsafe_allow_html=True)

# ============================================================
# STAT SETTINGS
# ============================================================

STAT_COLUMNS = ["pts", "reb", "ast", "fg3m"]
DEFAULT_MONTE_CARLO_SIMS = 5000

# ============================================================
# HELPERS
# ============================================================

@st.cache_data(ttl=300)
def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return {"data": []}

@st.cache_data(ttl=600)
def search_players(query):
    if len(query) < 2:
        return []
    return api_get("players", {"search": query, "per_page": 25}).get("data", [])

@st.cache_data(ttl=600)
def get_player_stats(player_id, n_games=40):
    data = api_get("stats", {"player_ids[]": player_id, "per_page": n_games}).get("data", [])
    if not data:
        return pd.DataFrame()
    return pd.json_normalize(data)

def get_headshot(player_id):
    return f"https://balldontlie.io/images/headshots/{player_id}.png"

def preprocess(df):
    keep = {}
    for stat in STAT_COLUMNS:
        cols = [c for c in df.columns if c.split(".")[-1] == stat]
        if cols:
            keep[stat] = pd.to_numeric(df[cols[0]], errors="coerce")

    df2 = pd.DataFrame(keep).dropna(how="all")
    if df2.empty:
        return df2

    for s in STAT_COLUMNS:
        if s in df2:
            df2[f"{s}_roll"] = df2[s].rolling(5).mean()
            df2[f"{s}_prev"] = df2[s].shift(1)

    df2 = df2.dropna()
    return df2

# ============================================================
# MODEL TRAINING
# ============================================================

def train_models(df):
    models = {}
    stds = {}

    features = [c for c in df.columns if ("roll" in c or "prev" in c)]
    if not features:
        return {}, {}

    X = df[features].values

    for stat in STAT_COLUMNS:
        if stat not in df:
            continue
        y = df[stat].values
        if len(y) < 8:
            continue

        model = XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.9,
            colsample_bytree=0.9, objective="reg:squarederror"
        )
        model.fit(X, y)

        preds = model.predict(X)
        stds[stat] = float(np.std(y - preds))
        models[stat] = (model, features)

    return models, stds

# ============================================================
# PREDICTION + MONTE CARLO
# ============================================================

def predict_next(models, stds, row):
    preds = {}
    pred_stds = {}

    for stat, (model, features) in models.items():
        x = np.array([row[features].values], float)
        mu = float(model.predict(x)[0])
        preds[stat] = max(0, mu)
        pred_stds[stat] = max(0.5, stds.get(stat, 1.0))

    return preds, pred_stds

def run_monte(preds, stds):
    sims = {}
    for stat in preds:
        sims[stat] = np.clip(
            np.random.normal(preds[stat], stds[stat], DEFAULT_MONTE_CARLO_SIMS),
            0, None
        )
    return pd.DataFrame(sims)

# ============================================================
# EDGE + EV
# ============================================================

def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def compute_ev(prob, odds):
    payout = odds/100 if odds > 0 else 100/(-odds)
    return prob * payout - (1 - prob)

# ============================================================
# UI
# ============================================================

def main():

    st.markdown("<h1 style='text-align:center;'>🏀 NBA Luxe AI Predictor</h1>", unsafe_allow_html=True)

    # PLAYER SEARCH
    query = st.text_input("Search Player", "")
    players = search_players(query) if len(query) >= 2 else []

    if players:
        names = [f"{p['first_name']} {p['last_name']} - {p['team']['abbreviation']}" for p in players]
        idx = st.selectbox("Select Player", range(len(names)), format_func=lambda x: names[x])
        player = players[idx]
    else:
        st.info("Type at least 2 letters to search.")
        return

    # IMAGE + INFO
    colA, colB = st.columns([1,3])
    with colA:
        st.image(get_headshot(player["id"]), use_column_width=True)
    with colB:
        st.markdown(f"### {player['first_name']} {player['last_name']}")
        st.markdown(f"Team: **{player['team']['full_name']}**")

    # LOAD STATS
    df_raw = get_player_stats(player["id"])
    if df_raw.empty:
        st.error("No stats available.")
        return

    df = preprocess(df_raw)
    if df.empty:
        st.error("Not enough data after preprocessing.")
        return

    models, stds = train_models(df)
    last_row = df.iloc[-1]

    preds, pred_stds = predict_next(models, stds, last_row)
    sims = run_monte(preds, pred_stds)

    # METRICS
    st.markdown("## 🔮 Core Projection")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PTS", f"{preds['pts']:.1f}")
    c2.metric("REB", f"{preds['reb']:.1f}")
    c3.metric("AST", f"{preds['ast']:.1f}")
    c4.metric("3PM", f"{preds['fg3m']:.1f}")

    # PROP INPUTS
    st.markdown("## 📊 Prop Lines (Enter Book Numbers)")
    colp1, colp2, colp3, colp4 = st.columns(4)

    line_pts = colp1.number_input("PTS Line", value=float(round(preds["pts"],1)))
    odds_pts = colp1.number_input("Odds (PTS)", value=-110)

    line_reb = colp2.number_input("REB Line", value=float(round(preds["reb"],1)))
    odds_reb = colp2.number_input("Odds (REB)", value=-110)

    line_ast = colp3.number_input("AST Line", value=float(round(preds["ast"],1)))
    odds_ast = colp3.number_input("Odds (AST)", value=-110)

    line_3pm = colp4.number_input("3PM Line", value=float(round(preds["fg3m"],1)))
    odds_3pm = colp4.number_input("Odds (3PM)", value=-110)

    # COMPUTE PROBS
    st.markdown("## 🎲 Monte Carlo Probabilities + EV")

    results = []
    for stat, line, odds in [
        ("pts", line_pts, odds_pts),
        ("reb", line_reb, odds_reb),
        ("ast", line_ast, odds_ast),
        ("fg3m", line_3pm, odds_3pm),
    ]:
        prob_over = np.mean(sims[stat] > line)
        imp = american_to_prob(odds)
        edge = prob_over - imp
        ev = compute_ev(prob_over, odds)

        results.append([stat.upper(), f"{prob_over:.3f}", f"{edge:.3f}", f"{ev:.3f}"])

    st.dataframe(pd.DataFrame(results, columns=["Prop","Prob Over","Edge","EV"]), use_container_width=True)

    # DISTRIBUTION VISUAL
    st.markdown("## 📈 Distribution Viewer")
    chosen = st.selectbox("Select Stat", STAT_COLUMNS)
    st.bar_chart(sims[chosen])

if __name__ == "__main__":
    main()
