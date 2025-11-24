
# Mobile Luxe NBA Predictor - Full Rebuild
import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor

st.set_page_config(page_title="NBA Luxe Mobile Predictor", layout="centered")

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY, "X-API-KEY": API_KEY}

LUXE = """
<style>
.stApp { background: #07010f; color: #f3eaff; }
.big-card {
    background: rgba(25,10,50,0.85);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #8f47ff;
    box-shadow: 0 0 20px rgba(143,71,255,0.4);
}
.metric-card {
    background: rgba(50,20,90,0.9);
    padding: 12px;
    border-radius: 14px;
    border: 1px solid #a574ff;
    text-align: center;
}
.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
}
.metric-label {
    font-size: 14px;
    color: #d4c7ff;
}
</style>
"""
st.markdown(LUXE, unsafe_allow_html=True)

def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=8)
        if r.status_code != 200:
            return {"data": []}
        return r.json()
    except:
        return {"data": []}

@st.cache_data(ttl=300)
def search_players(q):
    if len(q) < 2:
        return []
    return api_get("players", {"search": q, "per_page": 10}).get("data", [])

@st.cache_data(ttl=300)
def get_stats(pid):
    data = api_get("stats", {"player_ids[]": pid, "per_page": 40}).get("data", [])
    if not data:
        return pd.DataFrame()
    return pd.json_normalize(data)

def prep(df):
    cols = {}
    for s in ["pts","reb","ast","fg3m"]:
        matches = [c for c in df.columns if c.endswith(f".{s}")]
        if matches:
            cols[s] = pd.to_numeric(df[matches[0]], errors="coerce")
    df2 = pd.DataFrame(cols).dropna()
    for s in cols:
        df2[f"{s}_roll"] = df2[s].rolling(5).mean()
        df2[f"{s}_prev"] = df2[s].shift(1)
    return df2.dropna()

def train(df):
    feats = [c for c in df.columns if "roll" in c or "prev" in c]
    models, stds = {}, {}
    X = df[feats].values
    for s in ["pts","reb","ast","fg3m"]:
        if s not in df:
            continue
        y = df[s].values
        if len(y) < 8:
            continue
        model = XGBRegressor(
            n_estimators=220, learning_rate=0.05, max_depth=3,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror"
        )
        model.fit(X,y)
        preds = model.predict(X)
        stds[s] = float(np.std(y - preds))
        models[s] = (model, feats)
    return models, stds

def predict(models, stds, row):
    preds, pstd = {}, {}
    for s,(m,feats) in models.items():
        X = np.array([row[feats].values], float)
        mu = float(m.predict(X)[0])
        preds[s] = max(0, mu)
        pstd[s] = max(0.5, stds.get(s, 1.0))
    return preds, pstd

def monte(preds, stds, n=4000):
    sims = {}
    for s in preds:
        sims[s] = np.clip(np.random.normal(preds[s], stds[s], n), 0, None)
    return pd.DataFrame(sims)

st.markdown("<h2 style='text-align:center;'>ð NBA Luxe Mobile Predictor</h2>", unsafe_allow_html=True)

query = st.text_input("Search Player", "")
players = search_players(query)

if not players:
    st.info("Type at least 2 lettersâ¦")
else:
    names = [f"{p['first_name']} {p['last_name']} ({p['team']['abbreviation']})" for p in players]
    idx = st.selectbox("Select Player", range(len(names)), format_func=lambda i: names[i])
    p = players[idx]

    st.markdown("<div class='big-card'>", unsafe_allow_html=True)
    st.image(f"https://balldontlie.io/images/headshots/{p['id']}.png", width=200)
    st.markdown(f"<h3>{p['first_name']} {p['last_name']}</h3>", unsafe_allow_html=True)

    stats = get_stats(p["id"])
    df = prep(stats)

    if df.empty:
        st.error("Not enough data to model.")
    else:
        models, stds = train(df)
        last = df.iloc[-1]
        preds, pstd = predict(models, stds, last)
        sims = monte(preds, pstd)

        col1, col2 = st.columns(2)
        col1.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{preds['pts']:.1f}</div>
            <div class='metric-label'>Points</div>
        </div>""", unsafe_allow_html=True)

        col2.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{preds['reb']:.1f}</div>
            <div class='metric-label'>Rebounds</div>
        </div>""", unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        col3.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{preds['ast']:.1f}</div>
            <div class='metric-label'>Assists</div>
        </div>""", unsafe_allow_html=True)

        col4.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{preds['fg3m']:.1f}</div>
            <div class='metric-label'>3PM</div>
        </div>""", unsafe_allow_html=True)

        st.subheader("Prop Probability Checker")
        stat = st.selectbox("Choose Stat", ["pts","reb","ast","fg3m"])
        line = st.number_input("Line", value=float(round(preds[stat],1)))
        prob = float(np.mean(sims[stat] > line))
        st.markdown(f"<h3 style='text-align:center;'>ð Probability Over: {prob:.1%}</h3>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
