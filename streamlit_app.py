import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE_URL = "https://api.balldontlie.io/v1"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

st.set_page_config(page_title="NBA AI Predictor", layout="centered")


# -------------------------------------------------------------------
# CACHED REQUEST FUNCTION (Prevents Rate Limits)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch(endpoint, params=None):
    try:
        r = requests.get(
            f"{BASE_URL}/{endpoint}",
            params=params,
            headers=HEADERS,
            timeout=8
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "data": []}


# -------------------------------------------------------------------
# PLAYER SEARCH (FIXED)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def get_player_id(name):
    params = {"search": name, "per_page": 50, "page": 1}
    data = fetch("players", params)

    if "data" in data and len(data["data"]) > 0:
        p = data["data"][0]
        return p["id"], p

    return None, None


# -------------------------------------------------------------------
# STATS FETCH (FIXED / LAST 30 GAMES)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def get_recent_stats(player_id):
    stats = fetch(
        "stats",
        {"player_ids[]": player_id, "per_page": 30}
    )

    if "data" not in stats or len(stats["data"]) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(stats["data"])
    return df


# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------
def prepare_features(df):
    df["pts_roll"] = df["pts"].rolling(5).mean()
    df["ast_roll"] = df["ast"].rolling(5).mean()
    df["reb_roll"] = df["reb"].rolling(5).mean()
    df = df.dropna()
    return df


# -------------------------------------------------------------------
# TRAIN XGBOOST MODEL
# -------------------------------------------------------------------
def train_model(X, y):
    model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=2
    )
    model.fit(X, y)
    return model


# -------------------------------------------------------------------
# UI LAYOUT
# -------------------------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'>🏀 NBA AI Predictor</h1>
<p style='text-align:center;'>30-Game Rolling XGBoost Model Powered by BallDontLie</p>
""", unsafe_allow_html=True)

player_name = st.text_input("Enter NBA Player Name", "", key="player_input")

if player_name:
    st.write(" ")

    # ---------------------------------------------------------------
    # SEARCH PLAYER
    # ---------------------------------------------------------------
    with st.spinner("Searching player…"):
        player_id, player_info = get_player_id(player_name)

    if not player_id:
        st.error("❌ Player not found. Try full names (e.g., 'Luka Doncic').")
        st.stop()

    st.subheader(f"📌 {player_info['first_name']} {player_info['last_name']}")

    # ---------------------------------------------------------------
    # FETCH STATS
    # ---------------------------------------------------------------
    with st.spinner("Loading last 30 games…"):
        df = get_recent_stats(player_id)

    if df.empty:
        st.error("❌ No recent stats found for this player.")
        st.stop()

    df = df[["pts", "ast", "reb"]].copy()
    df = prepare_features(df)

    if df.empty:
        st.error("❌ Not enough data for model (need 5+ games).")
        st.stop()

    # ML Inputs
    X = df[["pts_roll", "ast_roll", "reb_roll"]]

    # Train 3 lightweight models
    pts_model = train_model(X, df["pts"])
    ast_model = train_model(X, df["ast"])
    reb_model = train_model(X, df["reb"])

    latest = X.iloc[-1:]

    # Predictions
    pred_pts = float(pts_model.predict(latest)[0])
    pred_ast = float(ast_model.predict(latest)[0])
    pred_reb = float(reb_model.predict(latest)[0])

    # ---------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------
    st.markdown("## 🔮 Predicted Next Game Stats")

    c1, c2, c3 = st.columns(3)
    c1.metric("Points", f"{pred_pts:.1f}")
    c2.metric("Assists", f"{pred_ast:.1f}")
    c3.metric("Rebounds", f"{pred_reb:.1f}")

    st.markdown("---")
    st.markdown("### 📈 Last 10 Games Trend")
    st.line_chart(df[["pts", "ast", "reb"]].tail(10))

    st.markdown("### 📊 Model Inputs")
    st.dataframe(latest)
