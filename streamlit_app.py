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

st.set_page_config(
    page_title="NBA AI Predictor",
    layout="centered",
)

st.markdown("""
<h1 style='text-align:center;'>🏀 NBA AI Prop Predictor</h1>
<p style='text-align:center;'>Using XGBoost + BallDontLie Real-Time Data</p>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def fetch(endpoint, params=None):
    r = requests.get(f"{BASE_URL}/{endpoint}", params=params, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def get_player_id(name):
    params = {
        "search": name,
        "per_page": 100,   # REQUIRED or BDL returns empty
        "page": 1
    }
    
    data = fetch("players", params)

    if "data" in data and len(data["data"]) > 0:
        player = data["data"][0]
        return player["id"], player
    
    return None, None


def get_recent_stats(player_id, games=15):
    stats = fetch("stats", {"player_ids[]": player_id, "per_page": games})
    df = pd.DataFrame(stats["data"])
    return df


def prepare_features(df):
    # Rolling features
    df['pts_roll'] = df['pts'].rolling(5).mean()
    df['ast_roll'] = df['ast'].rolling(5).mean()
    df['reb_roll'] = df['reb'].rolling(5).mean()

    df = df.dropna()
    return df


def train_model(X, y):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror"
    )
    model.fit(X, y)
    return model


# -------------------------------------------------------------------
# UI — SELECT PLAYER
# -------------------------------------------------------------------
player_name = st.text_input("Enter NBA Player Name", "")

if player_name:

    with st.spinner("Searching player..."):
        player_id, player_info = get_player_id(player_name)

    if not player_id:
        st.error("Player not found.")
        st.stop()

    st.subheader(f"{player_info['first_name']} {player_info['last_name']}")

    # -------------------------------------------------------------------
    # FETCH PLAYER GAME LOGS
    # -------------------------------------------------------------------
    with st.spinner("Fetching recent games..."):
        df = get_recent_stats(player_id, 25)

    if df.empty:
        st.error("No player stats available.")
        st.stop()

    # Clean data
    df = df[["pts", "ast", "reb"]]
    df = prepare_features(df)

    # -------------------------------------------------------------------
    # TRAIN MODELS
    # -------------------------------------------------------------------
    X = df[["pts_roll", "ast_roll", "reb_roll"]]

    pts_model = train_model(X, df["pts"])
    ast_model = train_model(X, df["ast"])
    reb_model = train_model(X, df["reb"])

    latest = X.iloc[-1:].copy()

    # -------------------------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------------------------
    pred_pts = float(pts_model.predict(latest)[0])
    pred_ast = float(ast_model.predict(latest)[0])
    pred_reb = float(reb_model.predict(latest)[0])

    # -------------------------------------------------------------------
    # DISPLAY
    # -------------------------------------------------------------------
    st.markdown("## 📊 Predicted Next Game Stats")

    col1, col2, col3 = st.columns(3)
    col1.metric("Points", f"{pred_pts:.1f}")
    col2.metric("Assists", f"{pred_ast:.1f}")
    col3.metric("Rebounds", f"{pred_reb:.1f}")

    st.markdown("---")
    st.markdown("### 📈 Recent Form (Last 10 Games)")
    st.line_chart(df[["pts", "ast", "reb"]].tail(10))

    st.markdown("### 🔍 Feature Inputs")
    st.dataframe(latest)
