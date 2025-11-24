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
    page_title="NBA AI Prop Predictor",
    layout="centered"
)

# -------------------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------------------
def fetch(endpoint, params=None):
    """Generic requester with error handling."""
    try:
        r = requests.get(
            f"{BASE_URL}/{endpoint}",
            params=params,
            headers=HEADERS,
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return {"data": []}


def get_player_id(name):
    """Search player using BDL v2 requirements."""
    params = {
        "search": name,
        "per_page": 100,
        "page": 1
    }
    
    data = fetch("players", params)

    if "data" in data and len(data["data"]) > 0:
        player = data["data"][0]
        return player["id"], player

    return None, None


def get_recent_stats(player_id, games=20):
    """Pull player stat logs."""
    stats = fetch(
        "stats",
        {
            "player_ids[]": player_id,
            "per_page": games
        }
    )
    if "data" not in stats or len(stats["data"]) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(stats["data"])
    return df


def prepare_features(df):
    """Create rolling averages for modeling."""
    df['pts_roll'] = df['pts'].rolling(5).mean()
    df['ast_roll'] = df['ast'].rolling(5).mean()
    df['reb_roll'] = df['reb'].rolling(5).mean()

    df = df.dropna()
    return df


def train_model(X, y):
    """Train an XGBoost model."""
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
# UI
# -------------------------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'>🏀 NBA AI Predictor</h1>
<p style='text-align:center;'>Real-Time XGBoost Model Powered by BallDontLie</p>
""", unsafe_allow_html=True)

player_name = st.text_input("Enter NBA Player Name", "")

if player_name:
    with st.spinner("Searching for player..."):
        player_id, player_info = get_player_id(player_name)

    if not player_id:
        st.error("❌ Player not found. Try full names like 'LeBron James'.")
        st.stop()

    st.subheader(f"📌 {player_info['first_name']} {player_info['last_name']}")

    with st.spinner("Fetching recent performance..."):
        df = get_recent_stats(player_id, 30)

    if df.empty:
        st.error("No recent stats available.")
        st.stop()

    # Keep only needed stats fields
    df = df[['pts', 'ast', 'reb']].copy()
    df = prepare_features(df)

    if df.empty:
        st.error("Not enough games for feature engineering.")
        st.stop()

    # Training Data
    X = df[['pts_roll', 'ast_roll', 'reb_roll']]

    pts_model = train_model(X, df["pts"])
    ast_model = train_model(X, df["ast"])
    reb_model = train_model(X, df["reb"])

    latest = X.iloc[-1:].copy()

    # Predictions
    pred_pts = float(pts_model.predict(latest)[0])
    pred_ast = float(ast_model.predict(latest)[0])
    pred_reb = float(reb_model.predict(latest)[0])

    st.markdown("## 🔮 Predicted Stats for Next Game")
    c1, c2, c3 = st.columns(3)
    c1.metric("Points", f"{pred_pts:.1f}")
    c2.metric("Assists", f"{pred_ast:.1f}")
    c3.metric("Rebounds", f"{pred_reb:.1f}")

    st.markdown("---")
    st.markdown("### 📈 Recent Trends (Last 10 Games)")
    st.line_chart(df[['pts', 'ast', 'reb']].tail(10))

    st.markdown("### 📊 Model Input Features")
    st.dataframe(latest)
