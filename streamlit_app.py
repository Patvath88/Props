import streamlit as st
import requests
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# --- CONFIG ---
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = "https://api.balldontlie.io/v1"

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("model/xgb_points_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- FUNCTIONS ---
def get_player_id(name):
    response = requests.get(f"{BASE_URL}/players", params={"search": name}, headers=HEADERS)
    data = response.json()["data"]
    if not data:
        return None
    return data[0]["id"], data[0]["first_name"] + " " + data[0]["last_name"]

def get_recent_stats(player_id, num_games=10):
    response = requests.get(f"{BASE_URL}/stats", params={
        "player_ids[]": player_id,
        "per_page": num_games,
        "postseason": False
    }, headers=HEADERS)
    data = response.json()["data"]
    if not data:
        return None
    return pd.DataFrame(data)

def engineer_features(df):
    # Add rolling averages as features (simple for now)
    return pd.DataFrame({
        "min": [df["min"].astype(float).mean()],
        "fgm": [df["fgm"].mean()],
        "fga": [df["fga"].mean()],
        "fg3m": [df["fg3m"].mean()],
        "ast": [df["ast"].mean()],
        "reb": [df["reb"].mean()],
        "stl": [df["stl"].mean()],
        "blk": [df["blk"].mean()],
        "turnover": [df["turnover"].mean()]
    })

# --- STREAMLIT APP ---
st.set_page_config(page_title="NBA Player Prop Predictor", layout="centered")

st.title("🏀 NBA Player Prop Predictor")
st.markdown("Use the tool below to predict a player's performance and get betting suggestions.")

player_name = st.text_input("Enter NBA Player Name (e.g., LeBron James)")
prop_type = st.selectbox("Choose Prop Type", ["Points", "Assists", "Rebounds"])
line = st.number_input(f"Input Betting Line for {prop_type}", min_value=0.0, max_value=100.0, step=0.5)

if st.button("Predict"):
    with st.spinner("Fetching data & predicting..."):
        player_id, full_name = get_player_id(player_name)
        if not player_id:
            st.error("Player not found.")
        else:
            stats_df = get_recent_stats(player_id)
            if stats_df is None or stats_df.empty:
                st.error("No recent games found for this player.")
            else:
                features = engineer_features(stats_df)

                prediction = model.predict(features)[0]

                st.subheader(f"🔮 Prediction for {full_name}")
                st.metric(label=f"Expected {prop_type}", value=f"{prediction:.1f}")
                st.metric(label=f"Your Line", value=line)

                verdict = "OVER ✅" if prediction > line else "UNDER ❌"
                st.success(f"Suggested Bet: **{verdict}**")

                # Plot recent stat trend
                prop_column = {
                    "Points": "pts",
                    "Assists": "ast",
                    "Rebounds": "reb"
                }[prop_type]

                fig, ax = plt.subplots()
                ax.plot(stats_df["game"]["date"], stats_df[prop_column], marker='o')
                ax.set_title(f"{full_name} - Last {len(stats_df)} Games ({prop_type})")
                ax.set_ylabel(prop_type)
                ax.set_xlabel("Game Date")
                plt.xticks(rotation=45)
                st.pyplot(fig)
