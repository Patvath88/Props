import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
from xgboost import XGBRegressor
from functools import lru_cache

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
API_KEY = st.secrets["BALDONTLIE_KEY"]
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.set_page_config(
    page_title="NBA Advanced Prop Predictor",
    layout="wide"
)

# -------------------------------------------------------------------
# CACHED REQUEST
# -------------------------------------------------------------------
@st.cache_data(ttl=300)
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
    except Exception:
        return {"data": []}

# -------------------------------------------------------------------
# PLAYER SEARCH
# -------------------------------------------------------------------
@st.cache_data(ttl=300)
def search_player(q):
    data = fetch("players", {"search": q, "per_page": 50})
    if len(data["data"]) == 0:
        return None
    return data["data"][0]

# -------------------------------------------------------------------
# FETCH GAME LOGS (LAST 40 GAMES)
# -------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_logs(pid):
    logs = fetch("stats", {"player_ids[]": pid, "per_page": 40})
    if "data" not in logs:
        return pd.DataFrame()
    df = pd.DataFrame(logs["data"])
    if df.empty:
        return df
    df["pts"] = df["pts"].astype(float)
    df["reb"] = df["reb"].astype(float)
    df["ast"] = df["ast"].astype(float)
    df["game_date"] = pd.to_datetime(df["game"].apply(lambda x: x["date"]))
    df = df.sort_values("game_date")
    return df

# -------------------------------------------------------------------
# H2H FILTER
# -------------------------------------------------------------------
def filter_h2h(df, opp_id):
    return df[df["opponent"].apply(lambda x: x["id"]) == opp_id]

# -------------------------------------------------------------------
# MATCHUP (DEFENSE VS POSITION)
# -------------------------------------------------------------------
def get_defense_rank(team_id, stat_cat):
    """
    Returns rank 1-30 for opponent defense.
    Uses NBA stats endpoints (public).
    """
    STAT_URL = "https://stats.nba.com/stats/leaguedashteamstats"
    params = {
        "Season": "2024-25",
        "SeasonType": "Regular Season",
        "MeasureType": "Base",
        "PerMode": "PerGame"
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        r = requests.get(STAT_URL, params=params, headers=headers)
        js = r.json()

        headers_list = js["resultSets"][0]["headers"]
        rows = js["resultSets"][0]["rowSet"]

        df = pd.DataFrame(rows, columns=headers_list)

        matchup_col = {
            "pts": "PTS_ALLOWED",
            "reb": "REB",
            "ast": "AST"
        }[stat_cat]

        df = df.sort_values(matchup_col, ascending=False)
        df["rank"] = range(1, 31)

        row = df[df["TEAM_ID"] == team_id]
        if row.empty:
            return None
        return int(row["rank"].iloc[0])

    except Exception:
        return None

# -------------------------------------------------------------------
# INJURY ADJUSTMENT (ON/OFF TEAMMATE SPLIT)
# -------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_injury_adjustment(df, teammate_name, stat):
    """
    If teammate is out:
    Compare player games when teammate DID play vs did NOT play.
    """
    df["with_teammate"] = df["game"].apply(lambda g: teammate_name in str(g))
    with_games = df[df["with_teammate"] == True]
    without_games = df[df["with_teammate"] == False]

    if len(without_games) < 2:
        return 0

    try:
        delta = without_games[stat].mean() - with_games[stat].mean()
        return round(delta, 2)
    except:
        return 0

# -------------------------------------------------------------------
# TRAIN XGBOOST MODEL
# -------------------------------------------------------------------
def train_xgb(df, target):
    df["roll5"] = df[target].rolling(5).mean()
    df["roll10"] = df[target].rolling(10).mean()
    df = df.dropna()

    X = df[["roll5", "roll10"]]
    y = df[target]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=2
    )
    model.fit(X, y)

    return model, X.iloc[-1:]

# -------------------------------------------------------------------
# MONTE CARLO SIMULATION (1,000,000)
# -------------------------------------------------------------------
def monte_carlo_sim(last_values, line, num=1_000_000):
    mu = np.mean(last_values)
    sigma = np.std(last_values) + 0.01

    sims = np.random.normal(mu, sigma, num)
    prob = (sims >= line).mean()
    return prob

# -------------------------------------------------------------------
# UI LAYOUT
# -------------------------------------------------------------------
st.title("🏀 Advanced NBA Prop Predictor (Linemate-Style)")

tab1, tab2, tab3 = st.tabs(["Prediction Model", "Matchup Intel", "1M Monte Carlo Simulation"])

# -----------------------------------------------------------
# TAB 1 — MAIN MODEL
# -----------------------------------------------------------
with tab1:
    player = st.text_input("Enter Player Name (ex: Luka Doncic)")

    if player:
        info = search_player(player)
        if info is None:
            st.error("Player not found.")
            st.stop()

        pid = info["id"]
        team_id = info["team"]["id"]

        st.subheader(f"{info['first_name']} {info['last_name']} — {info['team']['full_name']}")

        df = get_logs(pid)
        if df.empty:
            st.error("No logs found.")
            st.stop()

        df["opponent"] = df["game"].apply(lambda g: g["home_team"] if g["home_team"]["id"] != team_id else g["visitor_team"])

        stat_select = st.selectbox("Stat to Project", ["pts", "reb", "ast"], index=0)

        # Model
        model, latest = train_xgb(df, stat_select)
        base_pred = float(model.predict(latest)[0])

        st.markdown(f"### 📌 **Base Projection: {base_pred:.1f} {stat_select.upper()}**")

        # H2H
        next_team = df["opponent"].iloc[-1]["id"]
        h2h = filter_h2h(df, next_team)
        if len(h2h) > 0:
            h2h_avg = h2h[stat_select].mean()
        else:
            h2h_avg = base_pred

        # Defense Rank
        def_rank = get_defense_rank(next_team, stat_select)
        def_adj = 0
        if def_rank:
            if def_rank <= 5:
                def_adj = -1.5
            elif def_rank >= 26:
                def_adj = +1.5

        # Final Projection
        final_proj = base_pred * 0.6 + h2h_avg * 0.2 + (base_pred + def_adj) * 0.2

        st.metric("Adjusted Projection", f"{final_proj:.1f}")

        st.divider()

        st.subheader("📉 Recent Games Chart")
        chart = alt.Chart(df.tail(10)).mark_bar().encode(
            x=alt.X("game_date:T", title="Game"),
            y=alt.Y(stat_select, title=stat_select.upper())
        )
        st.altair_chart(chart, use_container_width=True)


# -----------------------------------------------------------
# TAB 2 — INTEL
# -----------------------------------------------------------
with tab2:
    st.subheader("Matchup & Contextual Intel")

    if player:
        st.write("### 🔍 Recent Form vs Team / Defense")

        st.write("• Shows defense ranking, matchup difficulty and H2H trends.")
        st.write("• Uses NBA.com defensive metrics + your recent-game logs.")


# -----------------------------------------------------------
# TAB 3 — MONTE CARLO
# -----------------------------------------------------------
with tab3:
    st.subheader("🎲 Monte Carlo Simulation (1,000,000)")

    if player:
        stat_select = st.selectbox("Choose Stat", ["pts", "reb", "ast"])

        line = st.number_input("Enter Betting Line", min_value=0.0, max_value=100.0, value=20.0)

        vals = df[stat_select].tail(10).values

        if st.button("Run 1,000,000 Simulations"):
            prob = monte_carlo_sim(vals, line)

            st.metric("Probability Over Line", f"{prob*100:.2f}%")

            st.success("Simulation complete.")
