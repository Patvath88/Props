# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
from datetime import datetime
from xgboost import XGBRegressor

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.set_page_config(page_title="NBA AI Predictor", layout="wide")

# -------------------------------------------------------------------
# GLOBAL CSS (mobile-friendly spacing, stacked columns on small screens)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
      /* tighter top spacing, better mobile tap targets */
      .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px;}
      .stMetric {border-radius: 12px; border: 1px solid rgba(0,0,0,0.08); padding: 0.6rem;}
      /* stack columns under ~800px */
      @media (max-width: 820px) {
        .st-emotion-cache-0 {display: block !important;}
      }
      /* card-like sections */
      .card {border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 1rem; background: white;}
      .subtle {color: #666;}
      .page-title {text-align:center; margin: 0.2rem 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# CACHED REQUEST FUNCTION
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch(endpoint: str, params: dict | None = None) -> dict:
    """HTTP GET to BallDontLie with basic error wrapping."""
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # why: keep UI alive when API blips
        return {"error": str(e), "data": []}

# -------------------------------------------------------------------
# PLAYER SEARCH
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def get_player_id(name: str) -> tuple[int | None, dict | None]:
    params = {"search": name, "per_page": 50, "page": 1}
    data = fetch("players", params)
    if "data" in data and data["data"]:
        p = data["data"][0]
        return p["id"], p
    return None, None

# -------------------------------------------------------------------
# STATS FETCH (LAST 30 GAMES)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def get_recent_stats(player_id: int) -> pd.DataFrame:
    stats = fetch("stats", {"player_ids[]": player_id, "per_page": 30})
    if "data" not in stats or not stats["data"]:
        return pd.DataFrame()
    return pd.DataFrame(stats["data"])

# -------------------------------------------------------------------
# HEADSHOT URL
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def get_headshot_url(player_id: int, first: str, last: str) -> str:
    """
    Prefer BallDontLie CDN headshot by player_id.
    Fallback to initials avatar to avoid broken images.
    """
    primary = f"https://cdn.balldontlie.io/images/headshots/{player_id}.png?width=260"
    try:
        # light check; why: avoid showing broken image boxes
        r = requests.head(primary, timeout=5)
        if r.ok:
            return primary
    except Exception:
        pass
    initials_seed = requests.utils.quote(f"{first} {last}")
    return f"https://api.dicebear.com/7.x/initials/png?seed={initials_seed}&backgroundType=gradientLinear"

# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pts_roll"] = df["pts"].rolling(5).mean()
    df["ast_roll"] = df["ast"].rolling(5).mean()
    df["reb_roll"] = df["reb"].rolling(5).mean()
    return df.dropna()

# -------------------------------------------------------------------
# MODEL
# -------------------------------------------------------------------
def train_model(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=2,
        random_state=42,
    )
    model.fit(X, y)
    return model

# -------------------------------------------------------------------
# FLATTEN STATS
# -------------------------------------------------------------------
def flatten_stats(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Extracts game metadata and ensures types."""
    df = df_raw.copy()
    # Nested fields may be dicts; guard missing keys
    df["game_date"] = pd.to_datetime(df["game"].apply(lambda g: g.get("date") if isinstance(g, dict) else None))
    df["game_id"] = df["game"].apply(lambda g: g.get("id") if isinstance(g, dict) else None)
    df["team_abbr"] = df["team"].apply(lambda t: t.get("abbreviation") if isinstance(t, dict) else None)
    # Ensure numeric
    for col in ["pts", "ast", "reb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    return df

# -------------------------------------------------------------------
# CHARTS
# -------------------------------------------------------------------
def bar_last_10(df: pd.DataFrame) -> alt.Chart:
    """Grouped bars for last 10 games across PTS/AST/REB."""
    df10 = df.tail(10)[["game_date", "pts", "ast", "reb"]].copy()
    melted = df10.melt(id_vars="game_date", var_name="Stat", value_name="Value")
    return (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("yearmonthdate(game_date):T", title="Game"),
            xOffset=alt.XOffset("Stat:N"),
            y=alt.Y("Value:Q", title=""),
            color=alt.Color("Stat:N", legend=alt.Legend(orient="top")),
            tooltip=[
                alt.Tooltip("yearmonthdate(game_date):T", title="Game"),
                alt.Tooltip("Stat:N"),
                alt.Tooltip("Value:Q"),
            ],
        )
        .properties(height=320)
        .interactive()
    )

def bar_rolling_vs_avg(df: pd.DataFrame) -> alt.Chart:
    """5-game rolling avg vs overall avg bars."""
    recent = df.tail(5)[["pts", "ast", "reb"]].mean()
    overall = df[["pts", "ast", "reb"]].mean()
    comp = pd.DataFrame(
        {
            "Stat": ["PTS", "AST", "REB", "PTS", "AST", "REB"],
            "Window": ["Last 5"] * 3 + ["Overall"] * 3,
            "Value": [recent["pts"], recent["ast"], recent["reb"], overall["pts"], overall["ast"], overall["reb"]],
        }
    )
    return (
        alt.Chart(comp)
        .mark_bar()
        .encode(
            x=alt.X("Stat:N", title=""),
            xOffset=alt.XOffset("Window:N"),
            y=alt.Y("Value:Q", title="Average"),
            color=alt.Color("Window:N", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("Window:N"), alt.Tooltip("Stat:N"), alt.Tooltip("Value:Q")],
        )
        .properties(height=320)
        .interactive()
    )

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.markdown("<h1 class='page-title'>🏀 NBA AI Predictor</h1>", unsafe_allow_html=True)
st.caption("30-Game Rolling XGBoost • Data by BallDontLie")

colA, colB = st.columns([3, 1])
with colA:
    player_name = st.text_input("Search player", placeholder="e.g., Luka Doncic", label_visibility="collapsed")
with colB:
    compact = st.toggle("Compact (mobile) mode", value=True)

if player_name:
    with st.spinner("Searching player…"):
        player_id, player_info = get_player_id(player_name)

    if not player_id or not player_info:
        st.error("❌ Player not found. Try full names (e.g., 'Luka Doncic').")
        st.stop()

    with st.spinner("Loading last 30 games…"):
        df_raw = get_recent_stats(player_id)

    if df_raw.empty:
        st.error("❌ No recent stats found for this player.")
        st.stop()

    df = flatten_stats(df_raw)
    if df[["pts", "ast", "reb"]].dropna().shape[0] < 6:
        st.error("❌ Not enough data for model (need 5+ games).")
        st.stop()

    # Headshot + Bio card
    headshot = get_headshot_url(player_id, player_info["first_name"], player_info["last_name"])
    team = player_info.get("team", {}) or {}
    pos = (player_info.get("position") or "").strip() or "—"

    # Averages
    last10 = df.tail(10)[["pts", "ast", "reb"]].mean()
    last30 = df[["pts", "ast", "reb"]].mean()

    # Layout: compact stacks; otherwise two columns
    if compact:
        c_img = st.container()
        with c_img:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(headshot, width=180)
            st.subheader(f"{player_info['first_name']} {player_info['last_name']}")
            st.caption(f"{team.get('full_name','')} • {pos}")
            c1, c2, c3 = st.columns(3)
            c1.metric("PTS (10)", f"{last10['pts']:.1f}")
            c2.metric("AST (10)", f"{last10['ast']:.1f}")
            c3.metric("REB (10)", f"{last10['reb']:.1f}")
            c1, c2, c3 = st.columns(3)
            c1.metric("PTS (30)", f"{last30['pts']:.1f}")
            c2.metric("AST (30)", f"{last30['ast']:.1f}")
            c3.metric("REB (30)", f"{last30['reb']:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        left, right = st.columns([1, 2])
        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(headshot, width=200)
            st.subheader(f"{player_info['first_name']} {player_info['last_name']}")
            st.caption(f"{team.get('full_name','')} • {pos}")
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            r1.metric("PTS (10)", f"{last10['pts']:.1f}")
            r2.metric("AST (10)", f"{last10['ast']:.1f}")
            r3.metric("REB (10)", f"{last10['reb']:.1f}")
            r1, r2, r3 = st.columns(3)
            r1.metric("PTS (30)", f"{last30['pts']:.1f}")
            r2.metric("AST (30)", f"{last30['ast']:.1f}")
            r3.metric("REB (30)", f"{last30['reb']:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)

    # Features + Model
    df_feats = prepare_features(df[["pts", "ast", "reb"]].copy())
    X = df_feats[["pts_roll", "ast_roll", "reb_roll"]]
    pts_model = train_model(X, df_feats["pts"])
    ast_model = train_model(X, df_feats["ast"])
    reb_model = train_model(X, df_feats["reb"])
    latest = X.iloc[[-1]]

    pred_pts = float(pts_model.predict(latest)[0])
    pred_ast = float(ast_model.predict(latest)[0])
    pred_reb = float(reb_model.predict(latest)[0])

    st.markdown("### 🔮 Predicted Next Game")
    if compact:
        m1 = st.container()
        with m1:
            a, b, c = st.columns(3)
            a.metric("Points", f"{pred_pts:.1f}")
            b.metric("Assists", f"{pred_ast:.1f}")
            c.metric("Rebounds", f"{pred_reb:.1f}")
    else:
        a, b, c = st.columns(3)
        a.metric("Points", f"{pred_pts:.1f}")
        b.metric("Assists", f"{pred_ast:.1f}")
        c.metric("Rebounds", f"{pred_reb:.1f}")

    st.markdown("---")

    # Charts: Professional bar graphs
    st.markdown("### 📊 Last 10 Games (Grouped Bars)")
    st.altair_chart(bar_last_10(df), use_container_width=True)

    st.markdown("### 📊 Rolling vs Overall Averages")
    st.altair_chart(bar_rolling_vs_avg(df), use_container_width=True)

    # Data tools
    with st.expander("🔧 Model Inputs (latest row)"):
        st.dataframe(latest, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download last 30 games (CSV)",
        data=csv,
        file_name=f"{player_info['first_name']}_{player_info['last_name']}_last30.csv",
        mime="text/csv",
    )
