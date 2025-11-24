# file: streamlit_app.py
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st

# Why: keep app decoupled from transport; allows live API or offline mock
from balldontlie_client import BalldontlieClient, BalldontlieError
from mock_client import MockBalldontlieClient
from features import build_supervised_rows, normalize_stats_rows, rolling_features, attach_season_averages
from modeling import TrainedModel, predict_one, train_regressor


# -------------------------
# Config & Client Selection
# -------------------------
st.set_page_config(page_title="NBA Player Prop Prediction Dashboard", layout="wide")

def _get_api_key() -> str:
    # Why: Streamlit Cloud prefers st.secrets; local dev uses env
    try:
        key = st.secrets.get("BALLDONTLIE_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = None
    return key or os.getenv("BALLDONTLIE_API_KEY", "")

def _use_mock() -> bool:
    # Why: ensure a working demo without network/API key
    try:
        if st.secrets.get("MOCK", False):  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    return os.getenv("BALD_MOCK", "0") == "1"

@st.cache_resource(show_spinner=False)
def get_client():
    return MockBalldontlieClient() if _use_mock() else BalldontlieClient(_get_api_key())


# -------------------------
# Cached Data Accessors
# -------------------------
@st.cache_data(show_spinner=False, ttl=300)
def df_games(d: str) -> pd.DataFrame:
    api = get_client()
    rows: List[Dict[str, Any]] = []
    for p in api.games_list(dates=[d], per_page=100):
        rows.extend(p.get("data", []))
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=300)
def df_odds(d: str) -> pd.DataFrame:
    api = get_client()
    rows: List[Dict[str, Any]] = []
    for p in api.odds_list(dates=[d], per_page=100):
        rows.extend(p.get("data", []))
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=600)
def df_injuries() -> pd.DataFrame:
    api = get_client()
    rows: List[Dict[str, Any]] = []
    for p in api.injuries_list(per_page=100):
        rows.extend(p.get("data", []))
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=600)
def df_leaders(season: int, stat_type: str) -> pd.DataFrame:
    api = get_client()
    return pd.DataFrame(api.leaders_get(season=season, stat_type=stat_type).get("data", []))

@st.cache_data(show_spinner=False, ttl=600)
def df_standings(season: int) -> pd.DataFrame:
    api = get_client()
    return pd.DataFrame(api.standings_get(season=season).get("data", []))

@st.cache_data(show_spinner=False, ttl=120)
def df_search_active(q: str) -> pd.DataFrame:
    api = get_client()
    rows: List[Dict[str, Any]] = []
    for p in api.players_list_active(search=q, per_page=50):
        rows.extend(p.get("data", []))
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=300)
def df_player_stats(player_id: int, max_pages: int = 6) -> pd.DataFrame:
    # Why: limit pages to keep latency reasonable in Streamlit
    api = get_client()
    rows: List[Dict[str, Any]] = []
    pages = api.stats_list(player_ids=[player_id], per_page=100)
    for i, page in enumerate(pages):
        rows.extend(page.get("data", []))
        if i + 1 >= max_pages:
            break
    return normalize_stats_rows(rows)

@st.cache_data(show_spinner=False, ttl=300)
def df_season_averages(player_ids: List[int], season: int) -> pd.DataFrame:
    api = get_client()
    data = api.season_averages(
        category="general", season=season, season_type="regular", type="base", player_ids=player_ids
    )
    rows = data.get("data", data if isinstance(data, dict) else [])
    df = pd.DataFrame(rows)
    if "player_id" not in df and "player" in df:
        df["player_id"] = df["player"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
    return df


# -------------------------
# ML Training Cache
# -------------------------
@st.cache_resource(show_spinner=False)
def train_player_model(player_id: int, season: int, target: str, window: int, algo: str) -> TrainedModel:
    stats = df_player_stats(player_id)
    if stats.empty:
        raise RuntimeError("No stats available to train.")
    sa = df_season_averages([player_id], season)
    X, y = build_supervised_rows(stats, window=window, target=target, season_avgs=sa)
    if X.empty:
        raise RuntimeError("Not enough history for rolling-window features.")
    model = train_regressor(X, y, algo=algo)
    model.window = window
    return model


# -------------------------
# UI
# -------------------------
st.title("NBA Player Prop Prediction Dashboard")

mock_mode = _use_mock()
api_key_present = bool(_get_api_key())
if not api_key_present and not mock_mode:
    st.info("Add API key to `st.secrets['BALLDONTLIE_API_KEY']` or set `BALD_MOCK=1` for demo mode.")
    st.stop()

with st.sidebar:
    st.write("**Mode**:", "Mock" if mock_mode else "Live")
    today = date.today()
    pick_date = st.date_input("Slate date", today, min_value=today - timedelta(days=60), max_value=today + timedelta(days=60))
    # Why: NBA season counts by starting fall; adjust default accordingly
    default_season = today.year if today.month >= 8 else today.year - 1
    season = st.number_input("Season", min_value=2000, max_value=2100, value=default_season, step=1)
    stat_type = st.selectbox("Leaders stat", ["pts","ast","reb","stl","blk","tov","dreb","oreb","min"])
    vendors_filter = st.multiselect(
        "Filter odds vendors",
        ["betmgm","fanduel","draftkings","bet365","caesars","espnbet"]
    )

tabs = st.tabs(["Games", "Odds", "Injuries", "Leaders", "Standings", "Props (ML)"])

# --- Games ---
with tabs[0]:
    d = pick_date.strftime("%Y-%m-%d")
    try:
        gdf = df_games(d)
        if gdf.empty:
            st.write("No games.")
        else:
            show = gdf[["id","date","status","home_team","visitor_team","home_team_score","visitor_team_score"]].copy()
            show["home_team"] = show["home_team"].apply(lambda t: t.get("abbreviation") if isinstance(t, dict) else t)
            show["visitor_team"] = show["visitor_team"].apply(lambda t: t.get("abbreviation") if isinstance(t, dict) else t)
            st.dataframe(show, use_container_width=True)
    except BalldontlieError as e:
        st.error(str(e))

# --- Odds ---
with tabs[1]:
    d = pick_date.strftime("%Y-%m-%d")
    try:
        odf = df_odds(d)
        if vendors_filter and "vendor" in odf.columns:
            odf = odf[odf["vendor"].isin(vendors_filter)]
        st.dataframe(odf, use_container_width=True)
    except Exception as e:
        st.error(str(e))

# --- Injuries ---
with tabs[2]:
    try:
        st.dataframe(df_injuries(), use_container_width=True)
    except Exception as e:
        st.error(str(e))

# --- Leaders ---
with tabs[3]:
    try:
        st.dataframe(df_leaders(int(season), stat_type), use_container_width=True)
    except Exception as e:
        st.error(str(e))

# --- Standings ---
with tabs[4]:
    try:
        st.dataframe(df_standings(int(season)), use_container_width=True)
    except Exception as e:
        st.error(str(e))

# --- Props (ML) ---
with tabs[5]:
    st.subheader("Train & Predict")
    q = st.text_input("Search active player")
    algo = st.selectbox("Algorithm", ["ridge", "xgb"])
    window = st.slider("Rolling window (games)", 3, 15, 7)
    target = st.selectbox("Target", ["pts", "reb", "ast"])

    if q:
        candidates = df_search_active(q)
        if candidates.empty:
            st.info("No active player matched.")
        else:
            candidates["display"] = candidates.apply(
                lambda r: f"{r.get('first_name')} {r.get('last_name')} ({r.get('team',{}).get('abbreviation','?')})",
                axis=1,
            )
            idx = st.selectbox("Choose player", range(len(candidates)), format_func=lambda i: candidates.iloc[i]["display"])
            player_id = int(candidates.iloc[idx]["id"])
            st.caption(f"Player ID: {player_id}")

            if st.button("Train / Refresh model"):
                try:
                    tr = train_player_model(player_id, int(season), target, window, algo)
                    st.session_state["model"] = tr  # Why: keep current trained model per session
                    st.success(f"Trained {tr.algo} | In-sample MAE: {tr.metrics['mae_in_sample']:.2f}")
                except Exception as e:
                    st.error(str(e))

            tr: Optional[TrainedModel] = st.session_state.get("model")  # type: ignore[assignment]
            if tr:
                stats = df_player_stats(player_id)
                if stats.empty:
                    st.warning("No stats for prediction.")
                else:
                    feats = rolling_features(stats, window=tr.window)
                    sa = df_season_averages([player_id], int(season))
                    feats = attach_season_averages(feats, sa)
                    if feats.empty:
                        st.warning("Not enough history for prediction.")
                    else:
                        latest = feats.sort_values("game_date").tail(1)
                        try:
                            pred_val = predict_one(tr, latest)
                            st.metric(f"Predicted {target.upper()}", f"{pred_val:.1f}")
                        except Exception as e:
                            st.error(str(e))

                        # Compare with odds lines if present
                        d = pick_date.strftime("%Y-%m-%d")
                        odf = df_odds(d)
                        if not odf.empty and "player_name" in odf.columns:
                            ply_name = candidates.iloc[idx]["display"].split(" (")[0]
                            mask = odf["player_name"].astype(str).str.contains(ply_name, case=False, na=False)
                            lines = odf.loc[mask]
                            market_map = {"pts": "points", "reb": "rebounds", "ast": "assists"}
                            mkt = market_map.get(target, target)
                            if "market" in lines.columns:
                                lines = lines[lines["market"].str.contains(mkt, case=False, na=False)]
                            if not lines.empty:
                                st.write("Odds lines:")
                                st.dataframe(lines, use_container_width=True)
                        st.caption("Baseline demo; replace with your own feature set/model for production.")
            else:
                st.info("Train a model to see predictions.")
    else:
        st.info("Search a player to get started.")
