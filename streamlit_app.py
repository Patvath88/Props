# file: streamlit_app.py
from __future__ import annotations

import importlib
import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Feature/ML modules (assumed present)
from features import (
    build_supervised_rows,
    normalize_stats_rows,
    rolling_features,
    attach_season_averages,
)
from modeling import TrainedModel, predict_one, train_regressor

# -------------------------
# Import/Client Resolver
# -------------------------

class BalldontlieError(Exception):
    """HTTP or client error from BallDontLie API."""

def _expand_params(params: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not params:
        return out
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, list):
            key = f"{k}[]" if not k.endswith("[]") else k
            for vv in v:
                if vv is None:
                    continue
                out.append((key, str(vv)))
        else:
            out.append((k, str(v)))
    return out

class _InlineBalldontlieClient:
    """
    Minimal inline fallback client so Streamlit never crashes if your SDK import breaks.
    Implements only the endpoints used by this app.
    """
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.balldontlie.io",
        timeout: float = 20.0,
    ) -> None:
        if not api_key:
            raise ValueError("API key required")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        # API expects raw key in Authorization (no 'Bearer ')
        self.s.headers.update({"Authorization": api_key, "Accept": "application/json"})

    # --- low-level ---
    def _request(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = self.s.get(url, params=_expand_params(params), timeout=self.timeout)
        except requests.RequestException as e:
            raise BalldontlieError(str(e)) from e
        if not (200 <= r.status_code < 300):
            raise BalldontlieError(f"{r.status_code} {url} -> {r.text[:200]}")
        try:
            return r.json()
        except ValueError as e:
            raise BalldontlieError("Invalid JSON") from e

    def _paginate(self, path: str, params: Optional[Dict[str, Any]] = None, per_page: int = 25, cursor: Optional[Union[int,str]] = None):
        q: Dict[str, Any] = dict(params or {})
        q["per_page"] = per_page
        if cursor is not None:
            q["cursor"] = cursor
        while True:
            page = self._request(path, params=q)
            yield page
            meta = page.get("meta") or {}
            nxt = meta.get("next_cursor")
            if not nxt:
                break
            q["cursor"] = nxt

    # --- endpoints used by app ---
    def games_list(self, *, dates: Optional[List[str]] = None, per_page: int = 25, cursor: Optional[Union[int,str]] = None, **_) :
        params: Dict[str, Any] = {}
        if dates: params["dates"] = dates
        return self._paginate("/v1/games", params=params, per_page=per_page, cursor=cursor)

    def odds_list(self, *, dates: Optional[List[str]] = None, game_ids: Optional[List[int]] = None, per_page: int = 25, cursor: Optional[Union[int,str]] = None):
        params: Dict[str, Any] = {}
        if dates: params["dates"] = dates
        if game_ids: params["game_ids"] = game_ids
        return self._paginate("/v2/odds", params=params, per_page=per_page, cursor=cursor)

    def injuries_list(self, *, per_page: int = 25, cursor: Optional[Union[int,str]] = None, **_):
        return self._paginate("/v1/player_injuries", params={}, per_page=per_page, cursor=cursor)

    def leaders_get(self, *, season: int, stat_type: str):
        return self._request("/v1/leaders", params={"season": season, "stat_type": stat_type})

    def standings_get(self, *, season: int):
        return self._request("/v1/standings", params={"season": season})

    def players_list_active(self, *, search: Optional[str] = None, per_page: int = 25, cursor: Optional[Union[int,str]] = None, **_):
        params: Dict[str, Any] = {}
        if search: params["search"] = search
        return self._paginate("/v1/players/active", params=params, per_page=per_page, cursor=cursor)

    def stats_list(self, *, player_ids: Optional[List[int]] = None, per_page: int = 25, cursor: Optional[Union[int,str]] = None, **_):
        params: Dict[str, Any] = {}
        if player_ids: params["player_ids"] = player_ids
        return self._paginate("/v1/stats", params=params, per_page=per_page, cursor=cursor)

    def season_averages(self, *, category: str, season: int, season_type: str, type: str, player_ids: Optional[List[int]] = None):
        params: Dict[str, Any] = {"season": season, "season_type": season_type, "type": type}
        if player_ids: params["player_ids"] = player_ids
        # NOTE: some docs show /v1/season_averages/{category}
        return self._request(f"/v1/season_averages/{category}", params=params)

def _resolve_bdl_client_class() -> Tuple[Type[Any], str]:
    """
    Try multiple class names/modules. On failure, return inline fallback.
    Returns (ClientClass, info_message).
    """
    candidates = [
        ("balldontlie_client", "BalldontlieClient"),
        ("balldontlie_client", "BallDontLieClient"),
        ("balldontlie", "BalldontlieClient"),
        ("balldontlie", "Client"),
    ]
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            return cls, f"Using {mod_name}.{cls_name}"
        except Exception:
            continue
    return _InlineBalldontlieClient, "Using inline fallback client (could not import your SDK)."

# Optional mock
try:
    from mock_client import MockBalldontlieClient  # type: ignore
except Exception:  # pragma: no cover
    MockBalldontlieClient = None  # type: ignore

# -------------------------
# Streamlit Config
# -------------------------

st.set_page_config(page_title="NBA Player Prop Prediction Dashboard", layout="wide")

def _get_api_key() -> str:
    try:
        key = st.secrets.get("BALLDONTLIE_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = None
    return key or os.getenv("BALLDONTLIE_API_KEY", "")

def _use_mock() -> bool:
    try:
        if st.secrets.get("MOCK", False):  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    return os.getenv("BALD_MOCK", "0") == "1"

@st.cache_resource(show_spinner=False)
def get_client():
    if _use_mock() and MockBalldontlieClient is not None:
        return MockBalldontlieClient()
    ClientClass, _msg = _resolve_bdl_client_class()
    api_key = _get_api_key()
    if not api_key:
        st.stop()
    return ClientClass(api_key)

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

# Mode / key guards
mock_mode = _use_mock()
api_key_present = bool(_get_api_key())
if not api_key_present and not mock_mode:
    st.info("Add API key to `st.secrets['BALLDONTLIE_API_KEY']` or set `BALD_MOCK=1` for demo.")
    st.stop()

with st.sidebar:
    # Show which client got resolved
    if mock_mode and MockBalldontlieClient is not None:
        st.write("**Mode:** Mock")
    else:
        _, msg = _resolve_bdl_client_class()
        st.write(f"**Mode:** Live")
        st.caption(msg)

    today = date.today()
    pick_date = st.date_input(
        "Slate date",
        today,
        min_value=today - timedelta(days=60),
        max_value=today + timedelta(days=60),
    )
    default_season = today.year if today.month >= 8 else today.year - 1
    season = st.number_input("Season", min_value=2000, max_value=2100, value=default_season, step=1)
    stat_type = st.selectbox("Leaders stat", ["pts", "ast", "reb", "stl", "blk", "tov", "dreb", "oreb", "min"])
    vendors_filter = st.multiselect(
        "Filter odds vendors",
        ["betmgm", "fanduel", "draftkings", "bet365", "caesars", "espnbet"],
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
            show = gdf[
                ["id", "date", "status", "home_team", "visitor_team", "home_team_score", "visitor_team_score"]
            ].copy()
            show["home_team"] = show["home_team"].apply(lambda t: t.get("abbreviation") if isinstance(t, dict) else t)
            show["visitor_team"] = show["visitor_team"].apply(lambda t: t.get("abbreviation") if isinstance(t, dict) else t)
            st.dataframe(show, use_container_width=True)
    except Exception as e:
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
                    st.session_state["model"] = tr
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
