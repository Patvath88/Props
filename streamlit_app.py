# file: streamlit_app.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================
# Inline BallDontLie Client
# =========================

class BalldontlieError(Exception):
    pass

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

class BDLClient:
    """Minimal client for endpoints used in this app."""
    def __init__(self, api_key: str, base_url: str = "https://api.balldontlie.io", timeout: float = 20.0) -> None:
        if not api_key:
            raise ValueError("API key required (BALLDONTLIE_API_KEY)")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        # Why: API expects raw key, no 'Bearer '
        self.s.headers.update({"Authorization": api_key, "Accept": "application/json"})

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
            nxt = (page.get("meta") or {}).get("next_cursor")
            if not nxt:
                break
            q["cursor"] = nxt

    # --- endpoints used ---
    def games_list(self, *, dates: Optional[List[str]] = None, per_page: int = 25, cursor: Optional[Union[int,str]] = None, **_):
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
        return self._request(f"/v1/season_averages/{category}", params=params)

# ===================
# Inline Feature Layer
# ===================

STAT_COLS = ["pts","reb","ast","blk","stl","fg3a","fg3m","fga","fgm","fta","ftm","turnover","min"]

def _parse_min(val: Any) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    if ":" in s:
        try:
            mm, ss = s.split(":")
            return float(int(mm) + int(ss) / 60.0)
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_stats_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    flat: List[Dict[str, Any]] = []
    for r in rows:
        g = r.get("game", {})
        team = r.get("team", {})
        out = {k: r.get(k) for k in STAT_COLS}
        out.update({
            "game_id": g.get("id"),
            "game_date": g.get("date"),
            "season": g.get("season"),
            "postseason": g.get("postseason"),
            "home_team_id": g.get("home_team_id"),
            "visitor_team_id": g.get("visitor_team_id"),
            "team_id": team.get("id"),
            "player_id": r.get("player", {}).get("id", r.get("player_id")),
        })
        flat.append(out)
    df = pd.DataFrame(flat)
    if "min" in df:
        df["min"] = df["min"].apply(_parse_min)
    return df

def rolling_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["player_id", "game_date"]).copy()
    feats: Dict[str, pd.Series] = {}
    for col in ["pts","reb","ast","min","fga","fta","fg3a","fgm","fg3m","ftm","turnover"]:
        if col in df:
            feats[f"{col}_r{window}"] = (
                df.groupby("player_id")[col].transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
            )
    out = pd.concat([df, pd.DataFrame(feats)], axis=1)
    need = [c for c in out.columns if c.endswith(f"_r{window}")]
    out = out.dropna(subset=need)
    return out

def attach_season_averages(out: pd.DataFrame, season_avgs: pd.DataFrame) -> pd.DataFrame:
    if season_avgs.empty or out.empty:
        return out
    sa = season_avgs.add_prefix("sa_")
    sa = sa.rename(columns={"sa_player_id": "player_id"})
    return out.merge(sa, on="player_id", how="left")

def build_supervised_rows(
    df: pd.DataFrame,
    window: int,
    target: str,
    season_avgs: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    feat_df = rolling_features(df, window)
    if season_avgs is not None:
        feat_df = attach_season_averages(feat_df, season_avgs)
    y = feat_df[target].astype(float)
    # Drop raw stat columns from features
    drop_stats = [c for c in ["pts","reb","ast"] if c in feat_df]
    X = feat_df.drop(columns=drop_stats)
    keep_ids = ["player_id","game_id","game_date","team_id","season","postseason"]
    id_cols = [c for c in keep_ids if c in feat_df]
    X = pd.concat([X.drop(columns=[c for c in id_cols if c in X]), feat_df[id_cols]], axis=1)
    return X, y

# ===================
# Inline Modeling
# ===================

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    _HAS_SK = True
except Exception:
    _HAS_SK = False

try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

@dataclass
class TrainedModel:
    algo: str
    feature_cols: List[str]
    target: str
    window: int
    metrics: Dict[str, float]
    # store either sklearn Pipeline or simple tuple for fallback
    model: Any

def _train_ridge_numpy(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Very small ridge fallback with standardization."""
    Xn = X.fillna(X.median(numeric_only=True)).to_numpy(dtype=float)
    mu = Xn.mean(axis=0)
    sd = Xn.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (Xn - mu) / sd
    lam = 2.0
    w = np.linalg.pinv(Xs.T @ Xs + lam * np.eye(Xs.shape[1])) @ Xs.T @ y.to_numpy(dtype=float)
    return w, np.vstack([mu, sd])

def _predict_ridge_numpy(model_tuple: Tuple[np.ndarray, np.ndarray], Xrow: pd.DataFrame) -> float:
    w, stats = model_tuple
    mu, sd = stats
    x = Xrow.fillna(Xrow.median(numeric_only=True)).to_numpy(dtype=float)
    x = (x - mu) / sd
    return float(x @ w)[0]

def train_regressor(X: pd.DataFrame, y: pd.Series, *, algo: str = "ridge", random_state: int = 42) -> TrainedModel:
    feat_cols = [c for c in X.columns if c not in {"player_id","game_id","game_date","team_id","season","postseason"}]
    if algo == "xgb" and _HAS_XGB and _HAS_SK:
        reg = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=random_state, n_jobs=0, reg_lambda=1.0
        )
        pre = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), feat_cols)], remainder="drop")
        pipe = Pipeline([("pre", pre), ("reg", reg)])
        pipe.fit(X[feat_cols], y)
        yhat = pipe.predict(X[feat_cols])
        mae = float(np.mean(np.abs(yhat - y.to_numpy())))
        return TrainedModel(algo="xgb", feature_cols=feat_cols, target=y.name or "pts", window=-1, metrics={"mae_in_sample": mae}, model=pipe)

    if _HAS_SK:
        pre = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), feat_cols)], remainder="drop")
        pipe = Pipeline([("pre", pre), ("reg", Ridge(alpha=2.0, random_state=random_state))])
        pipe.fit(X[feat_cols], y)
        yhat = pipe.predict(X[feat_cols])
        mae = float(np.mean(np.abs(yhat - y.to_numpy())))
        return TrainedModel(algo="ridge", feature_cols=feat_cols, target=y.name or "pts", window=-1, metrics={"mae_in_sample": mae}, model=pipe)

    # NumPy fallback
    w, stats = _train_ridge_numpy(X[feat_cols], y)
    # in-sample MAE
    mae = float(np.mean(np.abs([_predict_ridge_numpy((w, stats), X[feat_cols].iloc[[i]]) - y.iloc[i] for i in range(len(y))])))
    return TrainedModel(algo="ridge_np", feature_cols=feat_cols, target=y.name or "pts", window=-1, metrics={"mae_in_sample": mae}, model=(w, stats))

def predict_one(tr: TrainedModel, row: pd.DataFrame) -> float:
    missing = sorted(set(tr.feature_cols) - set(row.columns))
    if missing:
        raise ValueError(f"Missing features: {missing[:8]}")
    if isinstance(tr.model, tuple):
        return _predict_ridge_numpy(tr.model, row[tr.feature_cols].iloc[[0]])
    else:
        return float(tr.model.predict(row[tr.feature_cols])[0])

# =========================
# Streamlit: Config/Helpers
# =========================

st.set_page_config(page_title="NBA Player Prop Prediction Dashboard", layout="wide")

def _get_api_key() -> str:
    try:
        key = st.secrets.get("BALLDONTLIE_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = None
    return key or os.getenv("BALLDONTLIE_API_KEY", "")

@st.cache_resource(show_spinner=False)
def get_client() -> BDLClient:
    return BDLClient(_get_api_key())

# ---------------
# Cached loaders
# ---------------

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

# ----------------------
# Training cache wrapper
# ----------------------

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

# =========
# The UI
# =========

st.title("NBA Player Prop Prediction Dashboard")

api_key_present = bool(_get_api_key())
if not api_key_present:
    st.info("Set `BALLDONTLIE_API_KEY` in environment or Streamlit secrets.")
    st.stop()

with st.sidebar:
    today = date.today()
    pick_date = st.date_input("Slate date", today, min_value=today - timedelta(days=60), max_value=today + timedelta(days=60))
    default_season = today.year if today.month >= 8 else today.year - 1
    season = st.number_input("Season", min_value=2000, max_value=2100, value=default_season, step=1)
    stat_type = st.selectbox("Leaders stat", ["pts","ast","reb","stl","blk","tov","dreb","oreb","min"])
    vendors_filter = st.multiselect("Filter odds vendors", ["betmgm","fanduel","draftkings","bet365","caesars","espnbet"])

tabs = st.tabs(["Games", "Odds", "Injuries", "Leaders", "Standings", "Props (ML)"])

# Games
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
    except Exception as e:
        st.error(str(e))

# Odds
with tabs[1]:
    d = pick_date.strftime("%Y-%m-%d")
    try:
        odf = df_odds(d)
        if vendors_filter and "vendor" in odf.columns:
            odf = odf[odf["vendor"].isin(vendors_filter)]
        st.dataframe(odf, use_container_width=True)
    except Exception as e:
        st.error(str(e))

# Injuries
with tabs[2]:
    try:
        st.dataframe(df_injuries(), use_container_width=True)
    except Exception as e:
        st.error(str(e))

# Leaders
with tabs[3]:
    try:
        st.dataframe(df_leaders(int(season), stat_type), use_container_width=True)
    except Exception as e:
        st.error(str(e))

# Standings
with tabs[4]:
    try:
        st.dataframe(df_standings(int(season)), use_container_width=True)
    except Exception as e:
        st.error(str(e))

# Props (ML)
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
                lambda r: f"{r.get('first_name')} {r.get('last_name')} ({r.get('team',{}).get('abbreviation','?')})", axis=1
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

                        # Compare to odds
                        d = pick_date.strftime("%Y-%m-%d")
                        odf = df_odds(d)
                        if not odf.empty and "player_name" in odf.columns:
                            ply_name = candidates.iloc[idx]["display"].split(" (")[0]
                            mask = odf["player_name"].astype(str).str.contains(ply_name, case=False, na=False)
                            lines = odf.loc[mask]
                            if "market" in lines.columns:
                                mkt = {"pts":"points","reb":"rebounds","ast":"assists"}.get(target, target)
                                lines = lines[lines["market"].str.contains(mkt, case=False, na=False)]
                            if not lines.empty:
                                st.write("Odds lines:")
                                st.dataframe(lines, use_container_width=True)
                        st.caption("Baseline demo; replace with your own feature set/model for production.")
    else:
        st.info("Search a player to get started.")
