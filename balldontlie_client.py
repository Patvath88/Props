# balldontlie_client.py
"""
BALLDONTLIE client with base-URL auto-detection.
Probes: /v1 → /v2 → legacy /api/v1. Keeps Bearer auth for paid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


@dataclass
class APIConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # if provided, validate; else auto-detect


KNOWN_BASES = (
    "https://api.balldontlie.io/v1",     # Pro/current
    "https://api.balldontlie.io/v2",     # Pro/alt
    "https://www.balldontlie.io/api/v1", # Legacy free
)


class BallDontLieClient:
    def __init__(self, cfg: APIConfig):
        self.api_key = (cfg.api_key or "").strip() or None
        self._explicit_base = (cfg.base_url or "").rstrip("/") or None
        self.base_url: Optional[str] = None  # resolved lazily
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    # ---- public helpers ----
    def search_players(self, search: str) -> List[Dict]:
        return self._paged("players", {"search": search}, per_page=50, max_pages=2)

    def player_stats_by_seasons(self, player_id: int, seasons: List[int]) -> List[Dict]:
        rows: List[Dict] = []
        for s in seasons:
            rows.extend(self._paged("stats", {"player_ids[]": player_id, "seasons[]": s}))
        return rows

    def next_team_game(self, team_id: int, start_date) -> Optional[Dict]:
        data = self._get("games", {"team_ids[]": team_id, "start_date": str(start_date), "per_page": 1, "page": 1})
        items = data.get("data", [])
        return items[0] if items else None

    # ---- core HTTP ----
    def _paged(self, path: str, params: Dict, per_page: int = 100, max_pages: int = 50) -> List[Dict]:
        out: List[Dict] = []
        page = 1
        while page <= max_pages:
            p = dict(params)
            p.update({"page": page, "per_page": per_page})
            data = self._get(path, p)
            items = data.get("data", [])
            out.extend(items)
            meta = data.get("meta", {}) or {}
            if not meta.get("next_page") or len(items) < per_page:
                break
            page += 1
        return out

    def _get(self, path: str, params: Dict) -> Dict:
        if self.base_url is None:
            self.base_url = self._resolve_base_or_raise()
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(5):
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 404 and attempt == 0:
                # why: user-provided base may be wrong or API moved; re-probe once
                self.base_url = self._resolve_base_or_raise(force=True)
                url = f"{self.base_url}/{path.lstrip('/')}"
                continue
            if resp.status_code == 429 and attempt < 4:
                time.sleep(1.0 + 0.5 * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Exhausted retries calling BALLDONTLIE.")

    # ---- base resolution ----
    def _resolve_base_or_raise(self, force: bool = False) -> str:
        if self._explicit_base and not force:
            self._assert_route_exists(self._explicit_base)
            return self._explicit_base.rstrip("/")

        tried: List[Tuple[str, int]] = []
        for base in ([self._explicit_base] if self._explicit_base else []) + list(KNOWN_BASES):
            if not base:
                continue
            try:
                code = self._probe_players_endpoint(base)
                if code in (200, 401):  # 401 is fine → route exists
                    return base.rstrip("/")
                tried.append((base, code))
            except requests.RequestException:
                tried.append((base, -1))
        detail = ", ".join([f"{b}→{c}" for b, c in tried]) or "no bases tried"
        raise RuntimeError(
            "No working BALLDONTLIE base URL found. "
            f"Tried: {detail}. 404 → wrong path/domain. "
            "Use https://api.balldontlie.io/v1 (Pro) or set an explicit base."
        )

    def _probe_players_endpoint(self, base: str) -> int:
        url = f"{base.rstrip('/')}/players"
        resp = self.session.get(url, params={"per_page": 1, "search": "a"}, timeout=15)
        return resp.status_code

    def _assert_route_exists(self, base: str) -> None:
        code = self._probe_players_endpoint(base)
        if code not in (200, 401):
            raise RuntimeError(
                f"Provided base invalid for /players: {base} → HTTP {code}."
            )


# app.py
"""
Streamlit NBA Next-Game Predictor (BALLDONTLIE, auto-base)
Run:
  streamlit run app.py
Env:
  export BALLDONTLIE_API_KEY="YOUR_PRO_KEY"
"""

from __future__ import annotations

import os
import math
import time
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional XGBoost
try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# 🔁 New client
from balldontlie_client import APIConfig, BallDontLieClient

st.set_page_config(page_title="NBA Next-Game Predictor", layout="wide")
TODAY = dt.date.today()
DEFAULT_SEASONS_BACK = 3
PER_PAGE = 100

def _to_minutes(min_str: Optional[str]) -> float:
    if not min_str or min_str in {"0", "00", "00:00"}:
        return 0.0
    try:
        parts = str(min_str).split(":")
        if len(parts) == 1:
            return float(parts[0])
        m, s = int(parts[0]), int(parts[1])
        return m + s / 60.0
    except Exception:
        return 0.0

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _safe_int(x: Optional[int]) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None

# ---- cached API calls (auto-base) ----
@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_search_players(cfg: APIConfig, search: str) -> List[Dict]:
    return BallDontLieClient(cfg).search_players(search)

@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_player_stats(cfg: APIConfig, player_id: int, seasons: Tuple[int, ...]) -> pd.DataFrame:
    client = BallDontLieClient(cfg)
    raw = client.player_stats_by_seasons(player_id, list(seasons))
    if not raw:
        return pd.DataFrame()
    df = pd.json_normalize(raw)
    keep = {
        "id": "stat_id",
        "game.id": "game_id",
        "game.date": "game_date",
        "game.season": "season",
        "game.home_team.id": "home_team_id",
        "game.visitor_team.id": "visitor_team_id",
        "team.id": "team_id",
        "player.id": "player_id",
        "pts": "PTS",
        "reb": "REB",
        "ast": "AST",
        "fg3m": "FG3M",
        "min": "MIN_STR",
        "stl": "STL",
        "blk": "BLK",
        "turnover": "TOV",
        "pf": "PF",
        "fga": "FGA",
        "fgm": "FGM",
        "fta": "FTA",
        "ftm": "FTM",
    }
    df = df[list(keep.keys())].rename(columns=keep)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_convert(None)
    df["MIN"] = df["MIN_STR"].map(_to_minutes).astype(float)
    df = df.sort_values("game_date").reset_index(drop=True)
    df["is_home"] = (df["team_id"] == df["home_team_id"]).astype(int)
    df["opponent_id"] = np.where(df["is_home"] == 1, df["visitor_team_id"], df["home_team_id"])
    df["rest_days"] = df["game_date"].diff().dt.days.fillna(2).clip(lower=0).astype(int)
    return df

@st.cache_data(show_spinner=False, ttl=60 * 10)
def cached_next_game(cfg: APIConfig, team_id: int, from_date: dt.date) -> Optional[Dict]:
    return BallDontLieClient(cfg).next_team_game(team_id, from_date)

TARGETS = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST", "3PM": "FG3M"}
ROLL_WINDOWS = [3, 5, 10]

def build_features(df: pd.DataFrame, target_col: str):
    work = df.copy()
    work[f"{target_col}_lag1"] = work[target_col].shift(1)
    for w in ROLL_WINDOWS:
        for col in ["PTS", "REB", "AST", "FG3M", "MIN"]:
            work[f"{col}_roll{w}"] = work[col].rolling(w, min_periods=1).mean().shift(1)
    work["vs_opp_mean"] = (
        work.groupby("opponent_id")[target_col]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    work["overall_mean10"] = work[target_col].rolling(10, min_periods=1).mean().shift(1)
    work["vs_opp_mean"] = work["vs_opp_mean"].fillna(work["overall_mean10"])
    work["vs_opp_mean"] = work["vs_opp_mean"].fillna(work[f"{target_col}_lag1"])
    feat_cols = [
        f"{target_col}_lag1", "is_home", "rest_days", "vs_opp_mean",
    ] + [f"{c}_roll{w}" for w in ROLL_WINDOWS for c in ["PTS", "REB", "AST", "FG3M", "MIN"]]
    X = work[feat_cols].copy()
    y = work[target_col].astype(float)
    valid = ~X.isna().any(axis=1) & y.notna()
    return X[valid], y[valid], feat_cols

def next_game_features(df: pd.DataFrame, target_col: str, next_game: Optional[Dict]):
    last = df.iloc[-1:].copy()
    if next_game:
        next_gdate = pd.to_datetime(next_game["date"]).tz_convert(None)
        rest_days = int(max(0, (next_gdate.date() - last["game_date"].iloc[0].date()).days))
        home_team_id = next_game["home_team"]["id"]
        visitor_team_id = next_game["visitor_team"]["id"]
        team_id = int(last["team_id"].iloc[0])
        is_home = 1 if team_id == home_team_id else 0
        opponent_id = visitor_team_id if is_home == 1 else home_team_id
    else:
        rest_days = int(last["rest_days"].iloc[0])
        is_home = int(last["is_home"].iloc[0])
        opponent_id = int(last["opponent_id"].iloc[0])

    tmp = df.copy()
    tmp[f"{target_col}_lag1"] = tmp[target_col].shift(1)
    for w in ROLL_WINDOWS:
        for col in ["PTS", "REB", "AST", "FG3M", "MIN"]:
            tmp[f"{col}_roll{w}"] = tmp[col].rolling(w, min_periods=1).mean().shift(1)

    hist_vs = (
        tmp[tmp["opponent_id"] == opponent_id][target_col]
        .shift(1).expanding().mean().dropna()
    )
    vs_opp_mean = float(hist_vs.iloc[-1]) if len(hist_vs) else float(tmp[target_col].rolling(10, min_periods=1).mean().shift(1).iloc[-1])

    row = {
        f"{target_col}_lag1": float(tmp[f"{target_col}_lag1"].iloc[-1]),
        "is_home": is_home,
        "rest_days": rest_days,
        "vs_opp_mean": vs_opp_mean,
    }
    for w in ROLL_WINDOWS:
        for col in ["PTS", "REB", "AST", "FG3M", "MIN"]:
            row[f"{col}_roll{w}"] = float(tmp[f"{col}_roll{w}"].iloc[-1])
    meta = {"opponent_id": opponent_id, "is_home": is_home, "rest_days": rest_days}
    return pd.DataFrame([row]), meta

@dataclass
class ModelResult:
    name: str
    model: object
    oof_pred: np.ndarray
    mae: float
    rmse: float

def make_base_models(enable_xgb: bool):
    models: Dict[str, object] = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=42))]),
        "RF": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, min_samples_leaf=2),
        "HGB": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42),
    }
    if enable_xgb and _HAS_XGB:
        models["XGB"] = XGBRegressor(
            n_estimators=350, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=-1, tree_method="hist",
        )
    return models

def timeseries_oof_and_fit(X, y, models: Dict[str, object], n_splits: int = 5):
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X) // 8)))
    results: List[ModelResult] = []
    oof_matrix = np.zeros((len(X), len(models)), dtype=float)
    for m_idx, (name, mdl) in enumerate(models.items()):
        oof = np.zeros(len(X), dtype=float)
        for tr, va in tscv.split(X):
            mdl.fit(X.iloc[tr], y.iloc[tr])
            oof[va] = mdl.predict(X.iloc[va])
        mae = mean_absolute_error(y, oof)
        rmse = _rmse(y.values, oof)
        results.append(ModelResult(name, models[name], oof, mae, rmse))
        oof_matrix[:, m_idx] = oof
    blender = LinearRegression(positive=True)
    blender.fit(oof_matrix, y.values)
    blended_oof = blender.predict(oof_matrix)
    blended_rmse = _rmse(y.values, blended_oof)
    return results, blended_oof, blender

def refit_all(models: Dict[str, object], blender: LinearRegression, X, y):
    fitted: Dict[str, object] = {}
    for name, mdl in models.items():
        mdl.fit(X, y)
        fitted[name] = mdl
    return fitted, blender

def predict_with_ensemble(fitted: Dict[str, object], blender: LinearRegression, X_new):
    base_preds = []
    indiv: Dict[str, float] = {}
    for name, mdl in fitted.items():
        p = float(np.squeeze(mdl.predict(X_new)))
        base_preds.append(p)
        indiv[name] = p
    pred = float(np.squeeze(blender.predict(np.array(base_preds).reshape(1, -1))))
    return pred, indiv

def sidebar_controls():
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("BALLDONTLIE API Key", value=os.getenv("BALLDONTLIE_API_KEY", ""), type="password")
    seasons_back = st.sidebar.slider("Seasons to include (back from current)", 1, 6, DEFAULT_SEASONS_BACK)
    last_n_games = st.sidebar.slider("Train on last N games (per player)", 10, 200, 80)
    target_label = st.sidebar.selectbox("Target stat", list(TARGETS.keys()), index=0)
    enable_xgb = st.sidebar.checkbox("Enable XGBoost (if installed)", value=False)
    n_splits = st.sidebar.slider("CV splits (TimeSeriesSplit)", 3, 8, 5)
    run = st.sidebar.button("Train & Predict")
    cfg = APIConfig(api_key=api_key or None)  # base auto-detected
    # Connection probe (why: immediate feedback if key/base invalid)
    with st.sidebar.expander("Connection status", expanded=False):
        try:
            base_used = BallDontLieClient(cfg)._resolve_base_or_raise()
            st.success(f"OK — using {base_used}")
        except Exception as e:
            st.error(str(e))
    if enable_xgb and not _HAS_XGB:
        st.sidebar.info("xgboost not installed. pip install xgboost to enable.")
    return cfg, {
        "seasons_back": seasons_back,
        "last_n_games": last_n_games,
        "target_label": target_label,
        "target_col": TARGETS[target_label],
        "enable_xgb": enable_xgb and _HAS_XGB,
        "n_splits": n_splits,
        "run": run,
    }

def player_picker(cfg: APIConfig) -> Optional[Dict]:
    st.subheader("Player")
    q = st.text_input("Search player", placeholder="e.g., LeBron James")
    if not q:
        return None
    try:
        results = cached_search_players(cfg, q)
    except Exception as e:
        st.error(f"Player search failed: {e}")
        return None
    if not results:
        st.warning("No players found.")
        return None
    options = []
    for r in results[:50]:
        team = r.get("team") or {}
        label = f"{r.get('first_name','')} {r.get('last_name','')} — {team.get('full_name','FA')} (id:{r.get('id')})"
        options.append((label, r))
    choice = st.selectbox("Select a player", options, format_func=lambda x: x[0])
    st.caption("Tip: refine the search text to reduce duplicates.")
    return choice[1] if choice else None

def show_recent_table(df: pd.DataFrame, target_col: str, n: int = 12):
    cols = ["game_date", "is_home", "opponent_id", "MIN", "PTS", "REB", "AST", "FG3M"]
    st.write("Recent games")
    st.dataframe(df[cols].tail(n).rename(columns={"is_home": "HOME"}).reset_index(drop=True), use_container_width=True)

def show_metrics_table(rows: List[ModelResult], blended_rmse: float):
    data = [{"Model": r.name, "MAE": round(r.mae, 3), "RMSE": round(r.rmse, 3)} for r in rows]
    data.append({"Model": "Ensemble", "MAE": "-", "RMSE": round(blended_rmse, 3)})
    st.write("Cross-validated metrics (OOF)")
    st.dataframe(pd.DataFrame(data), use_container_width=True)

def main():
    st.title("🏀 NBA Next-Game Predictor — BALLDONTLIE (Auto-Base)")
    cfg, params = sidebar_controls()
    st.markdown("- Enter your paid BALLDONTLIE key in the sidebar.  \n- Choose player, seasons, and target.")
    player = player_picker(cfg)
    if not player:
        st.stop()
    player_id = int(player["id"])
    team = player.get("team") or {}
    team_id = _safe_int(team.get("id"))
    st.info(f"Selected: **{player.get('first_name')} {player.get('last_name')}** — {team.get('full_name','FA')}")
    current_season = TODAY.year if TODAY.month >= 10 else TODAY.year - 1
    seasons = tuple(range(current_season - params["seasons_back"] + 1, current_season + 1))
    with st.spinner("Fetching game logs..."):
        try:
            df = cached_player_stats(cfg, player_id, seasons)
        except Exception as e:
            st.error(f"Stats fetch failed: {e}")
            st.stop()
    if df.empty or len(df) < 15:
        st.error("Not enough data retrieved. Try more seasons or a different player.")
        st.stop()
    df = df.tail(params["last_n_games"]).reset_index(drop=True)
    target_col = params["target_col"]
    X, y, feat_cols = build_features(df, target_col)
    if len(X) < 20:
        st.warning("Very small training set after feature creation; results may be noisy.")
    next_g = cached_next_game(cfg, team_id, TODAY) if team_id else None
    X_next, meta = next_game_features(df, target_col, next_g)
    models = make_base_models(params["enable_xgb"])
    with st.spinner("Training models..."):
        rows, blended_oof, blender = timeseries_oof_and_fit(X, y, models, n_splits=params["n_splits"])
        blended_rmse = _rmse(y.values, blended_oof)
        fitted, blender = refit_all(models, blender, X, y)
        pred, indiv = predict_with_ensemble(fitted, blender, X_next)
    lo, hi = pred - 1.96 * blended_rmse, pred + 1.96 * blended_rmse
    left, right = st.columns([2, 1])
    with left:
        opp_txt = f"vs **Team {meta['opponent_id']}**" + (" (HOME)" if meta["is_home"] == 1 else " (AWAY)")
        st.subheader("Prediction")
        st.markdown(f"**Target:** {params['target_label']} — **Pred:** `{pred:.2f}`  \n"
                    f"95% PI: [`{lo:.2f}`, `{hi:.2f}`]  \n"
                    f"{opp_txt}, rest days: `{meta['rest_days']}`")
        show_metrics_table(rows, blended_rmse)
    with right:
        st.metric(label=f"Predicted {params['target_label']}", value=f"{pred:.2f}")
        weight_map = dict(zip(list(models.keys()), blender.coef_.tolist()))
        wdf = pd.DataFrame({"Model": list(weight_map.keys()), "Weight": [round(w, 3) for w in weight_map.values()]})
        st.write("Model blend weights")
        st.dataframe(wdf, use_container_width=True)
    st.subheader("Feature Signals")
    imp_rows = []
    if isinstance(fitted.get("RF"), RandomForestRegressor):
        imp_rows.append(pd.Series(fitted["RF"].feature_importances_, index=feat_cols, name="RF"))
    if isinstance(fitted.get("HGB"), HistGradientBoostingRegressor):
        imp_rows.append(pd.Series(fitted["HGB"].feature_importances_, index=feat_cols, name="HGB"))
    if imp_rows:
        imp_df = pd.concat(imp_rows, axis=1).fillna(0.0)
        st.write("Tree feature importances")
        st.dataframe(imp_df.sort_values(by=list(imp_df.columns), ascending=False).head(20), use_container_width=True)
    ridge_pipe = fitted.get("Ridge")
    if isinstance(ridge_pipe, Pipeline):
        ridge = ridge_pipe.named_steps["reg"]
        scaler = ridge_pipe.named_steps["scaler"]
        coef = ridge.coef_ / (scaler.scale_ + 1e-12)
        coef_rows = pd.Series(coef, index=feat_cols, name="Ridge")
        st.write("Ridge coefficients (standardized back)")
        st.dataframe(coef_rows.sort_values(ascending=False).to_frame().head(20), use_container_width=True)
    st.subheader("Recent Game Log")
    show_recent_table(df, target_col, n=15)
    try:
        import altair as alt
        chart_df = df[["game_date", target_col]].tail(25).rename(columns={target_col: "Target"})
        roll_df = df[["game_date", target_col]].assign(roll5=df[target_col].rolling(5).mean()).tail(25)
        line1 = alt.Chart(chart_df).mark_line(point=True).encode(x="game_date:T", y="Target:Q")
        line2 = alt.Chart(roll_df).mark_line(strokeDash=[4, 2]).encode(x="game_date:T", y="roll5:Q")
        st.altair_chart((line1 + line2).interactive(), use_container_width=True)
    except Exception:
        pass
    with st.expander("Debug / Raw"):
        st.write("Next game raw object", next_g)
        st.write("Prediction breakdown per model", indiv)

if __name__ == "__main__":
    main()
