# app/streamlit_app.py
# Run: streamlit run app/streamlit_app.py

from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Constants ---
BALDONTLIE_BASE = "https://www.balldontlie.io/api/v1"
LEAGUE_AVG_DRTG_DEFAULT = 113.0
TODAY = dt.date.today()


# --- Utils ---
def _min_str_to_float(m: Optional[str | int | float]) -> float:
    """Parse 'MM:SS' to minutes; return 0.0 on bad input. Avoids crashes on messy feeds."""
    if m in (None, "", np.nan):
        return 0.0
    if isinstance(m, (int, float)):
        return float(m)
    try:
        s = str(m)
        if ":" in s:
            mm, ss = s.split(":")
            return float(mm) + float(ss) / 60.0
        return float(s)
    except Exception:
        return 0.0


def _rolling_feature(series: pd.Series, window: int) -> pd.Series:
    """Trailing mean before game. Shift prevents label leakage."""
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def _prop_column(prop: str) -> Optional[str]:
    return {"PTS": "pts", "REB": "reb", "AST": "ast", "3PM": "fg3m", "PRA": None}.get(prop)


def _compute_pra(df: pd.DataFrame) -> pd.Series:
    return df.get("pts", 0).fillna(0) + df.get("reb", 0).fillna(0) + df.get("ast", 0).fillna(0)


def _std_err(residuals: np.ndarray) -> float:
    if residuals is None or len(residuals) < 2:
        return float("nan")
    return float(np.std(residuals, ddof=1))


def _ci80(pred: float, se: float, n: int) -> Tuple[float, float]:
    """Approximate 80% CI; guarded for small samples."""
    if not np.isfinite(se) or n <= 5:
        return (float("nan"), float("nan"))
    z = 1.2816
    return (pred - z * se, pred + z * se)


# --- API Client (with API key support) ---
class BdlClient:
    def __init__(self, api_key: Optional[str] = None, base: str = BALDONTLIE_BASE):
        self.base = base
        self.sess = requests.Session()
        self.api_key = (api_key or "").strip()
        self._attach_auth_headers()

    def _attach_auth_headers(self) -> None:
        """Attach common auth headers. Kept broad to be compatible with provider setups."""
        if not self.api_key:
            return
        # Different providers use different headers; we set a few safely.
        self.sess.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",  # common pattern
                "X-API-Key": self.api_key,                  # fallback pattern
                "User-Agent": "NBA-Props-Projector/1.0 (+streamlit)",
            }
        )

    def _get(self, path: str, params: Dict) -> Dict:
        try:
            r = self.sess.get(f"{self.base}/{path}", params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            # Avoid echoing secrets.
            raise RuntimeError(f"HTTP error calling {path}: {e.response.status_code} {e.response.text[:200]}") from e
        except Exception as e:
            raise RuntimeError(f"Request error calling {path}: {e}") from e

    def search_players(self, query: str, per_page: int = 25) -> List[Dict]:
        data = self._get("players", {"search": query, "per_page": per_page})
        return data.get("data", [])

    def all_teams(self) -> pd.DataFrame:
        data = self._get("teams", {"per_page": 40})
        return pd.DataFrame(data.get("data", []))

    def player_season_stats(self, player_id: int, seasons: List[int], postseason: bool) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for season in seasons:
            page = 1
            while True:
                payload = {
                    "player_ids[]": player_id,
                    "seasons[]": season,
                    "per_page": 100,
                    "page": page,
                    "postseason": str(postseason).lower(),
                }
                data = self._get("stats", payload)
                rows = data.get("data", [])
                if not rows:
                    break
                frames.append(pd.json_normalize(rows))
                if page >= data.get("meta", {}).get("total_pages", 1):
                    break
                page += 1
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def games_for_team_in_range(self, team_id: int, start: dt.date, end: dt.date) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        page = 1
        while True:
            payload = {
                "team_ids[]": team_id,
                "per_page": 100,
                "page": page,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
            data = self._get("games", payload)
            rows = data.get("data", [])
            if not rows:
                break
            frames.append(pd.json_normalize(rows))
            if page >= data.get("meta", {}).get("total_pages", 1):
                break
            page += 1
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


# --- Cache wrappers (scope cache by API key to avoid cross-user bleed) ---
@st.cache_data(show_spinner=False)
def cached_teams_df(api_key: str) -> pd.DataFrame:
    return BdlClient(api_key).all_teams()


@st.cache_data(show_spinner=False)
def cached_player_search(q: str, api_key: str) -> pd.DataFrame:
    return pd.DataFrame(BdlClient(api_key).search_players(q))


@st.cache_data(show_spinner=False)
def cached_player_stats(player_id: int, seasons: Tuple[int, ...], postseason: bool, api_key: str) -> pd.DataFrame:
    return BdlClient(api_key).player_season_stats(player_id, list(seasons), postseason)


@st.cache_data(show_spinner=False)
def cached_games_range(team_id: int, start: dt.date, end: dt.date, api_key: str) -> pd.DataFrame:
    return BdlClient(api_key).games_for_team_in_range(team_id, start, end)


# --- Feature engineering ---
def build_game_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return stats_df.copy()

    cols = {
        "game.id": "game_id",
        "game.date": "game_date",
        "game.home_team_id": "home_team_id",
        "game.visitor_team_id": "visitor_team_id",
        "team.id": "team_id",
        "team.abbreviation": "team_abbr",
        "player.id": "player_id",
        "player.first_name": "first_name",
        "player.last_name": "last_name",
        "min": "min_raw",
        "pts": "pts",
        "reb": "reb",
        "ast": "ast",
        "stl": "stl",
        "blk": "blk",
        "fg3m": "fg3m",
        "fga": "fga",
        "fta": "fta",
        "turnover": "tov",
        "pf": "pf",
        "season": "season",
    }
    df = stats_df.rename(columns=cols)
    need = [v for v in cols.values() if v in df.columns]
    df = df[need].copy()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["min"] = df["min_raw"].apply(_min_str_to_float)
    df["is_home"] = (df["home_team_id"] == df["team_id"]).astype(int)
    df["opp_team_id"] = np.where(df["is_home"] == 1, df["visitor_team_id"], df["home_team_id"])
    df["player_name"] = df.get("first_name", "").fillna("") + " " + df.get("last_name", "").fillna("")

    df = df.sort_values("game_date").reset_index(drop=True)
    df["prev_game_date"] = df["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days.fillna(3).astype(int)

    for w in (5, 10, 20):
        df[f"r{w}_min"] = _rolling_feature(df["min"], w)
        df[f"r{w}_pts"] = _rolling_feature(df["pts"], w)
        df[f"r{w}_reb"] = _rolling_feature(df["reb"], w)
        df[f"r{w}_ast"] = _rolling_feature(df["ast"], w)
        df[f"r{w}_fg3m"] = _rolling_feature(df["fg3m"], w)
        df[f"r{w}_pra"] = _rolling_feature(_compute_pra(df), w)

    df["season_num"] = df["season"].astype(int)
    return df


def recent_windows_summary(df: pd.DataFrame, prop: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    s = _compute_pra(df) if prop == "PRA" else df[_prop_column(prop)]
    out = []
    for w in [1, 5, 10, 15, 20]:
        sub = s.tail(w)
        out.append(
            dict(
                Window=f"L{w}",
                Games=int(len(sub)),
                Avg=float(sub.mean()) if len(sub) else np.nan,
                Median=float(sub.median()) if len(sub) else np.nan,
                Std=float(sub.std(ddof=1)) if len(sub) > 1 else np.nan,
                High=float(sub.max()) if len(sub) else np.nan,
                Low=float(sub.min()) if len(sub) else np.nan,
            )
        )
    return pd.DataFrame(out)


def h2h_summary(df: pd.DataFrame, prop: str, opponent_team_id: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    opp_df = df[df["opp_team_id"] == opponent_team_id].copy()
    if opp_df.empty:
        return pd.DataFrame([{"Games": 0, "Avg": np.nan, "Median": np.nan, "Std": np.nan}])
    s = _compute_pra(opp_df) if prop == "PRA" else opp_df[_prop_column(prop)]
    return pd.DataFrame(
        [
            {
                "Games": int(len(s)),
                "Avg": float(s.mean()),
                "Median": float(s.median()),
                "Std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                "High": float(s.max()),
                "Low": float(s.min()),
            }
        ]
    )


def find_next_opponent(team_id: int, teams_df: pd.DataFrame, api_key: str) -> Tuple[Optional[int], Optional[str], Optional[dt.date], Optional[bool]]:
    start = TODAY
    end = TODAY + dt.timedelta(days=21)
    games = cached_games_range(team_id, start, end, api_key)
    if games.empty:
        return None, None, None, None
    games["date"] = pd.to_datetime(games["date"]).dt.date
    games = games.sort_values("date")
    for _, row in games.iterrows():
        gdate = row["date"]
        if gdate < TODAY:
            continue
        home_id = int(row.get("home_team.id"))
        away_id = int(row.get("visitor_team.id"))
        is_home_next = team_id == home_id
        opp_id = away_id if is_home_next else home_id
        opp_abbr = teams_df.loc[teams_df["id"] == opp_id, "abbreviation"].values
        return opp_id, (opp_abbr[0] if len(opp_abbr) else None), gdate, is_home_next
    return None, None, None, None


# --- Modeling ---
def prepare_model_table(df: pd.DataFrame, prop: str) -> Tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    feat_cols = [
        "is_home",
        "days_rest",
        "r5_min",
        "r10_min",
        "r5_pts",
        "r10_pts",
        "r5_reb",
        "r10_reb",
        "r5_ast",
        "r10_ast",
        "r5_fg3m",
        "r10_fg3m",
        "r5_pra",
        "r10_pra",
        "season_num",
    ]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = (_compute_pra(df) if prop == "PRA" else df[_prop_column(prop)]).fillna(0.0)
    return X, y


def train_ridge_or_wma(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Pipeline], float, np.ndarray]:
    """Ridge for n>=10, otherwise weighted moving average. Keeps small-sample behavior stable."""
    n = len(X)
    if n < 10:
        if n == 0:
            return None, np.nan, np.array([])
        weights = np.linspace(0.5, 1.0, num=min(n, 10))[-n:]
        pred = float(np.average(y.values[-len(weights):], weights=weights))
        return None, pred, np.array([])
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))])
    model.fit(X, y)
    yhat = model.predict(X)
    residuals = y.values - yhat
    return model, float(yhat[-1]), residuals  # last in-sample approx is stable enough for CI calc


def home_away_multiplier(df: pd.DataFrame, prop: str) -> Optional[float]:
    if df.empty:
        return None
    s = _compute_pra(df) if prop == "PRA" else df[_prop_column(prop)]
    home = s[df["is_home"] == 1]
    away = s[df["is_home"] == 0]
    if len(home) >= 5 and len(away) >= 5 and away.mean() > 0:
        return float(home.mean() / away.mean())
    return None


def project_next(
    model: Optional[Pipeline],
    X_latest: pd.DataFrame,
    base_pred: float,
    opp_drtg: Optional[float],
    league_avg: float,
    beta: float,
    home_mult: Optional[float],
) -> float:
    """Simple, explainable scaling on top of model/WMA prediction."""
    pred = float(model.predict(X_latest)[0]) if model is not None else base_pred
    if home_mult and np.isfinite(home_mult):
        pred *= float(home_mult)
    if opp_drtg and np.isfinite(opp_drtg) and opp_drtg > 0:
        adj = (league_avg / float(opp_drtg)) - 1.0
        pred *= (1.0 + beta * adj)
    return float(max(0.0, pred))


# --- UI ---
st.set_page_config(page_title="NBA Player Props Projector", layout="wide")
st.title("🏀 NBA Player Props Projector")

# Secure API key handling
default_key = st.secrets.get("balldontlie_api_key", os.getenv("BALDONTLIE_API_KEY", ""))
api_key = st.sidebar.text_input(
    "balldontlie API Key (All-Star tier)",
    value=default_key,
    type="password",
    help="Stored only in this session. You can also set st.secrets['balldontlie_api_key'] or env BALDONTLIE_API_KEY.",
)
if not api_key:
    st.warning("Enter your balldontlie API key to unlock full stats.")
    # Continue; some endpoints may still work if public.

st.sidebar.header("Controls")
season_current = TODAY.year if TODAY.month >= 10 else TODAY.year - 1
seasons = st.sidebar.multiselect(
    "Seasons (regular)",
    options=list(range(season_current, season_current - 6, -1)),
    default=[season_current, season_current - 1],
)
include_post = st.sidebar.checkbox("Include playoffs", value=False)
prop = st.sidebar.selectbox("Prop", ["PTS", "REB", "AST", "3PM", "PRA"])
beta = st.sidebar.slider("Defensive rating impact (β)", 0.0, 1.0, 0.6, 0.05)
league_avg_drtg = st.sidebar.number_input("League average DRtg", value=LEAGUE_AVG_DRTG_DEFAULT, step=0.5)

st.sidebar.markdown("### Defensive Ratings")
drtg_file = st.sidebar.file_uploader("Upload Team DRtg CSV (abbreviation,DefRtg)", type=["csv"])
manual_opp_drtg = st.sidebar.number_input("Manual Opp DRtg (override)", value=0.0, step=0.5)

st.sidebar.markdown("---")
player_query = st.sidebar.text_input("Search player", value="")
search_btn = st.sidebar.button("Search")

# Data
try:
    teams_df = cached_teams_df(api_key)
except Exception as e:
    st.error(f"Failed to load teams. {e}")
    teams_df = pd.DataFrame()

drtg_map: Dict[str, float] = {}
if drtg_file is not None:
    try:
        df_drtg = pd.read_csv(drtg_file)
        cand_abbr = [c for c in df_drtg.columns if c.lower() in ("abbr", "abbreviation", "team", "team_abbr")]
        cand_val = [c for c in df_drtg.columns if c.lower() in ("defrtg", "drtg", "def_rating", "defensive_rating")]
        if cand_abbr and cand_val:
            drtg_map = dict(zip(df_drtg[cand_abbr[0]].str.upper().str.strip(), df_drtg[cand_val[0]].astype(float)))
    except Exception as e:
        st.sidebar.warning(f"DRtg CSV parse error: {e}")

if search_btn and not player_query.strip():
    st.warning("Enter a player name to search.")

if player_query.strip():
    try:
        search_df = cached_player_search(player_query.strip(), api_key)
    except Exception as e:
        st.error(f"Player search failed. {e}")
        st.stop()

    if search_df.empty:
        st.info("No players found.")
        st.stop()

    def _row_to_label(r) -> str:
        t = r.get("team", {}) or {}
        return f"{r.get('first_name','')} {r.get('last_name','')} — {t.get('abbreviation','FA')} — #{r.get('id')}"

    labels = {_row_to_label(r): r for r in search_df.to_dict(orient="records")}
    sel_label = st.selectbox("Select player", list(labels.keys()))
    player_obj = labels[sel_label]
    player_id = int(player_obj["id"])
    player_name = f"{player_obj.get('first_name','')} {player_obj.get('last_name','')}"
    player_team = (player_obj.get("team") or {}).get("abbreviation", "FA")
    player_team_id = (player_obj.get("team") or {}).get("id", None)

    st.subheader(f"Player: {player_name} ({player_team})")

    if not seasons:
        st.error("Select at least one season.")
        st.stop()

    try:
        stats_raw = cached_player_stats(player_id, tuple(sorted(seasons)), include_post, api_key)
    except Exception as e:
        st.error(f"Failed to load player stats. {e}")
        st.stop()

    if stats_raw.empty:
        st.info("No stats returned for selected seasons/options.")
        st.stop()

    games_df = build_game_table(stats_raw)

    opp_id, opp_abbr, game_date, is_home_next = (None, None, None, None)
    if player_team_id:
        try:
            opp_id, opp_abbr, game_date, is_home_next = find_next_opponent(player_team_id, teams_df, api_key)
        except Exception as e:
            st.warning(f"Could not resolve next opponent. {e}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Next Game", f"{(game_date or 'N/A')}")
    with c2:
        st.metric("Opponent", f"{opp_abbr or 'N/A'}")
    with c3:
        st.metric("Venue", "Home" if is_home_next else ("Away" if is_home_next is False else "—"))

    opp_drtg = None
    if manual_opp_drtg > 0:
        opp_drtg = float(manual_opp_drtg)
    elif opp_abbr and opp_abbr.upper() in drtg_map:
        opp_drtg = float(drtg_map[opp_abbr.upper()])

    X, y = prepare_model_table(games_df, prop)
    model, base_pred_estimate, residuals = train_ridge_or_wma(X, y)
    X_latest = X.tail(1) if not X.empty else pd.DataFrame()
    ha_mult = home_away_multiplier(games_df, prop)

    pred_value = project_next(
        model=model,
        X_latest=X_latest if not X_latest.empty else pd.DataFrame([X.iloc[-1].to_dict()]) if not X.empty else pd.DataFrame(),
        base_pred=base_pred_estimate,
        opp_drtg=opp_drtg,
        league_avg=league_avg_drtg,
        beta=beta,
        home_mult=ha_mult,
    )

    se = _std_err(residuals)
    ci_lo, ci_hi = _ci80(pred_value, se, len(X))
    drivers = {
        "Home/Away Mult": round(ha_mult, 3) if ha_mult else None,
        "Opp DRtg": opp_drtg,
        "β": beta,
        "League Avg DRtg": league_avg_drtg,
        "Sample Size": int(len(X)),
        "Residual Std": round(se, 2) if np.isfinite(se) else None,
    }

    left, mid, right = st.columns([1.2, 1.2, 1])
    with left:
        st.markdown("### Projection")
        st.metric(f"{prop} Projection", f"{pred_value:.2f}")
        st.caption(
            f"80% CI: {'' if np.isfinite(ci_lo) else '—'}{f'{ci_lo:.1f}–{ci_hi:.1f}' if np.isfinite(ci_lo) else 'Insufficient data'}"
        )
    with mid:
        st.markdown("### Drivers")
        st.json(drivers)
    with right:
        st.markdown("### Next Game")
        st.write(
            {
                "Date": str(game_date) if game_date else "Unknown",
                "Opponent": opp_abbr or "Unknown",
                "Team": player_team or "Unknown",
                "Venue": "Home" if is_home_next else ("Away" if is_home_next is False else "Unknown"),
            }
        )

    st.markdown("---")
    st.header("Research")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recent Form (L1/5/10/15/20)")
        recent_df = recent_windows_summary(games_df, prop)
        st.dataframe(recent_df, use_container_width=True)
    with col2:
        st.subheader("H2H vs Upcoming Opponent (All-time)")
        if opp_id:
            h2h_df = h2h_summary(games_df, prop, opp_id)
            st.dataframe(h2h_df, use_container_width=True)
        else:
            st.info("Opponent not resolved yet.")

    st.subheader("Recent Trend")
    plot_df = games_df[["game_date", "pts", "reb", "ast", "fg3m"]].copy()
    plot_df["PRA"] = _compute_pra(games_df)
    plot_df = plot_df.set_index("game_date")
    st.line_chart(plot_df[[prop]] if prop in plot_df.columns else plot_df[["PRA"]])

    st.subheader("Distribution (Last 30)")
    last_n = 30
    series = (plot_df[prop] if prop in plot_df.columns else plot_df["PRA"]).tail(last_n).dropna()
    st.bar_chart(series.value_counts().sort_index())

    exp_cols = ["game_date", "team_abbr", "opp_team_id", "is_home", "min", "pts", "reb", "ast", "fg3m", "season"]
    exp_cols = [c for c in exp_cols if c in games_df.columns]
    export_df = games_df[exp_cols].copy()
    export_df["PRA"] = _compute_pra(games_df)
    st.download_button(
        "Download player game log (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{player_name.replace(' ', '_')}_gamelog.csv",
        mime="text/csv",
    )
else:
    st.info("Search a player to begin.")
