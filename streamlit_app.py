# app/streamlit_app.py
# Run: streamlit run app/streamlit_app.py

import math
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---- Constants ----
BALDONTLIE_BASE = "https://www.balldontlie.io/api/v1"
LEAGUE_AVG_DRTG_DEFAULT = 113.0  # neutral baseline
TODAY = dt.date.today()

# ---- Utils ----
def _min_str_to_float(m: Optional[str]) -> float:
    if not m:
        return 0.0
    if isinstance(m, (int, float)):
        return float(m)
    try:
        if ":" in m:
            mm, ss = m.split(":")
            return float(mm) + float(ss) / 60.0
        return float(m)
    except Exception:
        return 0.0

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b if b else default
    except Exception:
        return default

def _rolling_feature(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()

def _days_between(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    return int((d1.date() - d2.date()).days)

def _prop_column(prop: str) -> str:
    mapping = {
        "PTS": "pts",
        "REB": "reb",
        "AST": "ast",
        "3PM": "fg3m",
        "PRA": None,  # computed
    }
    return mapping.get(prop)

def _compute_pra(df: pd.DataFrame) -> pd.Series:
    return df["pts"].fillna(0) + df["reb"].fillna(0) + df["ast"].fillna(0)

def _std_err(residuals: np.ndarray) -> float:
    if len(residuals) < 2:
        return float("nan")
    return float(np.std(residuals, ddof=1))

def _ci80(pred: float, se: float, n: int) -> Tuple[float, float]:
    if not np.isfinite(se) or n <= 5:
        return (float("nan"), float("nan"))
    z = 1.2816
    return (pred - z * se, pred + z * se)

# ---- API Client ----
class BdlClient:
    def __init__(self, base: str = BALDONTLIE_BASE, session: Optional[requests.Session] = None):
        self.base = base
        self.sess = session or requests.Session()

    def _get(self, path: str, params: Dict) -> Dict:
        r = self.sess.get(f"{self.base}/{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

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
        df = pd.concat(frames, ignore_index=True)
        return df

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

# ---- Cache wrappers ----
@st.cache_data(show_spinner=False)
def cached_teams_df() -> pd.DataFrame:
    return BdlClient().all_teams()

@st.cache_data(show_spinner=False)
def cached_player_search(q: str) -> pd.DataFrame:
    return pd.DataFrame(BdlClient().search_players(q))

@st.cache_data(show_spinner=False)
def cached_player_stats(player_id: int, seasons: Tuple[int, ...], postseason: bool) -> pd.DataFrame:
    return BdlClient().player_season_stats(player_id, list(seasons), postseason)

@st.cache_data(show_spinner=False)
def cached_games_range(team_id: int, start: dt.date, end: dt.date) -> pd.DataFrame:
    return BdlClient().games_for_team_in_range(team_id, start, end)

# ---- Feature engineering ----
def build_game_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return stats_df

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
    need = list(cols.values())
    need = [c for c in need if c in df.columns]
    df = df[need].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["min"] = df["min_raw"].apply(_min_str_to_float)
    df["is_home"] = (df["home_team_id"] == df["team_id"]).astype(int)
    df["opp_team_id"] = np.where(df["is_home"] == 1, df["visitor_team_id"], df["home_team_id"])
    df["player_name"] = df["first_name"].fillna("") + " " + df["last_name"].fillna("")
    # Sort chronologically
    df = df.sort_values("game_date").reset_index(drop=True)
    # Days rest (pre-game)
    df["prev_game_date"] = df["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days.fillna(3).astype(int)
    # Rolling pre-game features
    for w in (5, 10, 20):
        df[f"r{w}_min"] = _rolling_feature(df["min"], w)
        df[f"r{w}_pts"] = _rolling_feature(df["pts"], w)
        df[f"r{w}_reb"] = _rolling_feature(df["reb"], w)
        df[f"r{w}_ast"] = _rolling_feature(df["ast"], w)
        df[f"r{w}_fg3m"] = _rolling_feature(df["fg3m"], w)
        df[f"r{w}_pra"] = _rolling_feature(_compute_pra(df), w)
    # Season numeric feature
    df["season_num"] = df["season"].astype(int)
    return df

def recent_windows_summary(df: pd.DataFrame, prop: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if prop == "PRA":
        series = _compute_pra(df)
    else:
        series = df[_prop_column(prop)]
    windows = [1, 5, 10, 15, 20]
    out = []
    for w in windows:
        sub = series.tail(w)
        out.append({
            "Window": f"L{w}",
            "Games": int(len(sub)),
            "Avg": float(sub.mean()) if len(sub) else np.nan,
            "Median": float(sub.median()) if len(sub) else np.nan,
            "Std": float(sub.std(ddof=1)) if len(sub) > 1 else np.nan,
            "High": float(sub.max()) if len(sub) else np.nan,
            "Low": float(sub.min()) if len(sub) else np.nan,
        })
    return pd.DataFrame(out)

def h2h_summary(df: pd.DataFrame, prop: str, opponent_team_id: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    opp_df = df[df["opp_team_id"] == opponent_team_id].copy()
    if opp_df.empty:
        return pd.DataFrame([{"Games": 0, "Avg": np.nan, "Median": np.nan, "Std": np.nan}])
    if prop == "PRA":
        s = _compute_pra(opp_df)
    else:
        s = opp_df[_prop_column(prop)]
    return pd.DataFrame([{
        "Games": int(len(s)),
        "Avg": float(s.mean()),
        "Median": float(s.median()),
        "Std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "High": float(s.max()),
        "Low": float(s.min()),
    }])

def find_next_opponent(team_id: int, teams_df: pd.DataFrame) -> Tuple[Optional[int], Optional[str], Optional[dt.date]]:
    start = TODAY
    end = TODAY + dt.timedelta(days=21)
    games = cached_games_range(team_id, start, end)
    if games.empty:
        return None, None, None
    games["date"] = pd.to_datetime(games["date"]).dt.date
    # choose earliest future game
    games = games.sort_values("date")
    for _, row in games.iterrows():
        home_id = row.get("home_team.id")
        away_id = row.get("visitor_team.id")
        gid = row["id"]
        gdate = row["date"]
        if gdate < TODAY:
            continue
        if team_id == home_id:
            opp_id = int(away_id)
        else:
            opp_id = int(home_id)
        opp_abbr = teams_df.loc[teams_df["id"] == opp_id, "abbreviation"].values
        opp_abbr = opp_abbr[0] if len(opp_abbr) else None
        return opp_id, opp_abbr, gdate
    return None, None, None

# ---- Modeling ----
def prepare_model_table(df: pd.DataFrame, prop: str) -> Tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    feat_cols = [
        "is_home", "days_rest",
        "r5_min", "r10_min",
        "r5_pts", "r10_pts",
        "r5_reb", "r10_reb",
        "r5_ast", "r10_ast",
        "r5_fg3m", "r10_fg3m",
        "r5_pra", "r10_pra",
        "season_num",
    ]
    # Target
    if prop == "PRA":
        y = _compute_pra(df)
    else:
        y = df[_prop_column(prop)]
    X = df[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.fillna(0.0)
    return X, y

def train_ridge_or_wma(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Pipeline], float, np.ndarray]:
    n = len(X)
    if n < 10:
        weights = np.linspace(0.5, 1.0, num=min(n, 10))[-n:]
        pred = float(np.average(y.values[-len(weights):], weights=weights)) if n else np.nan
        return None, pred, np.array([])
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    model.fit(X, y)
    yhat = model.predict(X)
    residuals = y.values - yhat
    return model, float(yhat[-1]), residuals  # last in-sample point approx

def project_next(
    model: Optional[Pipeline],
    X_latest: pd.DataFrame,
    base_pred: float,
    opp_drtg: Optional[float],
    league_avg: float,
    beta: float,
    home_mult: Optional[float]
) -> float:
    pred = float(model.predict(X_latest)[0]) if model is not None else base_pred
    if home_mult and np.isfinite(home_mult):
        pred *= home_mult
    if opp_drtg and np.isfinite(opp_drtg) and opp_drtg > 0:
        adj = (league_avg / opp_drtg) - 1.0
        pred *= (1.0 + beta * adj)
    return float(max(0.0, pred))

def home_away_multiplier(df: pd.DataFrame, prop: str) -> Optional[float]:
    if df.empty:
        return None
    if prop == "PRA":
        s = _compute_pra(df)
    else:
        s = df[_prop_column(prop)]
    home = s[df["is_home"] == 1]
    away = s[df["is_home"] == 0]
    if len(home) >= 5 and len(away) >= 5 and away.mean() > 0:
        return float(home.mean() / away.mean())
    return None

# ---- Streamlit UI ----
st.set_page_config(page_title="NBA Props Projector", layout="wide")
st.title("🏀 NBA Player Props Projector")

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

teams_df = cached_teams_df()
team_by_id = teams_df.set_index("id").to_dict(orient="index")
abbr_to_id = {r["abbreviation"]: r["id"] for _, r in teams_df.iterrows()}

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
    search_df = cached_player_search(player_query.strip())
    if search_df.empty:
        st.info("No players found.")
    else:
        # Build options "Name — TEAM — pos #id"
        def _row_to_label(r) -> str:
            t = r.get("team", {}) or {}
            return f"{r.get('first_name','')} {r.get('last_name','')} — {t.get('abbreviation','FA')} — #{r.get('id')}"
        labels = { _row_to_label(r): r for r in search_df.to_dict(orient="records") }
        sel_label = st.selectbox("Select player", list(labels.keys()))
        player_obj = labels[sel_label]
        player_id = int(player_obj["id"])
        player_name = f"{player_obj.get('first_name','')} {player_obj.get('last_name','')}"
        player_team = (player_obj.get("team") or {}).get("abbreviation", "FA")
        player_team_id = (player_obj.get("team") or {}).get("id", None)

        st.subheader(f"Player: {player_name} ({player_team})")

        # Pull stats
        if not seasons:
            st.error("Select at least one season.")
            st.stop()
        stats_raw = cached_player_stats(player_id, tuple(sorted(seasons)), include_post)
        if stats_raw.empty:
            st.info("No stats returned for selected seasons/options.")
            st.stop()

        games_df = build_game_table(stats_raw)

        # Next opponent
        opp_id, opp_abbr, game_date = (None, None, None)
        if player_team_id:
            opp_id, opp_abbr, game_date = find_next_opponent(player_team_id, teams_df)
        colt1, colt2, colt3 = st.columns(3)
        with colt1:
            st.metric("Next Game", f"{(game_date or 'N/A')}")
        with colt2:
            st.metric("Opponent", f"{opp_abbr or 'N/A'}")
        with colt3:
            st.metric("Home", "Yes" if not games_df.empty and player_team_id and opp_id and
                      (player_team_id == games_df.iloc[-1]["home_team_id"]) else "—")

        # Opponent DRtg
        opp_drtg = None
        if manual_opp_drtg > 0:
            opp_drtg = float(manual_opp_drtg)
        elif opp_abbr and opp_abbr.upper() in drtg_map:
            opp_drtg = float(drtg_map[opp_abbr.upper()])

        # Modeling table
        X, y = prepare_model_table(games_df, prop)
        model, base_pred_estimate, residuals = train_ridge_or_wma(X, y)
        # Latest features row
        X_latest = X.tail(1)
        # Home/away multiplier
        ha_mult = home_away_multiplier(games_df, prop)

        # Projection
        pred_value = project_next(
            model=model,
            X_latest=X_latest if not X_latest.empty else pd.DataFrame([X.iloc[-1].to_dict()]) if not X.empty else pd.DataFrame(),
            base_pred=base_pred_estimate,
            opp_drtg=opp_drtg,
            league_avg=league_avg_drtg,
            beta=beta,
            home_mult=ha_mult,
        )

        # Confidence via residual std
        se = _std_err(residuals)
        ci_lo, ci_hi = _ci80(pred_value, se, len(X))
        # Drivers
        drivers = {
            "Home/Away Mult": round(ha_mult, 3) if ha_mult else None,
            "Opp DRtg": opp_drtg,
            "β": beta,
            "League Avg DRtg": league_avg_drtg,
            "Sample Size": int(len(X)),
            "Residual Std": round(se, 2) if np.isfinite(se) else None,
        }

        # ---- Output cards ----
        top_left, top_mid, top_right = st.columns([1.2, 1.2, 1])
        with top_left:
            st.markdown("### Projection")
            st.metric(f"{prop} Projection", f"{pred_value:.2f}",
                      help="80% CI is an approximate uncertainty band.")
            st.caption(f"80% CI: {'' if np.isfinite(ci_lo) else '—'}{f'{ci_lo:.1f}–{ci_hi:.1f}' if np.isfinite(ci_lo) else 'Insufficient data'}")
        with top_mid:
            st.markdown("### Drivers")
            st.json(drivers)
        with top_right:
            st.markdown("### Next Game")
            st.write({
                "Date": str(game_date) if game_date else "Unknown",
                "Opponent": opp_abbr or "Unknown",
                "Team": player_team or "Unknown",
            })

        # ---- Research area ----
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

        # Charts
        st.subheader("Recent Trend")
        plot_df = games_df[["game_date", "pts", "reb", "ast", "fg3m"]].copy()
        plot_df["PRA"] = _compute_pra(games_df)
        plot_df = plot_df.set_index("game_date")
        st.line_chart(plot_df[[prop]] if prop in plot_df.columns else plot_df[["PRA"]])

        st.subheader("Distribution (Last 30)")
        last_n = 30
        dist_series = (plot_df[prop] if prop in plot_df.columns else plot_df["PRA"]).tail(last_n).dropna()
        st.bar_chart(dist_series.value_counts().sort_index())

        # Export
        exp_cols = ["game_date", "team_abbr", "opp_team_id", "is_home", "min", "pts", "reb", "ast", "fg3m", "season"]
        exp_cols = [c for c in exp_cols if c in games_df.columns]
        export_df = games_df[exp_cols].copy()
        export_df["PRA"] = _compute_pra(games_df)
        st.download_button(
            "Download player game log (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{player_name.replace(' ','_')}_gamelog.csv",
            mime="text/csv",
        )

        # Notes
        with st.expander("Method notes"):
            st.markdown(
                """
- Uses balldontlie game-level stats per selected seasons. Pre-game features are trailing averages (no target leakage).
- Model: Ridge regression with standardized features; falls back to weighted moving average if <10 samples.
- Optional defensive adjustment scales the projection by opponent DRtg relative to league average (β controls strength).
- Confidence is an approximate 80% interval based on in-sample residuals; treat as a heuristic.
- Upload a Team→DefRtg CSV to enable DRtg adjustment (e.g., columns: `abbreviation,DefRtg`).
                """
            )
else:
    st.info("Search a player to begin.")
