import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from scipy.stats import norm

# ============================================================
# CONFIG
# ============================================================

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # <- your BallDontLie key
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.set_page_config(page_title="NBA Luxe AI Predictor", layout="wide")

# ============================================================
# THEME: Black + Purple Luxe
# ============================================================

LUXE_CSS = """
<style>
/* Global */
.stApp {
    background: radial-gradient(circle at top, #3b0066 0, #050009 40%, #020006 100%) !important;
    color: #f5f2ff !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

/* Centered main container */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #f8f0ff !important;
    font-weight: 700 !important;
}

/* Purple cards */
.luxe-card {
    background: rgba(8, 1, 20, 0.9);
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
    border: 1px solid rgba(155, 89, 182, 0.8);
    box-shadow: 0 0 30px rgba(155, 89, 182, 0.45);
}

/* Soft cards */
.soft-card {
    background: rgba(15, 8, 30, 0.9);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    border: 1px solid rgba(155, 89, 182, 0.4);
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #fff;
    font-weight: 800;
}
[data-testid="stMetricLabel"] {
    color: #c2b5ff;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #8e44ad, #9b59b6);
    color: #fff;
    border-radius: 999px;
    border: none;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
}
.stButton>button:hover {
    box-shadow: 0 0 18px rgba(155, 89, 182, 0.75);
}

/* Inputs */
.stTextInput>div>div>input,
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background: rgba(18, 8, 40, 0.9) !important;
    border-radius: 999px !important;
    border: 1px solid rgba(155, 89, 182, 0.7) !important;
    color: #f8f0ff !important;
}

/* Sidebar saved players */
.sidebar-section {
    background: rgba(8, 1, 20, 0.96);
    border-radius: 16px;
    padding: 0.8rem 0.8rem 1.1rem 0.8rem;
    border: 1px solid rgba(155, 89, 182, 0.7);
}

.sidebar-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: #f5f2ff;
}

.sidebar-player {
    font-size: 0.87rem;
    margin-bottom: 0.3rem;
}

/* Images */
.player-image {
    border-radius: 16px;
    border: 2px solid rgba(155, 89, 182, 0.8);
    box-shadow: 0 0 22px rgba(155, 89, 182, 0.5);
    max-width: 100%;
}

/* Tables */
table {
    color: #f5f2ff !important;
}
</style>
"""
st.markdown(LUXE_CSS, unsafe_allow_html=True)

# ============================================================
# CONSTANTS / SETTINGS
# ============================================================

# Stats weâll model & simulate
STAT_COLUMNS = [
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "turnover",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "pf",
]

DEFAULT_MONTE_CARLO_SIMS = 5000  # can be increased later if you want

# ============================================================
# API HELPERS (with caching)
# ============================================================

@st.cache_data(show_spinner=False, ttl=300)
def api_get(endpoint: str, params: dict | None = None):
    try:
        r = requests.get(
            f"{BASE_URL}/{endpoint}",
            headers=HEADERS,
            params=params,
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "data": []}


@st.cache_data(show_spinner=False, ttl=600)
def search_players(query: str, per_page: int = 25):
    if not query or len(query.strip()) < 2:
        return []
    data = api_get("players", {"search": query.strip(), "per_page": per_page, "page": 1})
    return data.get("data", [])


@st.cache_data(show_spinner=False, ttl=600)
def get_player_stats(player_id: int, n_games: int = 40):
    # BDL returns most recent first
    data = api_get("stats", {"player_ids[]": player_id, "per_page": n_games})
    stats = data.get("data", [])
    if not stats:
        return pd.DataFrame()
    df = pd.json_normalize(stats)
    return df


@st.cache_data(show_spinner=False, ttl=600)
def get_teams():
    # get all teams for opponent dropdown
    data = api_get("teams", {"per_page": 50})
    return data.get("data", [])


def get_headshot_url(player_id: int) -> str:
    # Using Balldontlie headshot CDN as chosen (H2)
    return f"https://balldontlie.io/images/headshots/{player_id}.png"


# ============================================================
# FEATURE ENGINEERING & MODELS
# ============================================================

def normalize_stats_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract and normalize just what we need."""
    # Flatten nested columns if any (e.g. "stats.pts")
    cols = df_raw.columns
    # Try to map to base stat names
    stat_map = {}
    for stat in STAT_COLUMNS:
        candidates = [c for c in cols if c.split(".")[-1] == stat]
        if len(candidates) > 0:
            stat_map[stat] = candidates[0]

    if not stat_map:
        return pd.DataFrame()

    df = pd.DataFrame()
    for stat, col in stat_map.items():
        df[stat] = pd.to_numeric(df_raw[col], errors="coerce")

    # Drop rows with all nan
    df = df.dropna(how="all")
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling averages as features + simple form indicators."""
    feat_df = df.copy()
    for col in STAT_COLUMNS:
        if col in feat_df.columns:
            feat_df[f"{col}_roll_{window}"] = feat_df[col].rolling(window).mean()
    # Simple last-game features
    for col in STAT_COLUMNS:
        if col in feat_df.columns:
            feat_df[f"{col}_prev"] = feat_df[col].shift(1)

    # Drop initial NA rows from rolling features
    feat_df = feat_df.dropna()
    return feat_df


def build_pace_and_opponent_features(feat_df: pd.DataFrame, pace_factor: float, opp_difficulty: float) -> pd.DataFrame:
    """Inject pace & opponent sliders as features so models can condition on them."""
    feat_df = feat_df.copy()
    feat_df["pace_factor"] = pace_factor
    feat_df["opp_difficulty"] = opp_difficulty
    # interaction feature for 'deep-ish' behaviour
    feat_df["pace_x_opp"] = pace_factor * opp_difficulty
    return feat_df


def train_models_for_player(
    feat_df: pd.DataFrame,
    targets: list[str],
    pace_factor: float,
    opp_difficulty: float,
):
    """Train an XGB model per stat target, with simple residual std for MC."""
    models = {}
    residuals = {}

    # Add pace/opponent dims to features
    feat_df = build_pace_and_opponent_features(feat_df, pace_factor, opp_difficulty)

    # Features: all rolling + prev + pace/opponent
    feature_cols = [c for c in feat_df.columns if any(
        [
            c.endswith("_roll_5"),
            c.endswith("_prev"),
            c in ["pace_factor", "opp_difficulty", "pace_x_opp"],
        ]
    )]

    if len(feature_cols) == 0:
        return {}, {}

    X = feat_df[feature_cols].values

    for stat in targets:
        if stat not in feat_df.columns:
            continue
        y = feat_df[stat].values
        if len(np.unique(y)) <= 1 or len(y) < 8:
            # Too few games or no variance: skip modeling this stat
            continue

        model = XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=2,
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        resid = y - y_pred
        resid_std = float(np.std(resid)) if len(resid) > 1 else 0.0

        models[stat] = (model, feature_cols)
        residuals[stat] = resid_std

    return models, residuals


def predict_next_game(
    models: dict,
    residuals: dict,
    latest_feat_row: pd.Series,
    pace_factor: float,
    opp_difficulty: float,
):
    """Make stat predictions + attach residual std for MC."""
    preds = {}
    stds = {}

    base_row = latest_feat_row.copy()
    base_row["pace_factor"] = pace_factor
    base_row["opp_difficulty"] = opp_difficulty
    base_row["pace_x_opp"] = pace_factor * opp_difficulty

    for stat, (model, feature_cols) in models.items():
        x = np.array([base_row[feature_cols].values], dtype=float)
        mu = float(model.predict(x)[0])
        preds[stat] = max(0.0, mu)  # stats can't be negative
        stds[stat] = max(0.5, residuals.get(stat, 1.0))  # floor on std

    return preds, stds


# ============================================================
# MONTE CARLO ENGINE
# ============================================================

def run_monte_carlo(pred_means: dict, pred_stds: dict, n_sims: int = DEFAULT_MONTE_CARLO_SIMS):
    """
    Run simple normal-based MC for every stat. Easy to swap in
    more complex distributions later.
    """
    stats = list(pred_means.keys())
    sims = {}

    for stat in stats:
        mu = pred_means[stat]
        sigma = max(0.5, pred_stds.get(stat, 1.0))
        draws = np.random.normal(mu, sigma, size=n_sims)
        draws = np.clip(draws, 0, None)
        sims[stat] = draws

    sims_df = pd.DataFrame(sims)
    return sims_df


def summarize_distribution(draws: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(draws)),
        "std": float(np.std(draws)),
        "p25": float(np.percentile(draws, 25)),
        "p50": float(np.percentile(draws, 50)),
        "p75": float(np.percentile(draws, 75)),
        "p90": float(np.percentile(draws, 90)),
    }


# ============================================================
# SESSION STATE HELPERS
# ============================================================

if "saved_players" not in st.session_state:
    st.session_state["saved_players"] = []  # list of dicts


def save_player_prediction(name: str, preds: dict):
    st.session_state["saved_players"].append(
        {
            "name": name,
            **preds,
        }
    )


# ============================================================
# MAIN UI
# ============================================================

def main():
    st.markdown(
        "<h1 style='text-align:center;'>ð NBA Luxe AI Predictor</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#d9c0ff;'>Black + Purple single-page research lab powered by BallDontLie + XGBoost + Monte Carlo.</p>",
        unsafe_allow_html=True,
    )

    # ------------------ SIDEBAR: Saved players compare ------------------
    with st.sidebar:
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-title'>â­ Saved Players</div>", unsafe_allow_html=True)
        if st.session_state["saved_players"]:
            for p in st.session_state["saved_players"]:
                st.markdown(
                    f"<div class='sidebar-player'>{p['name']}</div>",
                    unsafe_allow_html=True,
                )
            if st.button("Clear Saved Players", key="clear_saved"):
                st.session_state["saved_players"] = []
        else:
            st.markdown(
                "<div class='sidebar-player'>No players saved yet.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ Player search + opponent/pace controls ----------
    with st.container():
        st.markdown("<div class='luxe-card'>", unsafe_allow_html=True)

        col_search, col_opp, col_pace = st.columns([2.3, 1.3, 1.3])

        with col_search:
            query = st.text_input("Search player (autocomplete)", value="", placeholder="Type at least 2 lettersâ¦")
            players = search_players(query) if len(query) >= 2 else []

            player_choice = None
            player_obj = None

            if players:
                names = [
                    f"{p['first_name']} {p['last_name']} ({p['team']['abbreviation']})"
                    for p in players
                ]
                idx = st.selectbox("Select player", list(range(len(names))), format_func=lambda i: names[i])
                player_choice = players[idx]
                player_obj = player_choice
            elif len(query) >= 2:
                st.warning("No players found for that search yet. Try full name.")

        with col_opp:
            teams = get_teams()
            if teams:
                team_names = [f"{t['full_name']} ({t['abbreviation']})" for t in teams]
                team_ids = [t["id"] for t in teams]
                opp_idx = st.selectbox("Opponent (for adjustment)", list(range(len(team_names))), format_func=lambda i: team_names[i])
                opp_team_id = team_ids[opp_idx]
            else:
                st.write("No team data.")
                opp_team_id = None

        with col_pace:
            pace_factor = st.slider("Expected Pace Factor", 0.8, 1.2, 1.0, 0.01)
            opp_difficulty = st.slider("Opponent Difficulty", 0.8, 1.2, 1.0, 0.01)

        st.markdown("</div>", unsafe_allow_html=True)

    if not player_obj:
        st.info("Search and select a player to get predictions.")
        return

    player_id = player_obj["id"]
    player_name = f"{player_obj['first_name']} {player_obj['last_name']}"

    # ------------------ Player header + image ---------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    top_col1, top_col2 = st.columns([1, 2])

    with top_col1:
        img_url = get_headshot_url(player_id, first=player_obj['first_name'], last=player_obj['last_name'])
        st.image(img_url, caption=player_name, use_column_width=True, output_format="PNG")

    with top_col2:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown(f"### {player_name}")
        st.markdown(
            f"**Team:** {player_obj['team']['full_name']} ({player_obj['team']['abbreviation']})  \n"
            f"**Position:** {player_obj.get('position') or 'N/A'}"
        )
        st.markdown(
            f"- Expected Pace Factor: `{pace_factor:.2f}`  \n"
            f"- Opponent Difficulty: `{opp_difficulty:.2f}`  \n"
            f"- Monte Carlo Sims: `{DEFAULT_MONTE_CARLO_SIMS}`"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ Fetch stats and build models --------------------
    with st.spinner("Pulling recent games and building modelsâ¦"):
        raw_stats = get_player_stats(player_id, n_games=40)
        if raw_stats.empty:
            st.error("No stats available for this player.")
            return

        core_stats = normalize_stats_df(raw_stats)
        if core_stats.empty or len(core_stats) < 8:
            st.error("Not enough usable games for modeling. Need at least ~8-10.")
            return

        feat_df = add_rolling_features(core_stats, window=5)
        if feat_df.empty:
            st.error("Not enough data after rolling feature generation.")
            return

        # Use current pace/opponent factors when training to fit to that context
        models, residuals = train_models_for_player(
            feat_df,
            STAT_COLUMNS,
            pace_factor=pace_factor,
            opp_difficulty=opp_difficulty,
        )

        if not models:
            st.error("Could not train any models for this player.")
            return

        latest_row = feat_df.iloc[-1]
        preds, pred_stds = predict_next_game(
            models,
            residuals,
            latest_row,
            pace_factor=pace_factor,
            opp_difficulty=opp_difficulty,
        )

        sims_df = run_monte_carlo(preds, pred_stds, n_sims=DEFAULT_MONTE_CARLO_SIMS)

    # ------------------ Top-line Pts / Reb / Ast ------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='luxe-card'>", unsafe_allow_html=True)
    st.markdown("## ð® Core Line (Next Game)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Points", f"{preds.get('pts', np.nan):.1f}")
    c2.metric("Rebounds", f"{preds.get('reb', np.nan):.1f}")
    c3.metric("Assists", f"{preds.get('ast', np.nan):.1f}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ Full stat table ------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ð Full Predicted Stat Line")

    stat_rows = []
    for stat in STAT_COLUMNS:
        if stat in preds:
            draws = sims_df[stat]
            summary = summarize_distribution(draws)
            stat_rows.append(
                {
                    "Stat": stat.upper(),
                    "Mean": f"{summary['mean']:.2f}",
                    "Std": f"{summary['std']:.2f}",
                    "P25": f"{summary['p25']:.1f}",
                    "Median": f"{summary['p50']:.1f}",
                    "P75": f"{summary['p75']:.1f}",
                    "P90": f"{summary['p90']:.1f}",
                }
            )

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        st.dataframe(stat_df, use_container_width=True)
    else:
        st.write("No modeled stats found.")

    # ------------------ Monte Carlo distribution viewer ---------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ð² Monte Carlo Distribution Explorer")

    if preds:
        stat_options = list(preds.keys())
        chosen_stat = st.selectbox(
            "Choose a stat to view its probability distribution",
            stat_options,
            index=stat_options.index("pts") if "pts" in stat_options else 0,
        )

        draws = sims_df[chosen_stat]
        summary = summarize_distribution(draws)

        c4, c5, c6 = st.columns(3)
        c4.metric(f"{chosen_stat.upper()} Mean", f"{summary['mean']:.2f}")
        c5.metric("Std Dev", f"{summary['std']:.2f}")
        c6.metric("P90", f"{summary['p90']:.1f}")

        st.bar_chart(draws)  # quick visual; can be upgraded to density later

    # ------------------ Recent form section -----------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ð Recent Form (Last 10 Games)")

    last10 = core_stats.tail(10)
    st.line_chart(last10[["pts", "reb", "ast"]])

    # ------------------ Save player + compare table ---------------------
    st.markdown("<br>", unsafe_allow_html=True)
    col_save, col_compare = st.columns([1, 3])

    with col_save:
        if st.button("â­ Save Player & Line", key="save_player"):
            save_player_prediction(player_name, preds)
            st.success("Player saved. Check sidebar for list.")

    with col_compare:
        if st.session_state["saved_players"]:
            st.markdown("#### ð Saved Players Comparison")
            comp_df = pd.DataFrame(st.session_state["saved_players"])
            st.dataframe(comp_df, use_container_width=True)


if __name__ == "__main__":
    main()

