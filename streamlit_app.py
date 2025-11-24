# ============================================================
# SECTION 1 — IMPORTS, CONFIG, STYLING, CONSTANTS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from scipy.stats import norm
from datetime import datetime
from functools import lru_cache

# ============================================================
# CONFIG
# ============================================================

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.set_page_config(
    page_title="NBA Luxe AI Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# MOBILE-OPTIMIZED LUXE THEME
# ============================================================

LUXE_CSS = """
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: radial-gradient(circle at top, #3b0066 0%, #050009 45%, #020006 100%) !important;
    color: #f5f2ff !important;
}

/* MOBILE PADDING */
.block-container {
    padding-top: 0.3rem !important;
    padding-left: 0.8rem !important;
    padding-right: 0.8rem !important;
}

/* HEADERS */
h1, h2, h3, h4 {
    color: #f4e8ff !important;
    font-weight: 700 !important;
}

/* CARDS */
.luxe-card {
    background: rgba(10, 2, 20, 0.92);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(150, 80, 200, 0.75);
    box-shadow: 0 0 24px rgba(155, 89, 182, 0.45);
    margin-bottom: 0.7rem;
}

.soft-card {
    background: rgba(18, 8, 30, 0.92);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    border: 1px solid rgba(150, 80, 200, 0.45);
    margin-bottom: 0.7rem;
}

/* METRIC TEXT */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.7rem !important;
    font-weight: 900 !important;
}

[data-testid="stMetricLabel"] {
    color: #cdb6ff !important;
    font-size: 0.9rem !important;
}

/* INPUTS */
.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background: rgba(25, 15, 40, 0.92) !important;
    border-radius: 14px !important;
    color: #fff !important;
    border: 1px solid rgba(155, 89, 182, 0.7) !important;
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(90deg, #8e44ad, #9b59b6);
    border-radius: 999px;
    border: none;
    padding: 0.45rem 1.3rem;
    font-weight: 600;
    color: white;
}

.stButton>button:hover {
    box-shadow: 0 0 16px rgba(155, 89, 182, 0.85);
}

/* IMAGES */
.player-image {
    width: 100%;
    border-radius: 18px;
    border: 2px solid rgba(155, 89, 182, 0.8);
    box-shadow: 0 0 18px rgba(155, 89, 182, 0.45);
}

</style>
"""

st.markdown(LUXE_CSS, unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

STAT_COLUMNS = [
    "pts", "reb", "ast",
    "stl", "blk", "turnover",
    "fgm", "fga", "fg3m", "fg3a",
    "ftm", "fta",
    "oreb", "dreb", "pf"
]

DEFAULT_MC_SIMS = 5000

# ============================================================
# SECTION 2 — API HELPERS, OPPONENT DETECTION, DEF RATING ENGINE
# ============================================================

# -----------------------------
# Basic API GET wrapper
# -----------------------------
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


# ---------------------------------------
# Player search (autocomplete)
# ---------------------------------------
@st.cache_data(ttl=600)
def search_players(query: str):
    if not query or len(query.strip()) < 2:
        return []
    data = api_get("players", {"search": query.strip(), "per_page": 25})
    return data.get("data", [])


# ---------------------------------------
# Player recent stats
# ---------------------------------------
@st.cache_data(ttl=600)
def get_player_stats(player_id: int, n_games: int = 40):
    params = {
        "player_ids[]": player_id,
        "per_page": n_games
    }
    data = api_get("stats", params)
    stats = data.get("data", [])
    if not stats:
        return pd.DataFrame()
    return pd.json_normalize(stats)


# ---------------------------------------
# Teams list
# ---------------------------------------
@st.cache_data(ttl=600)
def get_teams():
    data = api_get("teams", {"per_page": 50})
    return data.get("data", [])


# ---------------------------------------
# Get next opponent team for a player
# ---------------------------------------
@st.cache_data(ttl=600)
def get_next_opponent(player_team_id: int):
    """
    Pull the upcoming schedule, return opponent team ID + abbreviation.
    """
    today = datetime.today().strftime("%Y-%m-%d")

    schedule = api_get(
        "games",
        {"start_date": today, "per_page": 50}
    ).get("data", [])

    for g in schedule:
        home = g["home_team"]["id"]
        visitor = g["visitor_team"]["id"]

        if home == player_team_id:
            opp = g["visitor_team"]
            return opp["id"], opp["abbreviation"]

        if visitor == player_team_id:
            opp = g["home_team"]
            return opp["id"], opp["abbreviation"]

    return None, None  # no next game found


# ============================================================
# DEFENSIVE RATING ENGINE (2025–2026)
# ============================================================

# ---------------------------------------
# SCRAPE DEF RATING FROM BBREF (2026)
# ---------------------------------------
@st.cache_data(ttl=3600)
def scrape_bbref_def_rtg():
    """
    Tries to scrape DRtg from Basketball-Reference’s 2026 team page.
    If not yet available, return empty → fallback will fill.
    """
    url = "https://www.basketball-reference.com/leagues/NBA_2026.html"

    try:
        table = pd.read_html(url, match="Team Per 100 Poss")[0]
        table.columns = table.columns.droplevel(0)

        table["Team"] = table["Team"].str.replace("*", "", regex=False)
        table = table[["Team", "DRtg"]]

        table["DRtg"] = pd.to_numeric(table["DRtg"], errors="coerce")
        table = table.dropna()

        # map team names → DRtg
        return dict(zip(table["Team"], table["DRtg"]))

    except Exception:
        return {}  # will fallback


# ---------------------------------------
# FETCH LIVE DEFENSIVE COMPONENTS (BallDontLie)
# ---------------------------------------
@st.cache_data(ttl=1800)
def fetch_team_defense_components(season=2025):
    """
    Uses BallDontLie team stats to compute:
    - Points allowed per 100 possessions
    - eFG% allowed
    - 3PT defense
    - Pace normalization
    Creates a composite defensive score (scaled similarly to DRtg).
    """

    data = api_get("team_stats", {"season": season, "per_page": 50})

    df = pd.json_normalize(data.get("data", []))
    if df.empty:
        return {}

    df["opp_pts"] = df["opponent.points"]
    df["opp_fga"] = df["opponent.field_goals_attempted"]
    df["opp_fgm"] = df["opponent.field_goals_made"]
    df["opp_fg3a"] = df["opponent.three_point_attempts"]
    df["opp_fg3m"] = df["opponent.three_point_made"]
    df["pace"] = df["possessions"] / df["games_played"]

    df["efg_allowed"] = (df["opp_fgm"] + 0.5 * df["opp_fg3m"]) / df["opp_fga"]

    # composite score (scaled to DRtg range)
    df["composite"] = (
        100 * (df["opp_pts"] / df["possessions"]) * 0.55 +
        df["efg_allowed"] * 50 * 0.30 +
        df["pace"] * 1.2 * 0.15
    )

    df = df[["team.abbreviation", "composite"]]

    return dict(zip(df["team.abbreviation"], df["composite"]))


# ---------------------------------------
# BUILD FINAL 2025–26 DEF RATING TABLE
# ---------------------------------------
@st.cache_data(ttl=1800)
def build_def_rtg_2026():
    """
    Final DRtg table uses:
    - BBRef DRtg (if available)
    - Composite DRtg (from BallDontLie)
    - Traditional DRtg (computed live)
    Then averaged (C-choice).
    """
    bbref = scrape_bbref_def_rtg()
    comp = fetch_team_defense_components()

    final = {}

    for team, comp_val in comp.items():
        trad = comp_val  # comp based scaling
        if team in bbref:
            avg = (bbref[team] + comp_val + trad) / 3
        else:
            avg = (comp_val + trad) / 2

        final[team] = float(avg)

    return final


# ---------------------------------------
# GET DEF RATING FOR OPPONENT
# ---------------------------------------
def get_opp_def_rating(abbr: str):
    table = build_def_rtg_2026()
    return table.get(abbr, 113.0)  # default average defense


# ---------------------------------------
# HEADSHOT URL FIX
# ---------------------------------------
def get_headshot_url(player_id: int) -> str:
    """
    Balldontlie’s new CDN headshot format.
    """
    return f"https://balldontlie.io/images/headshots/{player_id}.png"

# ============================================================
# SECTION 3 — FEATURE ENGINEERING + XGBOOST MODEL ENGINE
# ============================================================

# Stats modeled
STAT_COLUMNS = [
    "pts", "reb", "ast", "stl", "blk",
    "turnover", "fgm", "fga", "fg3m", "fg3a",
    "ftm", "fta", "oreb", "dreb", "pf"
]


# ---------------------------------------
# Normalize incoming BDL stats DF
# ---------------------------------------
def normalize_stats_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts base stats from nested JSON fields.
    Maps fields like 'stats.pts' → pts.
    """
    stat_map = {}
    for stat in STAT_COLUMNS:
        candidates = [c for c in df_raw.columns if c.split(".")[-1] == stat]
        if candidates:
            stat_map[stat] = candidates[0]

    if not stat_map:
        return pd.DataFrame()

    df = pd.DataFrame()
    for stat, col in stat_map.items():
        df[stat] = pd.to_numeric(df_raw[col], errors="coerce")

    return df.dropna(how="all")


# ---------------------------------------
# Rolling averages & form indicators
# ---------------------------------------
def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    feat_df = df.copy()

    for col in STAT_COLUMNS:
        if col in feat_df.columns:
            feat_df[f"{col}_roll_{window}"] = feat_df[col].rolling(window).mean()
            feat_df[f"{col}_prev"] = feat_df[col].shift(1)

    feat_df = feat_df.dropna()
    return feat_df


# ---------------------------------------
# Add opponent DRtg + pace factor
# ---------------------------------------
def add_context_features(df: pd.DataFrame, pace_factor: float, opp_dr: float):
    out = df.copy()
    out["pace_factor"] = pace_factor
    out["opp_dr"] = opp_dr
    out["opp_def_scaled"] = opp_dr / 100
    out["pace_x_dr"] = pace_factor * opp_dr
    return out


# ---------------------------------------
# Train XGBoost model per stat
# ---------------------------------------
def train_player_models(feat_df: pd.DataFrame, targets: list[str]):
    models = {}
    residuals = {}

    feature_cols = [
        c for c in feat_df.columns
        if c.endswith("_roll_5")
        or c.endswith("_prev")
        or c in ["pace_factor", "opp_dr", "opp_def_scaled", "pace_x_dr"]
    ]

    if not feature_cols:
        return {}, {}

    X = feat_df[feature_cols].values

    for stat in targets:
        if stat not in feat_df.columns:
            continue

        y = feat_df[stat].values

        # not enough variation → cannot train reliably
        if len(y) < 8 or len(np.unique(y)) <= 1:
            continue

        model = XGBRegressor(
            n_estimators=240,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=2,
        )

        model.fit(X, y)

        # residual std for Monte Carlo variance
        pred_y = model.predict(X)
        resid = y - pred_y
        resid_std = float(np.std(resid)) if len(resid) > 1 else 1.0

        models[stat] = (model, feature_cols)
        residuals[stat] = max(0.4, resid_std)  # floor variance

    return models, residuals


# ---------------------------------------
# Predict the next game
# ---------------------------------------
def predict_next_game(models: dict, residuals: dict, last_row: pd.Series,
                      pace_factor: float, opp_dr: float):
    preds = {}
    stds = {}

    base = last_row.copy()
    base["pace_factor"] = pace_factor
    base["opp_dr"] = opp_dr
    base["opp_def_scaled"] = opp_dr / 100
    base["pace_x_dr"] = pace_factor * opp_dr

    for stat, (model, cols) in models.items():
        x = np.array([base[cols].values], dtype=float)
        mu = float(model.predict(x)[0])
        preds[stat] = max(0.0, mu)
        stds[stat] = residuals[stat]

    return preds, stds

    # ============================================================
# SECTION 4 — MONTE CARLO ENGINE + BOOK LINE / EV SYSTEM
# ============================================================

DEFAULT_SIMS = 5000


# ---------------------------------------
# Monte Carlo Simulation
# ---------------------------------------
def run_monte_carlo(pred_means: dict, pred_stds: dict,
                    n_sims: int = DEFAULT_SIMS):
    """
    Each stat follows a normal approximation with learned mean & std.
    Returns DataFrame: sims x stats.
    """
    sims = {}

    for stat, mu in pred_means.items():
        sigma = max(0.4, pred_stds.get(stat, 1.0))
        draws = np.random.normal(mu, sigma, size=n_sims)
        draws = np.clip(draws, 0, None)
        sims[stat] = draws

    return pd.DataFrame(sims)


# ---------------------------------------
# Summaries (for tables / cards)
# ---------------------------------------
def summarize_dist(arr: np.ndarray):
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


# ---------------------------------------
# Probability of clearing any line
# ---------------------------------------
def prob_clear(draws: np.ndarray, line: float):
    """
    P(X > line)
    """
    return float((draws > line).mean())


# ---------------------------------------
# Convert American odds to implied prob
# ---------------------------------------
def implied_prob_from_american(odds: int):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


# ---------------------------------------
# Edge & Expected Value
# ---------------------------------------
def compute_ev(prob_over: float, american_odds: int):
    """
    EV = (probability you win * payout) - (probability you lose * risk)
    """
    if american_odds > 0:
        payout = american_odds / 100
    else:
        payout = 100 / (-american_odds)

    risk = 1.0

    win_ev = prob_over * payout
    lose_ev = (1 - prob_over) * risk

    return win_ev - lose_ev


# ---------------------------------------
# Full betting evaluation wrapper
# ---------------------------------------
def evaluate_prop(draws: np.ndarray, line: float, odds: int):
    """
    Returns:
        prob_over, edge, ev
    """
    p = prob_clear(draws, line)
    implied = implied_prob_from_american(odds)
    edge = p - implied
    ev = compute_ev(p, odds)

    return {
        "prob_over": round(p, 4),
        "implied_prob": round(implied, 4),
        "edge": round(edge, 4),
        "ev": round(ev, 4),
    }

# ============================================================
# SECTION 5 — FULL UI + MODEL + MONTE CARLO + EV DASHBOARD
# ============================================================

def render_player_header(player_obj, player_name, img_url,
                         pace_factor, opp_dr, opp_abbr):
    """
    Renders the player card at the top of the screen.
    Mobile-friendly (stack on small screens).
    """
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img_url, use_column_width=True)
    
    with col2:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown(f"### {player_name}")
        st.markdown(
            f"**Team:** {player_obj['team']['full_name']} ({player_obj['team']['abbreviation']})  \n"
            f"**Position:** {player_obj.get('position','N/A')}  \n"
            f"**Next Opponent:** {opp_abbr}  \n"
            f"**Opponent DRtg:** `{opp_dr:.1f}`  \n"
            f"**Pace Factor Applied:** `{pace_factor:.2f}`"
        )
        st.markdown("</div>", unsafe_allow_html=True)



def render_core_line(preds):
    st.markdown("<div class='luxe-card'>", unsafe_allow_html=True)
    st.markdown("## 🔮 Core Line Predictions")

    c1, c2, c3 = st.columns(3)
    c1.metric("Points", f"{preds.get('pts', 0):.1f}")
    c2.metric("Rebounds", f"{preds.get('reb', 0):.1f}")
    c3.metric("Assists", f"{preds.get('ast', 0):.1f}")

    st.markdown("</div>", unsafe_allow_html=True)



def render_full_table(sims_df):
    st.markdown("### 📊 Full Projected Stat Line")

    rows = []
    for stat in sims_df.columns:
        s = summarize_dist(sims_df[stat])
        rows.append({
            "Stat": stat.upper(),
            "Mean": f"{s['mean']:.2f}",
            "Std": f"{s['std']:.2f}",
            "P25": f"{s['p25']:.1f}",
            "Median": f"{s['p50']:.1f}",
            "P75": f"{s['p75']:.1f}",
            "P90": f"{s['p90']:.1f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)



def render_prop_betting_ui(sims_df):
    st.markdown("<br>")
    st.markdown("## 💵 Prop Betting Evaluation")

    stat_choice = st.selectbox(
        "Choose a stat to evaluate",
        sims_df.columns.tolist(),
        index=0
    )

    line = st.number_input(f"{stat_choice.upper()} Line", value=20.5, step=0.5)
    odds = st.number_input("American Odds", value=-115, step=1)

    draws = sims_df[stat_choice].values
    result = evaluate_prop(draws, line, odds)

    st.markdown("### 📈 Probability & EV")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prob Over", f"{result['prob_over']*100:.1f}%")
    c2.metric("Edge", f"{result['edge']*100:.1f}%")
    c3.metric("EV", f"{result['ev']:.3f}")

    st.markdown("---")



def render_distribution_viewer(sims_df):
    st.markdown("### 🎲 Distribution Explorer")

    stat = st.selectbox(
        "Select stat distribution to view",
        sims_df.columns.tolist(),
        index=0
    )

    st.bar_chart(sims_df[stat])



def render_recent_form(core_stats):
    st.markdown("### 📈 Recent Form (Last 10 Games)")

    last10 = core_stats.tail(10)
    subset = last10[["pts", "reb", "ast"]]
    st.line_chart(subset)



# ============================================================
# MAIN APP WORKFLOW
# ============================================================

def main():
    st.markdown(
        "<h1 style='text-align:center;'>🏀 NBA Luxe AI Predictor</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; color:#d9c0ff;'>Fully automated DRtg integration • XGBoost • Monte Carlo • EV Engine</p>",
        unsafe_allow_html=True,
    )

    # --- Player Search ---
    query = st.text_input("Search for a player", value="", placeholder="Start typing…")

    players = search_players(query) if len(query) >= 2 else []
    if players:
        names = [f"{p['first_name']} {p['last_name']} ({p['team']['abbreviation']})"
                 for p in players]
        idx = st.selectbox("Select player", range(len(names)), format_func=lambda i: names[i])
        player_obj = players[idx]
    else:
        if len(query) >= 2:
            st.warning("No players found.")
        return

    # Extract metadata
    player_id = player_obj["id"]
    player_name = f"{player_obj['first_name']} {player_obj['last_name']}"
    team_id = player_obj["team"]["id"]
    img_url = get_headshot_url(player_id)

    # --- AUTO OPPONENT DETECTION ---
    opp_id, opp_abbr = get_next_opponent(team_id)
    if opp_id is None:
        st.error("Could not determine next opponent.")
        return

    # --- AUTO DEF RATING ---
    opp_dr = get_opp_def_rating(opp_abbr)

    # --- Pace slider ---
    pace_factor = st.slider("Pace Adjustment", 0.85, 1.20, 1.00, 0.01)

    # --- Fetch recent stats ---
    raw_stats = get_player_stats(player_id, n_games=40)
    core_stats = normalize_stats_df(raw_stats)
    feat_df = add_rolling_features(core_stats, window=5)
    feat_df = add_context_features(feat_df, pace_factor, opp_dr)

    # --- Train models ---
    models, residuals = train_player_models(feat_df, STAT_COLUMNS)
    latest = feat_df.iloc[-1]

    preds, pred_stds = predict_next_game(models, residuals, latest,
                                         pace_factor, opp_dr)

    # --- Run MC ---
    sims_df = run_monte_carlo(preds, pred_stds)

    # --- RENDER UI ---
    render_player_header(player_obj, player_name, img_url,
                         pace_factor, opp_dr, opp_abbr)

    st.markdown("<br>", unsafe_allow_html=True)
    render_core_line(preds)

    st.markdown("<br>", unsafe_allow_html=True)
    render_full_table(sims_df)

    st.markdown("<br>", unsafe_allow_html=True)
    render_prop_betting_ui(sims_df)

    st.markdown("<br>", unsafe_allow_html=True)
    render_distribution_viewer(sims_df)

    st.markdown("<br>", unsafe_allow_html=True)
    render_recent_form(core_stats)


# Run app
if __name__ == "__main__":
    main()
# ============================================================
# SECTION 6 — MOBILE OPTIMIZATION + UI POLISH
# ============================================================

MOBILE_CSS = """
<style>

    /* Mobile responsive layout ---------------------------------------- */
    @media (max-width: 900px) {

        /* Reduce global padding */
        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 1rem !important;
        }

        h1 {
            font-size: 1.7rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
        }

        h3 {
            font-size: 1.2rem !important;
        }

        .luxe-card, .soft-card {
            padding: 0.9rem !important;
        }

        /* Metrics adjust */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }

        /* Make sidebar collapsible on mobile */
        section[data-testid="stSidebar"] {
            width: 70% !important;
        }

        /* Fix bar chart overflow */
        .element-container {
            width: 100% !important;
        }
    }


    /* Fix Streamlit annoying padding ----------------------------- */
    .block-container {
        padding-top: 1rem !important;
    }

    /* Reduce bottom padding */
    footer {
        visibility: hidden;
    }

    /* Clean scrollbar for mobile */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(155,89,182,0.35);
        border-radius: 10px;
    }

    /* Card hover polish */
    .luxe-card:hover {
        box-shadow: 0 0 35px rgba(155,89,182,0.55);
        transition: 0.25s;
    }

    /* Make metric cards more compact on mobile */
    div[data-testid="stMetric"] {
        padding-top: 0.1rem;
        padding-bottom: 0.1rem;
    }

</style>
"""

st.markdown(MOBILE_CSS, unsafe_allow_html=True)



# ------------------------------
# Collapsible Sections Helper
# ------------------------------

def collapsible(title, content_block):
    """
    Wrap any UI block in a collapsible section.
    """
    with st.expander(title):
        content_block()
