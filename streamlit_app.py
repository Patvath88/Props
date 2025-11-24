
import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
import altair as alt

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE_URL = "https://api.balldontlie.io/v1"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

st.set_page_config(page_title="NBA AI Predictor", layout="wide")


# -------------------------------------------------------------------
# CACHED REQUEST FUNCTION (Prevents Rate Limits)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch(endpoint: str, params: dict | None = None) -> dict:
    """Perform a GET request to the BallDontLie API with caching.

    Parameters
    ----------
    endpoint : str
        The API endpoint (e.g. "players", "stats").
    params : dict | None
        Query string parameters for the request.

    Returns
    -------
    dict
        The JSON response parsed into a dictionary. If an error occurs,
        returns a dict with an "error" key.
    """
    try:
        r = requests.get(
            f"{BASE_URL}/{endpoint}",
            params=params,
            headers=HEADERS,
            timeout=8
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "data": []}


# -------------------------------------------------------------------
# PLAYER SEARCH
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def get_player_id(name: str) -> tuple[int | None, dict | None]:
    """Retrieve a player's ID by searching their name.

    Parameters
    ----------
    name : str
        The full or partial name of the player.

    Returns
    -------
    tuple
        A tuple containing the player's ID and the raw player
        information dict. Returns (None, None) if no player is found.
    """
    params = {"search": name, "per_page": 50, "page": 1}
    data = fetch("players", params)

    if "data" in data and len(data["data"]) > 0:
        p = data["data"][0]
        return p["id"], p
    return None, None


# -------------------------------------------------------------------
# STATS FETCH (Last 30 Games)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def get_recent_stats(player_id: int) -> pd.DataFrame:
    """Fetch the last 30 games of basic stats for a given player.

    Parameters
    ----------
    player_id : int
        The BallDontLie player ID.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the last 30 game logs with columns for
        points, assists and rebounds. Returns an empty DataFrame on
        error or if no data is found.
    """
    stats = fetch(
        "stats",
        {"player_ids[]": player_id, "per_page": 30}
    )

    if "data" not in stats or len(stats["data"]) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(stats["data"])
    # Only keep the columns we care about. The API returns nested
    # dictionaries for player and team; we extract numeric stats only.
    return df[["pts", "ast", "reb"]].copy()


# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling means for points, assists and rebounds.

    The model uses a 5‑game rolling average for each stat to predict
    the next game’s performance. Rows with insufficient history are
    dropped.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns "pts", "ast", and "reb".

    Returns
    -------
    pandas.DataFrame
        DataFrame with added columns "pts_roll", "ast_roll", "reb_roll"
        and dropped NaNs.
    """
    df = df.copy()
    df["pts_roll"] = df["pts"].rolling(5).mean()
    df["ast_roll"] = df["ast"].rolling(5).mean()
    df["reb_roll"] = df["reb"].rolling(5).mean()
    df = df.dropna()
    return df


# -------------------------------------------------------------------
# TRAIN XGBOOST MODEL
# -------------------------------------------------------------------
def train_model(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """Train a lightweight XGBoost regressor on the provided data.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable.

    Returns
    -------
    XGBRegressor
        A trained XGBoost model.
    """
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
    return model


# -------------------------------------------------------------------
# PROBABILITY CALCULATION
# -------------------------------------------------------------------
def compute_threshold_probability(values: pd.Series, threshold: float) -> float:
    """Compute the probability that a stat meets or exceeds a given threshold.

    This helper calculates the fraction of historical observations in
    `values` that are greater than or equal to `threshold`. If no
    observations are present, returns 0.0.

    Parameters
    ----------
    values : pandas.Series
        Historical observations of a given stat.
    threshold : float
        The user‑defined threshold to evaluate.

    Returns
    -------
    float
        The estimated probability (between 0 and 1).
    """
    if len(values) == 0:
        return 0.0
    return float((values >= threshold).mean())


# -------------------------------------------------------------------
# TRENDING MESSAGE
# -------------------------------------------------------------------
def get_trend_message(predicted: float, mean: float, recent_mean: float) -> str:
    """Generate a human‑readable trend message.

    Compares the predicted value against the overall mean and the last
    five‑game mean to indicate whether a player is trending up or
    down. The thresholds are heuristic; adjust as desired.

    Parameters
    ----------
    predicted : float
        The model’s predicted value for the next game.
    mean : float
        The mean of the last 30 games.
    recent_mean : float
        The mean of the last five games.

    Returns
    -------
    str
        A descriptive message about the trend.
    """
    diff_mean = predicted - mean
    diff_recent = predicted - recent_mean

    # Determine the trend direction and magnitude
    def classify(diff: float) -> str:
        if abs(diff) < 1.0:
            return "around his average"
        if diff > 0:
            return "upward"
        return "downward"

    parts = []
    # Compare to overall mean
    if abs(diff_mean) >= 1.0:
        direction = "higher" if diff_mean > 0 else "lower"
        parts.append(f"{abs(diff_mean):.1f} {direction} than his 30‑game average")
    # Compare to recent mean
    if abs(diff_recent) >= 1.0:
        direction = "higher" if diff_recent > 0 else "lower"
        parts.append(f"{abs(diff_recent):.1f} {direction} than his last 5‑game average")

    if not parts:
        return "This prediction is roughly in line with recent performance."
    return "; ".join(parts) + "."


# -------------------------------------------------------------------
# CHART GENERATOR
# -------------------------------------------------------------------
def build_distribution_chart(values: pd.Series, predicted_value: float, threshold: float, stat_name: str) -> alt.Chart:
    """Create a histogram with overlay lines for prediction and threshold.

    Parameters
    ----------
    values : pandas.Series
        Historical observations for a given statistic.
    predicted_value : float
        The model’s predicted value for the next game.
    threshold : float
        The user‑selected threshold to evaluate.
    stat_name : str
        The name of the statistic (e.g. "Points").

    Returns
    -------
    altair.Chart
        An Altair chart object rendering the distribution, prediction line
        and threshold line.
    """
    # Prepare a DataFrame for Altair
    data = pd.DataFrame({"value": values})
    base = alt.Chart(data).mark_bar(opacity=0.6).encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title=f"{stat_name} (last 30 games)"),
        y=alt.Y('count()', title='Frequency'),
    )

    # Prediction line
    pred_line = alt.Chart(pd.DataFrame({'x': [predicted_value]})).mark_rule(color='blue').encode(x='x:Q').properties(
        title=f"Distribution of {stat_name}"
    )

    # Threshold line
    thresh_line = alt.Chart(pd.DataFrame({'x': [threshold]})).mark_rule(color='orange', strokeDash=[4,2]).encode(x='x:Q')

    return (base + pred_line + thresh_line).configure_axis(labelFontSize=11, titleFontSize=13)


# -------------------------------------------------------------------
# MAIN STREAMLIT APP
# -------------------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🏀 NBA AI Predictor</h1>
    <p style='text-align:center;'>Enhanced Rolling XGBoost Model with Prop Insights</p>
    """,
    unsafe_allow_html=True,
)

player_name = st.text_input("Enter NBA Player Name", "", key="player_input")

if player_name:
    st.write(" ")

    # ---------------------------------------------------------------
    # SEARCH PLAYER
    # ---------------------------------------------------------------
    with st.spinner("Searching player…"):
        player_id, player_info = get_player_id(player_name)

    if not player_id:
        st.error("❌ Player not found. Try full names (e.g., 'Luka Doncic').")
        st.stop()

    st.subheader(f"📌 {player_info['first_name']} {player_info['last_name']}")

    # ---------------------------------------------------------------
    # FETCH STATS
    # ---------------------------------------------------------------
    with st.spinner("Loading last 30 games…"):
        raw_df = get_recent_stats(player_id)

    if raw_df.empty:
        st.error("❌ No recent stats found for this player.")
        st.stop()

    # Keep copy for probability calculations
    hist_df = raw_df.copy()

    # Prepare features for modelling
    df = prepare_features(raw_df)
    if df.empty:
        st.error("❌ Not enough data for model (need 5+ games).")
        st.stop()

    # ML Inputs
    X = df[["pts_roll", "ast_roll", "reb_roll"]]

    # Train models
    pts_model = train_model(X, df["pts"])
    ast_model = train_model(X, df["ast"])
    reb_model = train_model(X, df["reb"])

    # Latest row for prediction
    latest = X.iloc[-1:]
    pred_pts = float(pts_model.predict(latest)[0])
    pred_ast = float(ast_model.predict(latest)[0])
    pred_reb = float(reb_model.predict(latest)[0])

    # Historical means for trend analysis
    overall_means = hist_df.mean()
    recent_means = hist_df.tail(5).mean()

    # Create tabs: Overview (existing output) and Simulation
    tab_overview, tab_sim = st.tabs(["Overview", "Simulation"])

    # -------------------------------------------------------------------
    # Overview Tab
    # -------------------------------------------------------------------
    with tab_overview:
        # User threshold inputs
        st.markdown("## 🎯 Set Your Prop Lines")
        cols_thresholds = st.columns(3)
        # Determine reasonable slider ranges based on historical data
        # Use the min and max observed values with some padding
        pts_min, pts_max = hist_df["pts"].min(), hist_df["pts"].max()
        ast_min, ast_max = hist_df["ast"].min(), hist_df["ast"].max()
        reb_min, reb_max = hist_df["reb"].min(), hist_df["reb"].max()

        with cols_thresholds[0]:
            pts_threshold = st.slider(
                "Points Line", min_value=float(max(0, pts_min - 5)), max_value=float(pts_max + 5),
                value=float(round(pred_pts)), step=0.5, key="pts_threshold"
            )
        with cols_thresholds[1]:
            ast_threshold = st.slider(
                "Assists Line", min_value=float(max(0, ast_min - 2)), max_value=float(ast_max + 2),
                value=float(round(pred_ast, 1)), step=0.5, key="ast_threshold"
            )
        with cols_thresholds[2]:
            reb_threshold = st.slider(
                "Rebounds Line", min_value=float(max(0, reb_min - 2)), max_value=float(reb_max + 2),
                value=float(round(pred_reb, 1)), step=0.5, key="reb_threshold"
            )

        # Compute probabilities
        prob_pts = compute_threshold_probability(hist_df["pts"], pts_threshold)
        prob_ast = compute_threshold_probability(hist_df["ast"], ast_threshold)
        prob_reb = compute_threshold_probability(hist_df["reb"], reb_threshold)

        st.markdown("## 🔮 Predicted Next Game Stats")
        c_pred = st.columns(3)
        c_pred[0].metric(
            "Points", f"{pred_pts:.1f}", help=get_trend_message(pred_pts, overall_means["pts"], recent_means["pts"])
        )
        c_pred[1].metric(
            "Assists", f"{pred_ast:.1f}", help=get_trend_message(pred_ast, overall_means["ast"], recent_means["ast"])
        )
        c_pred[2].metric(
            "Rebounds", f"{pred_reb:.1f}", help=get_trend_message(pred_reb, overall_means["reb"], recent_means["reb"])
        )

        st.markdown("## 📊 Probability of Clearing Your Line")
        c_prob = st.columns(3)
        c_prob[0].metric("P(Points ≥ Line)", f"{prob_pts*100:.0f}%")
        c_prob[1].metric("P(Assists ≥ Line)", f"{prob_ast*100:.0f}%")
        c_prob[2].metric("P(Rebounds ≥ Line)", f"{prob_reb*100:.0f}%")

        st.markdown("---")
        st.markdown("### 📈 Recent Trends")
        # Show last 10 games trends for basic stats
        st.line_chart(hist_df[["pts", "ast", "reb"]].tail(10))

        # Summarise last five games vs 30 game means
        st.markdown("### 📋 Averages Comparison")
        comparison_data = pd.DataFrame({
            "Stat": ["Points", "Assists", "Rebounds"],
            "30‑Game Avg": [overall_means["pts"], overall_means["ast"], overall_means["reb"]],
            "Last 5 Avg": [recent_means["pts"], recent_means["ast"], recent_means["reb"]],
            "Prediction": [pred_pts, pred_ast, pred_reb],
        })
        st.dataframe(
            comparison_data.set_index("Stat").style.format("{:.1f}"),
            use_container_width=True,
        )

        st.markdown("### 🧮 Distributions and Lines")
        # Display distribution charts for each stat
        dist_cols = st.columns(3)
        with dist_cols[0]:
            chart = build_distribution_chart(hist_df["pts"], pred_pts, pts_threshold, "Points")
            st.altair_chart(chart, use_container_width=True)
        with dist_cols[1]:
            chart = build_distribution_chart(hist_df["ast"], pred_ast, ast_threshold, "Assists")
            st.altair_chart(chart, use_container_width=True)
        with dist_cols[2]:
            chart = build_distribution_chart(hist_df["reb"], pred_reb, reb_threshold, "Rebounds")
            st.altair_chart(chart, use_container_width=True)

        st.markdown("### 📊 Model Inputs (Latest Row)")
        st.dataframe(latest, use_container_width=True)

    # -------------------------------------------------------------------
    # Simulation Tab
    # -------------------------------------------------------------------
    with tab_sim:
        st.markdown("## 🎲 Monte Carlo Simulation")
        st.write(
            "Run a high‑volume simulation of the next game’s performance. "
            "Select a statistic and prop line to estimate how often the player "
            "would clear that line if the distribution holds."
        )

        # Select statistic to simulate
        stat_choice = st.selectbox("Select Statistic", ["Points", "Assists", "Rebounds"], key="sim_stat")

        # Map selection to data
        stat_map = {
            "Points": (hist_df["pts"], pred_pts),
            "Assists": (hist_df["ast"], pred_ast),
            "Rebounds": (hist_df["reb"], pred_reb),
        }
        values, predicted_value = stat_map[stat_choice]

        # Determine slider range for simulation line
        sim_min = float(max(0, values.min() - 5))
        sim_max = float(values.max() + 5)
        default_line = float(round(predicted_value))

        sim_line = st.slider(
            f"{stat_choice} Line", min_value=sim_min, max_value=sim_max,
            value=default_line, step=0.5, key="sim_line"
        )

        # Button to trigger simulation
        if st.button("Run 1M Simulations", key="run_sim"):
            with st.spinner("Simulating…"):
                # Adjust the distribution to centre on the predicted mean
                adjusted = values + (predicted_value - values.mean())
                sims = np.random.choice(adjusted, size=1_000_000, replace=True)
                prob_hit = float((sims >= sim_line).mean())

            st.success(
                f"Estimated probability of {stat_choice} ≥ {sim_line:.1f}: {prob_hit * 100:.1f}%"
            )
            # Show a histogram of simulated values with overlay lines
            sim_chart = build_distribution_chart(pd.Series(sims), predicted_value, sim_line, stat_choice)
            st.altair_chart(sim_chart, use_container_width=True)
