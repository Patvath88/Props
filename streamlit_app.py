
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from xgboost import XGBRegressor

# ============================================================
# SECTION 1 — CONFIG & BASE SETUP
# ============================================================

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.set_page_config(page_title="NBA Luxe AI Predictor", layout="wide")

# ============================================================
# SECTION 6 — MOBILE + POLISH CSS
# ============================================================

MOBILE_CSS = """
<style>
.main .block-container {padding-top:1rem!important;}
@media (max-width:900px){
 .main .block-container{padding-left:.5rem!important;padding-right:.5rem!important;}
 h1{font-size:1.7rem!important;}
 .luxe-card,.soft-card{padding:.9rem!important;}
}
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)

# ============================================================
# SECTION 2 — API HELPERS + DEF RATING ENGINE
# ============================================================

def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error":str(e),"data":[]}

@st.cache_data(ttl=600)
def search_players(q):
    if not q or len(q)<2: return []
    return api_get("players",{"search":q,"per_page":25}).get("data",[])

@st.cache_data(ttl=600)
def get_player_stats(pid, n=40):
    data = api_get("stats",{"player_ids[]":pid,"per_page":n}).get("data",[])
    return pd.json_normalize(data) if data else pd.DataFrame()

@st.cache_data(ttl=600)
def get_teams():
    return api_get("teams",{"per_page":50}).get("data",[])

@st.cache_data(ttl=600)
def get_next_opponent(team_id:int):
    today = datetime.today().strftime("%Y-%m-%d")
    games = api_get("games",{"start_date":today,"per_page":50}).get("data",[])
    for g in games:
        if g["home_team"]["id"]==team_id:
            v=g["visitor_team"]; return v["id"],v["abbreviation"]
        if g["visitor_team"]["id"]==team_id:
            h=g["home_team"]; return h["id"],h["abbreviation"]
    return None,None

@st.cache_data(ttl=3600)
def scrape_bbref_def():
    try:
        url="https://www.basketball-reference.com/leagues/NBA_2026.html"
        tbl=pd.read_html(url,match="Team Per 100 Poss")[0]
        tbl.columns=tbl.columns.droplevel(0)
        tbl["Team"]=tbl["Team"].str.replace("*","",regex=False)
        tbl=tbl[["Team","DRtg"]]
        tbl["DRtg"]=pd.to_numeric(tbl["DRtg"],errors="coerce")
        return dict(zip(tbl["Team"],tbl["DRtg"]))
    except:
        return {}

@st.cache_data(ttl=1800)
def fetch_def_components(season=2025):
    data=api_get("team_stats",{"season":season,"per_page":50}).get("data",[])
    df=pd.json_normalize(data)
    if df.empty: return {}
    df["opp_pts"]=df["opponent.points"]
    df["opp_fga"]=df["opponent.field_goals_attempted"]
    df["opp_fgm"]=df["opponent.field_goals_made"]
    df["opp_fg3a"]=df["opponent.three_point_attempts"]
    df["opp_fg3m"]=df["opponent.three_point_made"]
    df["pace"]=df["possessions"]/df["games_played"]
    df["efg_allowed"]=(df["opp_fgm"]+0.5*df["opp_fg3m"])/df["opp_fga"]
    df["comp"]=100*(df["opp_pts"]/df["possessions"])*0.55 + df["efg_allowed"]*50*0.30 + df["pace"]*1.2*0.15
    return dict(zip(df["team.abbreviation"],df["comp"]))

@st.cache_data(ttl=1800)
def build_def_table():
    bb= scrape_bbref_def()
    comp = fetch_def_components()
    out={}
    for t,v in comp.items():
        if t in bb: out[t]=(bb[t]+v+v)/3
        else: out[t]=(v+v)/2
    return out

def get_opp_def_rating(abbr:str):
    return build_def_table().get(abbr,113.0)

def get_headshot_url(pid:int):
    return f"https://balldontlie.io/images/headshots/{pid}.png"

# ============================================================
# SECTION 3 — FEATURE ENGINEERING + XGB MODELS
# ============================================================

STAT_COLUMNS=["pts","reb","ast","stl","blk","turnover","fgm","fga","fg3m","fg3a","ftm","fta","oreb","dreb","pf"]

def normalize_stats_df(df):
    smap={}
    for s in STAT_COLUMNS:
        cand=[c for c in df.columns if c.split(".")[-1]==s]
        if cand: smap[s]=cand[0]
    if not smap: return pd.DataFrame()
    out=pd.DataFrame()
    for s,c in smap.items(): out[s]=pd.to_numeric(df[c],errors="coerce")
    return out.dropna(how="all")

def add_rolling_features(df,window=5):
    f=df.copy()
    for c in STAT_COLUMNS:
        if c in f:
            f[f"{c}_roll_{window}"]=f[c].rolling(window).mean()
            f[f"{c}_prev"]=f[c].shift(1)
    return f.dropna()

def add_context(df,pace,dr):
    f=df.copy()
    f["pace_factor"]=pace
    f["opp_dr"]=dr
    f["opp_scaled"]=dr/100
    f["pace_x_dr"]=pace*dr
    return f

def train_player_models(df,targets):
    models,resid={},{}
    feat=[c for c in df.columns if c.endswith("_roll_5") or c.endswith("_prev") or c in ["pace_factor","opp_dr","opp_scaled","pace_x_dr"]]
    if not feat: return {},{}
    X=df[feat].values
    for s in targets:
        if s not in df: continue
        y=df[s].values
        if len(y)<8 or len(np.unique(y))<=1: continue
        m=XGBRegressor(
            n_estimators=240,learning_rate=0.05,max_depth=4,
            subsample=0.9,colsample_bytree=0.9,objective="reg:squarederror",n_jobs=2
        )
        m.fit(X,y)
        pr=m.predict(X)
        r=y-pr
        resid_std=float(np.std(r)) if len(r)>1 else 1.0
        models[s]=(m,feat); resid[s]=max(0.4,resid_std)
    return models,resid

def predict_next_game(models,resid,lastrow,pace,dr):
    preds,stds={},{}
    base=lastrow.copy()
    base["pace_factor"]=pace
    base["opp_dr"]=dr
    base["opp_scaled"]=dr/100
    base["pace_x_dr"]=pace*dr
    for s,(m,cols) in models.items():
        x=np.array([base[cols].values])
        mu=float(m.predict(x)[0])
        preds[s]=max(0,mu); stds[s]=resid[s]
    return preds,stds

# ============================================================
# SECTION 4 — MONTE CARLO + EV ENGINE
# ============================================================

def run_mc(preds,stds,n=5000):
    out={}
    for s,mu in preds.items():
        sigma=max(0.4,stds.get(s,1.0))
        d=np.random.normal(mu,sigma,n)
        d=np.clip(d,0,None)
        out[s]=d
    return pd.DataFrame(out)

def summarize_dist(a):
    return {"mean":float(a.mean()),"std":float(a.std()),
            "p25":float(np.percentile(a,25)),"p50":float(np.percentile(a,50)),
            "p75":float(np.percentile(a,75)),"p90":float(np.percentile(a,90))}

def prob_clear(a,line): return float((a>line).mean())

def implied_prob(odds):
    return 100/(odds+100) if odds>0 else -odds/(-odds+100)

def compute_ev(p,odds):
    payout=odds/100 if odds>0 else 100/(-odds)
    return p*payout - (1-p)*1.0

def evaluate_prop(draws,line,odds):
    p=prob_clear(draws,line); imp=implied_prob(odds); ev=compute_ev(p,odds)
    return {"prob_over":p,"implied_prob":imp,"edge":p-imp,"ev":ev}

# ============================================================
# SECTION 5 — UI LAYER
# ============================================================

def render_player_header(obj,name,img,pace,dr,abbr):
    c1,c2=st.columns([1,2])
    with c1: st.image(img,use_column_width=True)
    with c2:
        st.markdown(f"### {name}")
        st.markdown(f"Team: {obj['team']['full_name']} ({obj['team']['abbreviation']})")
        st.markdown(f"Position: {obj.get('position','N/A')}")
        st.markdown(f"Next Opponent: {abbr}")
        st.markdown(f"Opponent DRtg: {dr:.1f}")
        st.markdown(f"Pace Factor: {pace:.2f}")

def render_core(preds):
    c1,c2,c3=st.columns(3)
    c1.metric("Points",f"{preds.get('pts',0):.1f}")
    c2.metric("Rebounds",f"{preds.get('reb',0):.1f}")
    c3.metric("Assists",f"{preds.get('ast',0):.1f}")

def render_table(df):
    rows=[]
    for c in df.columns:
        s=summarize_dist(df[c])
        rows.append({
            "Stat":c.upper(),"Mean":f"{s['mean']:.2f}","Std":f"{s['std']:.2f}",
            "P25":f"{s['p25']:.1f}","Median":f"{s['p50']:.1f}",
            "P75":f"{s['p75']:.1f}","P90":f"{s['p90']:.1f}"
        })
    st.dataframe(pd.DataFrame(rows),use_container_width=True)

def render_betting(df):
    st.markdown("## Prop Betting Evaluation")
    stat=st.selectbox("Stat",df.columns)
    line=st.number_input("Line",value=20.5,step=0.5)
    odds=st.number_input("Odds",value=-115,step=1)
    res=evaluate_prop(df[stat].values,line,odds)
    c1,c2,c3=st.columns(3)
    c1.metric("Prob Over",f"{res['prob_over']*100:.1f}%")
    c2.metric("Edge",f"{res['edge']*100:.1f}%")
    c3.metric("EV",f"{res['ev']:.3f}")

def render_dist(df):
    st.markdown("### Distribution")
    stat=st.selectbox("View Distribution",df.columns)
    st.bar_chart(df[stat])

def render_recent(df):
    st.markdown("### Recent Form")
    last=df.tail(10)
    st.line_chart(last[["pts","reb","ast"]])

# ============================================================
# MAIN
# ============================================================

def main():
    st.title("🏀 NBA Luxe AI Predictor")

    query=st.text_input("Search Player")
    players=search_players(query) if len(query)>=2 else []
    if not players:
        if len(query)>=2: st.warning("No players found.")
        return

    names=[f"{p['first_name']} {p['last_name']} ({p['team']['abbreviation']})" for p in players]
    idx=st.selectbox("Select Player",range(len(names)),format_func=lambda i:names[i])
    obj=players[idx]; pid=obj["id"]; team_id=obj["team"]["id"]
    name=f"{obj['first_name']} {obj['last_name']}"
    img=get_headshot_url(pid)

    opp_id,opp_abbr=get_next_opponent(team_id)
    if opp_id is None:
        st.error("No upcoming opponent found.")
        return

    opp_dr=get_opp_def_rating(opp_abbr)
    pace=st.slider("Pace Factor",0.85,1.20,1.00,0.01)

    raw=get_player_stats(pid,40)
    core=normalize_stats_df(raw)
    feat=add_rolling_features(core)
    feat=add_context(feat,pace,opp_dr)

    models,resid=train_player_models(feat,STAT_COLUMNS)
    if not models:
        st.error("Not enough stats to model.")
        return

    last=feat.iloc[-1]
    preds,stds=predict_next_game(models,resid,last,pace,opp_dr)

    sims=run_mc(preds,stds)

    render_player_header(obj,name,img,pace,opp_dr,opp_abbr)
    render_core(preds)
    st.markdown("### Full Stat Table")
    render_table(sims)
    render_betting(sims)
    render_dist(sims)
    render_recent(core)

if __name__=="__main__":
    main()
