import streamlit as st
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──
st.set_page_config(
    page_title="Football XI Predictor",
    page_icon="⚽",
    layout="wide"
)

folder = os.path.dirname(os.path.abspath(__file__))
# ── Load Data ──
@st.cache_resource
def load_all():
    players  = pd.read_csv(os.path.join(folder, 'players_clustered.csv'))
    original = pd.read_csv(os.path.join(folder, 'players_data-2024_2025.csv'))
    comp_map = original[['Player','Squad','Comp']].drop_duplicates()
    players  = players.merge(comp_map, on=['Player','Squad'], how='left')

    with open(os.path.join(folder, 'rf_improved.pkl'),  'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(folder, 'xgb_improved.pkl'), 'rb') as f:
        xgb = pickle.load(f)
    with open(os.path.join(folder, 'team_stats_improved.pkl'), 'rb') as f:
        team_stats = pickle.load(f)
    with open(os.path.join(folder, 'matches_improved.pkl'), 'rb') as f:
        matches = pickle.load(f)
    return players, rf, xgb, team_stats, matches

players, rf_model, xgb_model, team_stats, matches = load_all()

# ── Constants ──
league_map = {
    'Premier League': 'eng Premier League',
    'La Liga':        'es La Liga',
    'Bundesliga':     'de Bundesliga',
    'Serie A':        'it Serie A',
    'Ligue 1':        'fr Ligue 1',
}

cluster_labels = {
    0: 'Rotation FW',
    1: 'Defensive MF/WB',
    2: 'Elite Attacker',
    3: 'Backup MF/DF',
    4: 'Striker',
    5: 'Creative MF'
}

# ── Helper Functions ──
def get_teams(league_comp):
    return sorted(players[players['Comp'] == league_comp]['Squad'].dropna().unique())

def get_squad(team):
    squad = players[players['Squad'] == team][['Player','Pos','cluster']].drop_duplicates('Player').reset_index(drop=True)
    squad['Role'] = squad['cluster'].map(cluster_labels)
    return squad

def get_stats(team):
    row = team_stats[team_stats['Squad'] == team]
    return row.iloc[0] if len(row) > 0 else None

def get_h2h(home_team, away_team):
    h2h = matches[
        ((matches['HomeTeam'] == home_team) & (matches['AwayTeam'] == away_team)) |
        ((matches['HomeTeam'] == away_team) & (matches['AwayTeam'] == home_team))
    ]
    if len(h2h) == 0:
        return 0.5, 0.5, 0
    hw = len(h2h[(h2h['HomeTeam']==home_team)&(h2h['FTR']=='H')]) + \
         len(h2h[(h2h['AwayTeam']==home_team)&(h2h['FTR']=='A')])
    aw = len(h2h[(h2h['HomeTeam']==away_team)&(h2h['FTR']=='H')]) + \
         len(h2h[(h2h['AwayTeam']==away_team)&(h2h['FTR']=='A')])
    total = len(h2h)
    return round(hw/total,2), round(aw/total,2), total

def predict(my_team, opp_live, opp_team):
    my  = get_stats(my_team)
    if my is None:
        return None, None
    h2h_hw, h2h_aw, h2h_tot = get_h2h(my_team, opp_team)
    features = [[
        my['team_attack'],       my['team_defend'],
        my['team_passing'],      my['team_movement'],
        opp_live['team_attack'], opp_live['team_defend'],
        opp_live['team_passing'],opp_live['team_movement'],
        my['team_attack_max'],   my['team_defend_max'],
        my['team_passing_max'],
        opp_live['team_attack'], opp_live['team_defend'],
        opp_live['team_passing'],
        my['elite_attackers'],   my['creative_mf'],   my['defensive_mf'],
        opp_live['elite_attackers'], opp_live['creative_mf'], opp_live['defensive_mf'],
        my['avg_scored'],        my['avg_conceded'],  my['goal_diff'],
        opp_live['avg_scored'],  opp_live['avg_conceded'], opp_live['goal_diff'],
        h2h_hw, h2h_aw, h2h_tot,
    ]]
    rf_p  = rf_model.predict_proba(features)[0]
    xgb_p = xgb_model.predict_proba(features)[0]
    avg   = (rf_p + xgb_p) / 2
    pred  = avg.argmax()
    return pred, avg

def pick_xi(my_team, opp_xi_full, opp_formation):
    opp_live = {
        'team_attack':    opp_xi_full['attack_score'].mean(),
        'team_defend':    opp_xi_full['defend_score'].mean(),
        'team_passing':   opp_xi_full['passing_score'].mean(),
        'team_movement':  opp_xi_full['movement_score'].mean(),
        'elite_attackers':(opp_xi_full['cluster']==2).sum(),
        'creative_mf':    (opp_xi_full['cluster']==5).sum(),
        'defensive_mf':   (opp_xi_full['cluster']==1).sum(),
        'avg_scored':     opp_xi_full['attack_score'].mean()/50,
        'avg_conceded':   opp_xi_full['defend_score'].mean()/50,
        'goal_diff':      (opp_xi_full['attack_score'].mean()-opp_xi_full['defend_score'].mean())/50,
    }

    parts         = opp_formation.split('-')
    opp_attackers = int(parts[-1])
    opp_defenders = int(parts[0])

    if opp_live['elite_attackers'] >= 2 or opp_attackers >= 3:
        strategy     = 'DEFENSIVE BLOCK'
        formation    = '5-4-1'
        roles_needed = {'GK':1,'DF':5,'MF':4,'FW':1}
        reason       = f"Opponent has {int(opp_live['elite_attackers'])} elite attackers"
    elif opp_live['creative_mf'] >= 2 or opp_live['team_passing'] > opp_live['team_attack']:
        strategy     = 'HIGH PRESS'
        formation    = '4-3-3'
        roles_needed = {'GK':1,'DF':4,'MF':3,'FW':3}
        reason       = "Opponent relies on passing — press and disrupt"
    elif opp_defenders >= 4 and opp_live['team_defend'] > opp_live['team_attack']:
        strategy     = 'COUNTER ATTACK'
        formation    = '4-5-1'
        roles_needed = {'GK':1,'DF':4,'MF':5,'FW':1}
        reason       = "Opponent is defensive — hit on the counter"
    else:
        strategy     = 'BALANCED'
        formation    = '4-4-2'
        roles_needed = {'GK':1,'DF':4,'MF':4,'FW':2}
        reason       = "Balanced matchup — flexible approach"

    my_players = players[players['Squad'] == my_team].copy()
    xi = []
    for role, count in roles_needed.items():
        pool = my_players[my_players['role'] == role].copy()
        if len(pool) == 0:
            continue
        if strategy == 'DEFENSIVE BLOCK':
            pool['final_score'] = pool['defend_score']*3 + pool['passing_score']
        elif strategy == 'HIGH PRESS':
            pool['final_score'] = pool['movement_score']*2 + pool['attack_score']*2
        elif strategy == 'COUNTER ATTACK':
            pool['final_score'] = pool['attack_score']*2 + pool['movement_score']*2
        else:
            pool['final_score'] = pool['attack_score'] + pool['defend_score'] + pool['passing_score']
        top = pool.nlargest(count, 'final_score')[['Player','Pos','cluster','final_score']]
        top['Role'] = role
        xi.append(top)

    full_xi = pd.concat(xi).reset_index(drop=True)
    full_xi['Cluster Role'] = full_xi['cluster'].map(cluster_labels)
    full_xi['final_score']  = full_xi['final_score'].round(1)
    return full_xi, formation, strategy, reason, opp_live

# ── UI ──
st.title("⚽ Football Best XI Predictor")
st.markdown("**Powered by XGBoost + Random Forest + K-Means | 75.9% Accuracy**")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 Your Team")
    my_league  = st.selectbox("Your League", list(league_map.keys()), key="my_league")
    my_teams   = get_teams(league_map[my_league])
    my_team    = st.selectbox("Your Team", my_teams, key="my_team")

with col2:
    st.subheader("🔴 Opponent Team")
    opp_league = st.selectbox("Opponent League", list(league_map.keys()), key="opp_league")
    opp_teams  = get_teams(league_map[opp_league])
    opp_team   = st.selectbox("Opponent Team", opp_teams, key="opp_team")

st.divider()

# Opponent XI selection
st.subheader(f"🔴 Select {opp_team}'s Starting XI")
squad = get_squad(opp_team)

# Show squad as table
st.dataframe(
    squad[['Player','Pos','Role']],
    use_container_width=True,
    hide_index=False
)

selected_players = st.multiselect(
    f"Pick exactly 11 players from {opp_team}",
    options=squad['Player'].tolist(),
    max_selections=11
)

opp_formation = st.selectbox(
    "Opponent Formation",
    ['4-3-3','4-4-2','3-5-2','4-2-3-1','5-3-2','5-4-1','4-1-4-1']
)

st.divider()

# ── Predict Button ──
if st.button("🚀 Predict Best XI", type="primary", use_container_width=True):
    if len(selected_players) != 11:
        st.error(f"Please select exactly 11 players. You selected {len(selected_players)}.")
    elif my_team == opp_team:
        st.error("Your team and opponent team cannot be the same!")
    else:
        opp_xi_full = players[players['Player'].isin(selected_players)]
        full_xi, formation, strategy, reason, opp_live = pick_xi(my_team, opp_xi_full, opp_formation)

        # Prediction
        pred, proba = predict(my_team, opp_live, opp_team)
        outcomes    = {2:'✅ WIN', 1:'🟡 DRAW', 0:'❌ LOSS'}

        st.divider()

        # Result header
        result_color = {'✅ WIN':'green','🟡 DRAW':'orange','❌ LOSS':'red'}
        pred_label   = outcomes[pred]
        st.markdown(f"## Prediction: {pred_label}")

        # Probabilities
        c1, c2, c3 = st.columns(3)
        c1.metric("Win Probability",  f"{proba[2]*100:.1f}%")
        c2.metric("Draw Probability", f"{proba[1]*100:.1f}%")
        c3.metric("Loss Probability", f"{proba[0]*100:.1f}%")

        st.divider()

        # Strategy
        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy",      strategy)
        c2.metric("Your Formation",formation)
        c3.metric("Reason",        reason)

        st.divider()

        # Opponent analysis
        st.subheader("🔴 Opponent XI Analysis")
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Attack",       f"{opp_live['team_attack']:.1f}")
        cb.metric("Defense",      f"{opp_live['team_defend']:.1f}")
        cc.metric("Passing",      f"{opp_live['team_passing']:.1f}")
        cd.metric("Elite FW",     f"{int(opp_live['elite_attackers'])}")

        st.divider()

        # Best XI
        st.subheader(f"🔵 {my_team} Best XI")
        st.dataframe(
            full_xi[['Player','Pos','Role','Cluster Role','final_score']].rename(
                columns={'final_score':'Score'}
            ),
            use_container_width=True,
            hide_index=True
        )

        # Download button
        csv = full_xi.to_csv(index=False)
        st.download_button(
            label="📥 Download Best XI as CSV",
            data=csv,
            file_name=f"BestXI_{my_team}_vs_{opp_team}.csv",
            mime="text/csv",
            use_container_width=True
        )