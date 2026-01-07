# -*- coding: utf-8 -*-
"""
ZONA DEL ANALISTA - Dashboard WhoScored
Código copiado directamente del Colab de John Triguero
"""
import streamlit as st
st.set_page_config(page_title="Zona del Analista", page_icon="⚽", layout="wide")

import json, re, pandas as pd, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import matplotlib.patches as patches
from io import BytesIO
from mplsoccer import Pitch, VerticalPitch, add_image
import matplotlib.patheffects as path_effects
from highlight_text import ax_text, fig_text
import warnings
from unidecode import unidecode
warnings.filterwarnings('ignore')

# ========== COLORES GLOBALES ==========
green = '#69f900'
red = '#ff4b44'
blue = '#00a0de'
violet = '#a369ff'
bg_color = '#363d4d'
line_color = '#ffffff'
col1 = '#ff4b44'
col2 = '#00FFD5'
hcol = '#ff4b44'
acol = '#00FFD5'
path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]

# ========== VARIABLES GLOBALES ==========
df = homedf = awaydf = None
hteamName = ateamName = None
Shotsdf = hShotsdf = aShotsdf = None
hgoal_count = agoal_count = 0
passes_df = None
home_passes_between_df = away_passes_between_df = None
home_average_locs_and_count_df = away_average_locs_and_count_df = None
defensive_home_average_locs_and_count_df = defensive_away_average_locs_and_count_df = None
defensive_actions_df = None
pearl_earring_cmaph = pearl_earring_cmapa = None
home_progressor_df = away_progressor_df = progressor_df = None
home_defender_df = away_defender_df = defender_df = None
home_sh_sq_df = away_sh_sq_df = sh_sq_df = None
xT_df = home_xT_df = away_xT_df = None
hxgot = axgot = 0
hTotalxG = aTotalxG = 0

# ========== FUNCIONES DE EXTRACCIÓN ==========
def extract_json_from_html(html_content):
    regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
    data_txt = re.findall(regex_pattern, html_content)[0]
    data_txt = data_txt.replace('matchId', '"matchId"')
    data_txt = data_txt.replace('matchCentreData', '"matchCentreData"')
    data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
    data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"')
    data_txt = data_txt.replace('};', '}')
    return data_txt

def extract_data_from_dict(data):
    events_dict = data["matchCentreData"]["events"]
    teams_dict = {data["matchCentreData"]['home']['teamId']: data["matchCentreData"]['home']['name'],
                  data["matchCentreData"]['away']['teamId']: data["matchCentreData"]['away']['name']}
    players_home_df = pd.DataFrame(data["matchCentreData"]['home']['players'])
    players_home_df["teamId"] = data["matchCentreData"]['home']['teamId']
    players_away_df = pd.DataFrame(data["matchCentreData"]['away']['players'])
    players_away_df["teamId"] = data["matchCentreData"]['away']['teamId']
    players_df = pd.concat([players_home_df, players_away_df])
    return events_dict, players_df, teams_dict

def get_short_name(full_name):
    if pd.isna(full_name): return ''
    if not isinstance(full_name, str): return str(full_name)
    parts = full_name.split()
    if len(parts) == 1: return full_name
    elif len(parts) == 2: return parts[0][0] + ". " + parts[1]
    else: return parts[0][0] + ". " + parts[-1]

def cumulative_match_mins(events_df):
    match_events = events_df.copy()
    match_events['cumulative_mins'] = match_events['minute'] + (1/60) * match_events['second']
    for period in np.arange(1, match_events['period'].max() + 1, 1):
        if period > 1:
            t_delta = match_events[match_events['period'] == period - 1]['cumulative_mins'].max() - match_events[match_events['period'] == period]['cumulative_mins'].min()
        else:
            t_delta = 0
        match_events.loc[match_events['period'] == period, 'cumulative_mins'] += t_delta
    return match_events

def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=60, min_carry_duration=1, max_carry_duration=10):
    match_events = events_df.reset_index(drop=True)
    match_carries = pd.DataFrame()
    for idx in range(len(match_events) - 1):
        match_event = match_events.iloc[idx]
        next_evt = match_events.iloc[idx + 1]
        
        endX = match_event.get('endX')
        if pd.isna(endX): continue
        
        same_team = match_event['teamId'] == next_evt['teamId']
        not_ball_touch = match_event['type'] != 'BallTouch'
        dx = (endX - next_evt['x'])
        dy = (match_event.get('endY', 0) - next_evt['y'])
        dist_sq = dx**2 + dy**2
        far_enough = dist_sq >= min_carry_length**2
        not_too_far = dist_sq <= max_carry_length**2
        dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
        min_time = dt >= min_carry_duration
        same_phase = dt < max_carry_duration
        same_period = match_event['period'] == next_evt['period']
        
        if same_team and not_ball_touch and far_enough and not_too_far and min_time and same_phase and same_period:
            carry = pd.DataFrame([{
                'minute': match_event['minute'], 'second': match_event['second'],
                'teamId': next_evt['teamId'], 'x': endX, 'y': match_event.get('endY', 0),
                'endX': next_evt['x'], 'endY': next_evt['y'], 'type': 'Carry',
                'outcomeType': 'Successful', 'qualifiers': '[]', 'isTouch': True,
                'playerId': next_evt.get('playerId'), 'period': next_evt['period'],
                'cumulative_mins': (match_event['cumulative_mins'] + next_evt['cumulative_mins']) / 2
            }])
            match_carries = pd.concat([match_carries, carry], ignore_index=True)
    
    result = pd.concat([match_carries, match_events], ignore_index=True)
    return result.sort_values(['period', 'cumulative_mins']).reset_index(drop=True)

# ========== PROCESAMIENTO DE DATOS ==========
def process_all_data(events_dict, players_df, teams_dict):
    global df, homedf, awaydf, hteamName, ateamName
    global Shotsdf, hShotsdf, aShotsdf, hgoal_count, agoal_count
    global passes_df, home_passes_between_df, away_passes_between_df
    global home_average_locs_and_count_df, away_average_locs_and_count_df
    global defensive_home_average_locs_and_count_df, defensive_away_average_locs_and_count_df
    global defensive_actions_df, pearl_earring_cmaph, pearl_earring_cmapa
    global home_progressor_df, away_progressor_df, progressor_df
    global home_defender_df, away_defender_df, defender_df
    global home_sh_sq_df, away_sh_sq_df, sh_sq_df
    global xT_df, home_xT_df, away_xT_df
    
    df = pd.DataFrame(events_dict)
    dfp = players_df.copy()
    
    # Extract displayName
    df['type'] = df['type'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    df['outcomeType'] = df['outcomeType'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    df['period'] = df['period'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    df['period'] = df['period'].replace({'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3, 'SecondPeriodOfExtraTime': 4, 'PenaltyShootout': 5})
    df['period'] = pd.to_numeric(df['period'], errors='coerce').fillna(1).astype(int)
    
    df['minute'] = pd.to_numeric(df['minute'], errors='coerce').fillna(0)
    df['second'] = pd.to_numeric(df['second'], errors='coerce').fillna(0)
    df = cumulative_match_mins(df)
    df = insert_ball_carries(df)
    df = df.reset_index(drop=True)
    df['index'] = range(1, len(df) + 1)
    
    df['teamName'] = df['teamId'].map(teams_dict)
    team_names = list(teams_dict.values())
    hteamName, ateamName = team_names[0], team_names[1]
    
    # Coordenadas
    df['x'] = pd.to_numeric(df['x'], errors='coerce') * 1.05
    df['y'] = pd.to_numeric(df['y'], errors='coerce') * 0.68
    df['endX'] = pd.to_numeric(df['endX'], errors='coerce') * 1.05
    df['endY'] = pd.to_numeric(df['endY'], errors='coerce') * 0.68
    if 'goalMouthY' in df.columns:
        df['goalMouthY'] = pd.to_numeric(df['goalMouthY'], errors='coerce') * 0.68
    if 'goalMouthZ' in df.columns:
        df['goalMouthZ'] = pd.to_numeric(df['goalMouthZ'], errors='coerce')
    
    # Merge player info
    df = df.merge(dfp[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left')
    df['qualifiers'] = df['qualifiers'].astype(str)
    
    # Progressive pass/carry
    df['prog_pass'] = np.where(df['type'] == 'Pass', np.sqrt((105-df['x'])**2 + (34-df['y'])**2) - np.sqrt((105-df['endX'])**2 + (34-df['endY'])**2), 0)
    df['prog_carry'] = np.where(df['type'] == 'Carry', np.sqrt((105-df['x'])**2 + (34-df['y'])**2) - np.sqrt((105-df['endX'])**2 + (34-df['endY'])**2), 0)
    df['pass_or_carry_angle'] = np.degrees(np.arctan2(df['endY'] - df['y'], df['endX'] - df['x']))
    df['isTouch'] = df['type'].isin(['Pass', 'Carry', 'TakeOn', 'MissedShots', 'SavedShot', 'Goal', 'ShotOnPost', 'BallTouch', 'Aerial'])
    df['shortName'] = df['name'].apply(get_short_name)
    
    # Calcular xT (Expected Threat)
    try:
        xT_grid = np.array([
            [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506, 0.02122129, 0.02812085, 0.03802537, 0.05310954],
            [0.00750072, 0.00878589, 0.00942382, 0.0105949, 0.01214719, 0.0138454, 0.01611813, 0.01870347, 0.02401521, 0.0319032, 0.04269431, 0.05765452],
            [0.00794052, 0.00952109, 0.01028009, 0.01168775, 0.01350509, 0.01553033, 0.01861617, 0.02256287, 0.02969768, 0.04004783, 0.05400562, 0.07072428],
            [0.00848996, 0.01015328, 0.01111454, 0.01267414, 0.01465428, 0.01697773, 0.02059657, 0.02537277, 0.03489163, 0.04852385, 0.06550272, 0.08150882],
            [0.00875947, 0.01062255, 0.01143469, 0.01303705, 0.01522247, 0.01820187, 0.02248052, 0.02804805, 0.03872101, 0.05765452, 0.07888746, 0.10948258],
            [0.00875947, 0.01062255, 0.01143469, 0.01303705, 0.01522247, 0.01820187, 0.02248052, 0.02804805, 0.03872101, 0.05765452, 0.07888746, 0.10948258],
            [0.00848996, 0.01015328, 0.01111454, 0.01267414, 0.01465428, 0.01697773, 0.02059657, 0.02537277, 0.03489163, 0.04852385, 0.06550272, 0.08150882],
            [0.00794052, 0.00952109, 0.01028009, 0.01168775, 0.01350509, 0.01553033, 0.01861617, 0.02256287, 0.02969768, 0.04004783, 0.05400562, 0.07072428]
        ])
        xT_rows, xT_cols = xT_grid.shape
        
        # Solo para pases y carries exitosos (sin corners)
        dfxT = df[(df['type'].isin(['Pass', 'Carry'])) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('Corner', na=False))].copy()
        
        # Convertir coordenadas a bins (x: 0-105 -> 0-12, y: 0-68 -> 0-8)
        dfxT['x1_bin'] = np.clip((dfxT['x'] / 105 * xT_cols).astype(int), 0, xT_cols-1)
        dfxT['y1_bin'] = np.clip((dfxT['y'] / 68 * xT_rows).astype(int), 0, xT_rows-1)
        dfxT['x2_bin'] = np.clip((dfxT['endX'] / 105 * xT_cols).astype(int), 0, xT_cols-1)
        dfxT['y2_bin'] = np.clip((dfxT['endY'] / 68 * xT_rows).astype(int), 0, xT_rows-1)
        
        dfxT['start_xT'] = dfxT.apply(lambda r: xT_grid[int(r['y1_bin']), int(r['x1_bin'])], axis=1)
        dfxT['end_xT'] = dfxT.apply(lambda r: xT_grid[int(r['y2_bin']), int(r['x2_bin'])], axis=1)
        dfxT['xT'] = dfxT['end_xT'] - dfxT['start_xT']
        
        # Merge xT back to main df
        df = df.merge(dfxT[['index', 'xT']], on='index', how='left')
        df['xT'] = df['xT'].fillna(0)
    except Exception as e:
        df['xT'] = 0
    
    homedf = df[df['teamName'] == hteamName]
    awaydf = df[df['teamName'] == ateamName]
    
    # Shots
    Shotsdf = df[df['type'].isin(['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost'])].copy()
    hShotsdf = Shotsdf[Shotsdf['teamName'] == hteamName].copy()
    aShotsdf = Shotsdf[Shotsdf['teamName'] == ateamName].copy()
    
    # Goals
    hgoal_count = len(homedf[(homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal', na=False))])
    hgoal_count += len(awaydf[(awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal', na=False))])
    agoal_count = len(awaydf[(awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal', na=False))])
    agoal_count += len(homedf[(homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal', na=False))])
    
    # Passes DF
    passes_df = df[~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card', na=False)].copy()
    passes_df["receiver"] = passes_df["playerId"].shift(-1)
    passes_df = passes_df[passes_df['type'] == 'Pass'].copy()
    passes_df = passes_df.merge(dfp[["playerId", "isFirstEleven"]], on='playerId', how='left', suffixes=('', '_dup'))
    
    # Home passes between
    home_passes_df = passes_df[passes_df['teamName'] == hteamName]
    home_avg = homedf.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']})
    home_avg.columns = ['pass_avg_x', 'pass_avg_y', 'count']
    home_avg = home_avg.merge(dfp[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left').set_index('playerId')
    home_average_locs_and_count_df = home_avg.copy()
    
    home_ids = home_passes_df[['playerId', 'receiver']].copy()
    home_ids['pos_max'] = home_ids[['playerId', 'receiver']].max(axis=1)
    home_ids['pos_min'] = home_ids[['playerId', 'receiver']].min(axis=1)
    home_passes_between_df = home_ids.groupby(['pos_min', 'pos_max']).size().reset_index(name='pass_count')
    home_passes_between_df = home_passes_between_df.merge(home_avg[['pass_avg_x', 'pass_avg_y', 'name']], left_on='pos_min', right_index=True)
    home_passes_between_df = home_passes_between_df.merge(home_avg[['pass_avg_x', 'pass_avg_y', 'name']], left_on='pos_max', right_index=True, suffixes=['', '_end'])
    
    # Away passes between
    away_passes_df = passes_df[passes_df['teamName'] == ateamName]
    away_avg = awaydf.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']})
    away_avg.columns = ['pass_avg_x', 'pass_avg_y', 'count']
    away_avg = away_avg.merge(dfp[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left').set_index('playerId')
    away_average_locs_and_count_df = away_avg.copy()
    
    away_ids = away_passes_df[['playerId', 'receiver']].copy()
    away_ids['pos_max'] = away_ids[['playerId', 'receiver']].max(axis=1)
    away_ids['pos_min'] = away_ids[['playerId', 'receiver']].min(axis=1)
    away_passes_between_df = away_ids.groupby(['pos_min', 'pos_max']).size().reset_index(name='pass_count')
    away_passes_between_df = away_passes_between_df.merge(away_avg[['pass_avg_x', 'pass_avg_y', 'name']], left_on='pos_min', right_index=True)
    away_passes_between_df = away_passes_between_df.merge(away_avg[['pass_avg_x', 'pass_avg_y', 'name']], left_on='pos_max', right_index=True, suffixes=['', '_end'])
    
    # Defensive actions
    defensive_actions_df = df[((df['type'] == 'Aerial') & (df['qualifiers'].str.contains('Defensive', na=False))) |
                              df['type'].isin(['BallRecovery', 'BlockedPass', 'Challenge', 'Clearance', 'Foul', 'Interception', 'Tackle'])]
    
    home_def = defensive_actions_df[defensive_actions_df['teamName'] == hteamName]
    home_def_avg = home_def.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']})
    home_def_avg.columns = ['x', 'y', 'count']
    home_def_avg = home_def_avg.merge(dfp[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left').set_index('playerId')
    defensive_home_average_locs_and_count_df = home_def_avg[home_def_avg['position'] != 'GK'] if 'position' in home_def_avg.columns else home_def_avg
    
    away_def = defensive_actions_df[defensive_actions_df['teamName'] == ateamName]
    away_def_avg = away_def.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']})
    away_def_avg.columns = ['x', 'y', 'count']
    away_def_avg = away_def_avg.merge(dfp[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left').set_index('playerId')
    defensive_away_average_locs_and_count_df = away_def_avg[away_def_avg['position'] != 'GK'] if 'position' in away_def_avg.columns else away_def_avg
    
    pearl_earring_cmaph = LinearSegmentedColormap.from_list("c", [bg_color, hcol], N=20)
    pearl_earring_cmapa = LinearSegmentedColormap.from_list("c", [bg_color, acol], N=20)
    
    # Player stats
    unique_players = df['name'].dropna().unique()
    home_unique = homedf['name'].dropna().unique()
    away_unique = awaydf['name'].dropna().unique()
    
    # Progressor
    def calc_progressor(players, team_df=None):
        counts = {'name': [], 'Progressive Passes': [], 'Progressive Carries': [], 'LineBreaking Pass': []}
        for n in players:
            counts['name'].append(n)
            counts['Progressive Passes'].append(len(df[(df['name']==n) & (df['prog_pass']>=9.144) & (df['x']>=35) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('CornerTaken|Freekick', na=False))]))
            counts['Progressive Carries'].append(len(df[(df['name']==n) & (df['prog_carry']>=9.144) & (df['endX']>=35)]))
            counts['LineBreaking Pass'].append(len(df[(df['name']==n) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Throughball', na=False))]))
        result = pd.DataFrame(counts)
        result['total'] = result['Progressive Passes'] + result['Progressive Carries'] + result['LineBreaking Pass']
        result = result.sort_values('total', ascending=False).reset_index(drop=True)
        result['shortName'] = result['name'].apply(get_short_name)
        return result
    
    home_progressor_df = calc_progressor(home_unique).head(5)
    away_progressor_df = calc_progressor(away_unique).head(5)
    progressor_df = calc_progressor(unique_players).head(10)
    
    # Defender
    def calc_defender(players):
        counts = {'name': [], 'Tackles': [], 'Interceptions': [], 'Clearance': []}
        for n in players:
            counts['name'].append(n)
            counts['Tackles'].append(len(df[(df['name']==n) & (df['type']=='Tackle') & (df['outcomeType']=='Successful')]))
            counts['Interceptions'].append(len(df[(df['name']==n) & (df['type']=='Interception')]))
            counts['Clearance'].append(len(df[(df['name']==n) & (df['type']=='Clearance')]))
        result = pd.DataFrame(counts)
        result['total'] = result['Tackles'] + result['Interceptions'] + result['Clearance']
        result = result.sort_values('total', ascending=False).reset_index(drop=True)
        result['shortName'] = result['name'].apply(get_short_name)
        return result
    
    home_defender_df = calc_defender(home_unique).head(5)
    away_defender_df = calc_defender(away_unique).head(5)
    defender_df = calc_defender(unique_players).head(10)
    
    # Shot sequence
    df_nc = df[df['type'] != 'Carry']
    def calc_shot_seq(players):
        counts = {'name': [], 'Shots': [], 'Shot Assist': [], 'Buildup to shot': []}
        for n in players:
            counts['name'].append(n)
            counts['Shots'].append(len(df[(df['name']==n) & (df['type'].isin(['MissedShots', 'SavedShot', 'ShotOnPost', 'Goal']))]))
            counts['Shot Assist'].append(len(df[(df['name']==n) & (df['type']=='Pass') & (df['qualifiers'].str.contains('KeyPass', na=False))]))
            counts['Buildup to shot'].append(len(df_nc[(df_nc['name']==n) & (df_nc['type']=='Pass') & (df_nc['qualifiers'].shift(-1).str.contains('KeyPass', na=False))]))
        result = pd.DataFrame(counts)
        result['total'] = result['Shots'] + result['Shot Assist'] + result['Buildup to shot']
        result = result.sort_values('total', ascending=False).reset_index(drop=True)
        result['shortName'] = result['name'].apply(get_short_name)
        return result
    
    home_sh_sq_df = calc_shot_seq(home_unique).head(5)
    away_sh_sq_df = calc_shot_seq(away_unique).head(5)
    sh_sq_df = calc_shot_seq(unique_players).head(10)
    
    # xT
    def calc_xT(players):
        counts = {'name': [], 'xT from Pass': [], 'xT from Carry': []}
        for n in players:
            counts['name'].append(n)
            if 'xT' in df.columns:
                xtp = df[(df['name']==n) & (df['type']=='Pass') & (df['xT']>=0) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn', na=False))]
                xtc = df[(df['name']==n) & (df['type']=='Carry') & (df['xT']>=0)]
                counts['xT from Pass'].append(round(xtp['xT'].sum(), 2))
                counts['xT from Carry'].append(round(xtc['xT'].sum(), 2))
            else:
                counts['xT from Pass'].append(0)
                counts['xT from Carry'].append(0)
        result = pd.DataFrame(counts)
        result['total'] = result['xT from Pass'] + result['xT from Carry']
        result = result.sort_values('total', ascending=False).reset_index(drop=True)
        result['shortName'] = result['name'].apply(get_short_name)
        return result
    
    home_xT_df = calc_xT(home_unique).head(5)
    away_xT_df = calc_xT(away_unique).head(5)
    xT_df = calc_xT(unique_players).head(10)
    
    return hgoal_count, agoal_count

# ========== FUNCIONES DE VISUALIZACIÓN - DASHBOARD 1 ==========
def pass_network_visualization(ax, passes_between_df, average_locs_and_count_df, col, teamName):
    MAX_LINE_WIDTH = 15
    passes_between_df['width'] = (passes_between_df.pass_count / passes_between_df.pass_count.max() * MAX_LINE_WIDTH)
    color = np.array(to_rgba(col))
    color = np.tile(color, (len(passes_between_df), 1))
    c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
    c_transparency = (c_transparency * 0.8) + 0.1
    color[:, 3] = c_transparency
    
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    pitch.lines(passes_between_df.pass_avg_x, passes_between_df.pass_avg_y, passes_between_df.pass_avg_x_end, passes_between_df.pass_avg_y_end,
                lw=passes_between_df.width, color=color, zorder=1, ax=ax)
    
    for index, row in average_locs_and_count_df.reset_index(drop=True).iterrows():
        marker = 'o' if row.get('isFirstEleven', True) else 's'
        pitch.scatter(row['pass_avg_x'], row['pass_avg_y'], s=1000, marker=marker, color=bg_color, edgecolor=line_color, linewidth=2, ax=ax)
        pitch.annotate(int(row["shirtNo"]) if pd.notna(row.get("shirtNo")) else '', xy=(row.pass_avg_x, row.pass_avg_y), c=col, ha='center', va='center', size=18, ax=ax)
    
    avgph = round(average_locs_and_count_df['pass_avg_x'].median(), 2)
    ax.axvline(x=avgph, color='gray', linestyle='--', alpha=0.75, linewidth=2)
    
    team_passes = passes_df[passes_df["teamName"] == teamName].copy() if passes_df is not None else pd.DataFrame()
    if len(team_passes) > 0:
        team_passes['angle'] = team_passes['pass_or_carry_angle'].abs()
        team_passes = team_passes[(team_passes['angle'] >= 0) & (team_passes['angle'] <= 90)]
        verticality = round((1 - team_passes['angle'].median()/90)*100, 2) if len(team_passes) > 0 else 0
    else:
        verticality = 0
    
    if teamName == ateamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.text(avgph-1, 73, f"{avgph}m", fontsize=15, color=line_color, ha='left')
        ax.text(105, 73, f"verticalidad: {verticality}%", fontsize=15, color=line_color, ha='left')
        ax.text(2, 2, "circulo = Titulares\ncuadro= suplentes", color=col, size=12, ha='right', va='top')
    else:
        ax.text(avgph-1, -5, f"{avgph}m", fontsize=15, color=line_color, ha='right')
        ax.text(105, -5, f"verticalidad: {verticality}%", fontsize=15, color=line_color, ha='right')
        ax.text(2, 66, "circulo = Titulares\ncuadro= suplentes", color=col, size=12, ha='left', va='top')
    ax.set_title(f"{teamName}\nRed de pases", color=line_color, size=25, fontweight='bold')

def plot_shotmap(ax):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, linewidth=2, line_color=line_color)
    pitch.draw(ax=ax)
    ax.set_ylim(-0.5, 68.5)
    ax.set_xlim(-0.5, 105.5)
    
    # Home shots (inverted)
    hGoal = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'Goal')]
    hPost = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'ShotOnPost')]
    hSave = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'SavedShot')]
    hMiss = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'MissedShots')]
    
    pitch.scatter((105 - hGoal.x), (68 - hGoal.y), s=350, edgecolors='white', c='None', marker='football', zorder=3, ax=ax)
    pitch.scatter((105 - hPost.x), (68 - hPost.y), s=200, edgecolors=hcol, c=hcol, marker='o', ax=ax)
    pitch.scatter((105 - hSave.x), (68 - hSave.y), s=200, edgecolors=hcol, c='None', hatch='///////', marker='o', ax=ax)
    pitch.scatter((105 - hMiss.x), (68 - hMiss.y), s=200, edgecolors=hcol, c='None', marker='o', ax=ax)
    
    # Away shots
    aGoal = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'Goal')]
    aPost = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'ShotOnPost')]
    aSave = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'SavedShot')]
    aMiss = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'MissedShots')]
    
    pitch.scatter(aGoal.x, aGoal.y, s=350, edgecolors='white', c='None', marker='football', zorder=3, ax=ax)
    pitch.scatter(aPost.x, aPost.y, s=200, edgecolors=acol, c=acol, marker='o', ax=ax)
    pitch.scatter(aSave.x, aSave.y, s=200, edgecolors=acol, c='None', hatch='///////', marker='o', ax=ax)
    pitch.scatter(aMiss.x, aMiss.y, s=200, edgecolors=acol, c='None', marker='o', ax=ax)
    
    ax.text(0, 70, f"{hteamName}\n<---Tiros", color=hcol, size=25, ha='left', fontweight='bold')
    ax.text(105, 70, f"{ateamName}\nTiros--->", color=acol, size=25, ha='right', fontweight='bold')

def defensive_block(ax, average_locs_and_count_df, team_name, col):
    defensive_actions_team_df = defensive_actions_df[defensive_actions_df["teamName"] == team_name]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, line_zorder=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if len(average_locs_and_count_df) == 0:
        ax.set_title(f"{team_name}\nBloque Defensivo", color=line_color, fontsize=25, fontweight='bold')
        return
    
    MAX_MARKER_SIZE = 3500
    average_locs_and_count_df = average_locs_and_count_df.copy()
    average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count'] / average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)
    
    flamingo_cmap = LinearSegmentedColormap.from_list("c", [bg_color, col], N=500)
    if len(defensive_actions_team_df) > 2:
        try:
            pitch.kdeplot(defensive_actions_team_df.x, defensive_actions_team_df.y, ax=ax, fill=True, levels=5000, thresh=0.02, cut=4, cmap=flamingo_cmap)
        except: pass
    
    for index, row in average_locs_and_count_df.reset_index(drop=True).iterrows():
        marker = 'o' if row.get('isFirstEleven', True) else 's'
        pitch.scatter(row['x'], row['y'], s=row['marker_size']+100, marker=marker, color=bg_color, edgecolor=line_color, linewidth=1, zorder=3, ax=ax)
        pitch.annotate(int(row["shirtNo"]) if pd.notna(row.get("shirtNo")) else '', xy=(row.x, row.y), c=line_color, ha='center', va='center', size=14, ax=ax)
    
    pitch.scatter(defensive_actions_team_df.x, defensive_actions_team_df.y, s=10, marker='x', color='yellow', alpha=0.2, ax=ax)
    
    dah = round(average_locs_and_count_df['x'].mean(), 2)
    ax.axvline(x=dah, color='gray', linestyle='--', alpha=0.75, linewidth=2)
    
    cbs = average_locs_and_count_df[average_locs_and_count_df['position'] == 'DC'] if 'position' in average_locs_and_count_df.columns else average_locs_and_count_df
    def_line_h = round(cbs['x'].median(), 2) if len(cbs) > 0 else dah
    fwds = average_locs_and_count_df.nlargest(2, 'x')
    fwd_line_h = round(fwds['x'].mean(), 2) if len(fwds) > 0 else dah
    compactness = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2) if fwd_line_h != def_line_h else 0
    
    if team_name == ateamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.text(dah-1, 73, f"{round(dah*1.05, 2)}m", fontsize=15, color=line_color, ha='left')
        ax.text(105, 73, f'Defensa Compacta: {compactness}%', fontsize=15, color=line_color, ha='left')
    else:
        ax.text(dah-1, -5, f"{round(dah*1.05, 2)}m", fontsize=15, color=line_color, ha='right')
        ax.text(105, -5, f'Defensa Compacta: {compactness}%', fontsize=15, color=line_color, ha='right')
    ax.set_title(f"{team_name}\nBloque Defensivo", color=line_color, fontsize=25, fontweight='bold')

def plot_goalPost(ax):
    # Copiar dataframes para no modificar originales
    hs = hShotsdf.copy()
    aws = aShotsdf.copy()
    
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=bg_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_ylim(-0.5, 68.5)
    ax.set_xlim(-0.5, 105.5)
    
    # Away goalpost (abajo)
    ax.plot([7.5, 7.5], [0, 30], color=line_color, linewidth=5)
    ax.plot([7.5, 97.5], [30, 30], color=line_color, linewidth=5)
    ax.plot([97.5, 97.5], [30, 0], color=line_color, linewidth=5)
    ax.plot([0, 105], [0, 0], color=line_color, linewidth=3)
    for y in np.arange(0, 6) * 6:
        ax.plot([7.5, 97.5], [y, y], color=line_color, linewidth=2, alpha=0.2)
    for x in (np.arange(0, 11) * 9) + 7.5:
        ax.plot([x, x], [0, 30], color=line_color, linewidth=2, alpha=0.2)
    
    # Home goalpost (arriba)
    ax.plot([7.5, 7.5], [38, 68], color=line_color, linewidth=5)
    ax.plot([7.5, 97.5], [68, 68], color=line_color, linewidth=5)
    ax.plot([97.5, 97.5], [68, 38], color=line_color, linewidth=5)
    ax.plot([0, 105], [38, 38], color=line_color, linewidth=3)
    for y in (np.arange(0, 6) * 6) + 38:
        ax.plot([7.5, 97.5], [y, y], color=line_color, linewidth=2, alpha=0.2)
    for x in (np.arange(0, 11) * 9) + 7.5:
        ax.plot([x, x], [38, 68], color=line_color, linewidth=2, alpha=0.2)
    
    hSaves = aSaves = 0
    
    # Verificar si hay datos de goalMouth
    if 'goalMouthZ' in hs.columns and 'goalMouthY' in hs.columns:
        # Transformar coordenadas - goalMouthY ya está en escala 0.68, necesitamos convertir a 0-100 primero
        # goalMouthY original era 0-100, lo multiplicamos por 0.68 en process_all_data
        # Para la transformación del Colab: ((37.66 - goalMouthY_original) * 12.295) + 7.5
        # goalMouthY_original = goalMouthY / 0.68
        
        for idx, row in hs.iterrows():
            if pd.notna(row.get('goalMouthZ')) and pd.notna(row.get('goalMouthY')):
                # Coordenada Z (altura)
                gz = row['goalMouthZ'] * 0.75
                # Coordenada Y (horizontal) - convertir de vuelta a escala original
                gy_orig = row['goalMouthY'] / 0.68 if row['goalMouthY'] != 0 else 50
                gy = ((37.66 - gy_orig) * 12.295) + 7.5
                gy = max(7.5, min(97.5, gy))  # Limitar al área de portería
                gz = max(0, min(30, gz))
                
                shot_type = row['type']
                is_bc = 'BigChance' in str(row.get('qualifiers', ''))
                size = 1000 if is_bc else 350
                
                if shot_type == 'Goal' and 'OwnGoal' not in str(row.get('qualifiers', '')):
                    ax.scatter(gy, gz, s=size, marker='o', c='white', edgecolor='gold', linewidth=3, zorder=5)
                elif shot_type == 'SavedShot':
                    ax.scatter(gy, gz, s=size, marker='o', c='none', edgecolor=acol, hatch='/////', linewidth=2, zorder=4)
                    hSaves += 1
                elif shot_type == 'ShotOnPost':
                    ax.scatter(gy, gz, s=size, marker='o', c='none', edgecolor='orange', hatch='/////', linewidth=2, zorder=4)
        
        for idx, row in aws.iterrows():
            if pd.notna(row.get('goalMouthZ')) and pd.notna(row.get('goalMouthY')):
                # Coordenada Z (altura) + offset para portería de arriba
                gz = (row['goalMouthZ'] * 0.75) + 38
                # Coordenada Y (horizontal)
                gy_orig = row['goalMouthY'] / 0.68 if row['goalMouthY'] != 0 else 50
                gy = ((37.66 - gy_orig) * 12.295) + 7.5
                gy = max(7.5, min(97.5, gy))
                gz = max(38, min(68, gz))
                
                shot_type = row['type']
                is_bc = 'BigChance' in str(row.get('qualifiers', ''))
                size = 1000 if is_bc else 350
                
                if shot_type == 'Goal' and 'OwnGoal' not in str(row.get('qualifiers', '')):
                    ax.scatter(gy, gz, s=size, marker='o', c='white', edgecolor='gold', linewidth=3, zorder=5)
                elif shot_type == 'SavedShot':
                    ax.scatter(gy, gz, s=size, marker='o', c='none', edgecolor=hcol, hatch='/////', linewidth=2, zorder=4)
                    aSaves += 1
                elif shot_type == 'ShotOnPost':
                    ax.scatter(gy, gz, s=size, marker='o', c='none', edgecolor='orange', hatch='/////', linewidth=2, zorder=4)
    
    # Si no hay coordenadas goalMouth, contar paradas de otra forma
    if hSaves == 0:
        hSaves = len(hs[hs['type'] == 'SavedShot'])
    if aSaves == 0:
        aSaves = len(aws[aws['type'] == 'SavedShot'])
    
    ax.text(52.5, 70, f"{hteamName} PT Paradas", color=hcol, fontsize=30, ha='center', fontweight='bold')
    ax.text(52.5, -2, f"{ateamName} PT Paradas", color=acol, fontsize=30, ha='center', va='top', fontweight='bold')
    ax.text(100, 68, f"Paradas = {aSaves}", color=hcol, fontsize=16, va='top', ha='left')
    ax.text(100, 2, f"Paradas = {hSaves}", color=acol, fontsize=16, va='bottom', ha='left')

def draw_progressive_pass_map(ax, team_name, col):
    dfpro = df[(df['teamName']==team_name) & (df['prog_pass']>=9.11) & (~df['qualifiers'].str.contains('CornerTaken|Freekick', na=False)) &
               (df['x']>=35) & (df['outcomeType']=='Successful')]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if team_name == ateamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    pro_count = len(dfpro)
    if pro_count == 0:
        ax.set_title(f"{team_name}\n0 Pases Progresivos", color=line_color, fontsize=25, fontweight='bold')
        return
    
    left_pro = len(dfpro[dfpro['y']>=45.33])
    mid_pro = len(dfpro[(dfpro['y']>=22.67) & (dfpro['y']<45.33)])
    right_pro = len(dfpro[dfpro['y']<22.67])
    
    ax.hlines(22.67, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.hlines(45.33, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
    
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    ax.text(8, 11.335, f'{right_pro}\n({round(right_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 34, f'{mid_pro}\n({round(mid_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 56.675, f'{left_pro}\n({round(left_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    
    pitch.lines(dfpro.x, dfpro.y, dfpro.endX, dfpro.endY, lw=3.5, comet=True, color=col, ax=ax, alpha=0.5)
    pitch.scatter(dfpro.endX, dfpro.endY, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2, ax=ax)
    ax.set_title(f"{team_name}\n{pro_count} Pases Progresivos", color=line_color, fontsize=25, fontweight='bold')

def draw_progressive_carry_map(ax, team_name, col):
    dfpro = df[(df['teamName']==team_name) & (df['prog_carry']>=9.11) & (df['endX']>=35)]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if team_name == ateamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    pro_count = len(dfpro)
    if pro_count == 0:
        ax.set_title(f"{team_name}\n0 Conducción Progresiva", color=line_color, fontsize=25, fontweight='bold')
        return
    
    left_pro = len(dfpro[dfpro['y']>=45.33])
    mid_pro = len(dfpro[(dfpro['y']>=22.67) & (dfpro['y']<45.33)])
    right_pro = len(dfpro[dfpro['y']<22.67])
    
    ax.hlines(22.67, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.hlines(45.33, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
    
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    ax.text(8, 11.335, f'{right_pro}\n({round(right_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 34, f'{mid_pro}\n({round(mid_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 56.675, f'{left_pro}\n({round(left_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    
    for _, row in dfpro.iterrows():
        if pd.notna(row['x']) and pd.notna(row['endX']):
            arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=col, zorder=4, mutation_scale=20, alpha=0.9, linewidth=2, linestyle='--')
            ax.add_patch(arrow)
    ax.set_title(f"{team_name}\n{pro_count} Conducción Progresiva", color=line_color, fontsize=25, fontweight='bold')

def plot_Momentum(ax):
    ax.set_facecolor(bg_color)
    ax.set_title("Momentum xT", color=line_color, fontsize=25, fontweight='bold')
    ax.text(52.5, 0.5, "xT no disponible", color=line_color, fontsize=15, ha='center', va='center', transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

def plotting_match_stats(ax):
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 105)
    ax.set_ylim(-5, 65)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    
    def get_stats(team):
        t = df[df['teamName'] == team]
        p = t[t['type'] == 'Pass']
        a = p[p['outcomeType'] == 'Successful']
        lb = p[p['qualifiers'].str.contains('Longball', na=False)]
        lba = lb[lb['outcomeType'] == 'Successful']
        co = p[p['qualifiers'].str.contains('CornerTaken', na=False)]
        tk = t[t['type'] == 'Tackle']
        tkw = tk[tk['outcomeType'] == 'Successful']
        i = t[t['type'] == 'Interception']
        c = t[t['type'] == 'Clearance']
        ae = t[t['type'] == 'Aerial']
        aew = ae[ae['outcomeType'] == 'Successful']
        return len(p), len(a), len(lb), len(lba), len(co), len(tk), len(tkw), len(i), len(c), len(ae), len(aew)
    
    hp, hac, hlb, hlba, hcor, htk, htkw, hint, hcl, har, harw = get_stats(hteamName)
    ap, aac, alb, alba, acor, atk, atkw, aint, acl, aar, aarw = get_stats(ateamName)
    
    tot = hp + ap
    hposs = round(hp/tot*100) if tot > 0 else 50
    aposs = 100 - hposs
    
    pe = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
    stats = ['Posesion', 'Pases (Acc.)', 'Balones Largos (Acc.)', 'Corners', 'Entradas (ganadas)', 'Intercepciones', 'Despejes', 'Duelos aéreos (ganados)']
    hv = [f"{hposs}%", f"{hp}({hac})", f"{hlb}({hlba})", str(hcor), f"{htk}({htkw})", str(hint), str(hcl), f"{har}({harw})"]
    av = [f"{aposs}%", f"{ap}({aac})", f"{alb}({alba})", str(acor), f"{atk}({atkw})", str(aint), str(acl), f"{aar}({aarw})"]
    
    ax.set_title("Estadisticas Partido", color=line_color, fontsize=25, fontweight='bold')
    for i, (s, h, a) in enumerate(zip(stats, hv, av)):
        y = 55 - i * 7
        ax.text(52.5, y, s, color=line_color, fontsize=15, ha='center', va='center', fontweight='bold', path_effects=pe)
        ax.text(5, y, h, color=hcol, fontsize=17, ha='left', va='center', fontweight='bold')
        ax.text(100, y, a, color=acol, fontsize=17, ha='right', va='center', fontweight='bold')

# ========== FUNCIONES DE VISUALIZACIÓN - DASHBOARD 2 ==========
def Final_third_entry(ax, team_name, col):
    dfpass = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['x']<70) & (df['endX']>=70) & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('Freekick', na=False))]
    dfcarry = df[(df['teamName']==team_name) & (df['type']=='Carry') & (df['x']<70) & (df['endX']>=70)]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if team_name == ateamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    total = len(dfpass) + len(dfcarry)
    if total == 0:
        ax.set_title(f"{team_name}\n0 Entradas al último tercio", color=line_color, fontsize=25, fontweight='bold')
        return
    
    left = len(dfpass[dfpass['y']>=45.33]) + len(dfcarry[dfcarry['y']>=45.33])
    mid = len(dfpass[(dfpass['y']>=22.67) & (dfpass['y']<45.33)]) + len(dfcarry[(dfcarry['y']>=22.67) & (dfcarry['y']<45.33)])
    right = len(dfpass[dfpass['y']<22.67]) + len(dfcarry[dfcarry['y']<22.67])
    
    ax.hlines([22.67, 45.33], 0, 70, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.vlines(70, -2, 70, colors=line_color, linestyle='dashed', alpha=0.55)
    
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    ax.text(8, 11.335, f'{right}\n({round(right/total*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 34, f'{mid}\n({round(mid/total*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 56.675, f'{left}\n({round(left/total*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    
    pitch.lines(dfpass.x, dfpass.y, dfpass.endX, dfpass.endY, lw=2, comet=True, color=col, ax=ax, alpha=0.5)
    for _, row in dfcarry.iterrows():
        if pd.notna(row['x']) and pd.notna(row['endX']):
            ax.add_patch(patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=col, zorder=4, mutation_scale=15, alpha=0.7, linewidth=1.5, linestyle='--'))
    ax.set_title(f"{team_name}\n{total} Entradas al último tercio", color=line_color, fontsize=25, fontweight='bold')

def box_entry(ax):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    def get_entries(team):
        bentry = df[((df['type']=='Pass') | (df['type']=='Carry')) & (df['teamName']==team) & (df['outcomeType']=='Successful') & 
                    (df['endX']>=88.5) & ~((df['x']>=88.5) & (df['y']>=13.6) & (df['y']<=54.6)) & 
                    (df['endY']>=13.6) & (df['endY']<=54.4) & (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn', na=False))]
        l = len(bentry[bentry['y']>=34])
        r = len(bentry[bentry['y']<34])
        return len(bentry), l, r
    
    ht, hl, hr = get_entries(hteamName)
    at, al, ar = get_entries(ateamName)
    
    ax.scatter(30, 50, color=hcol, s=3500, edgecolor=line_color, linewidth=2, marker='h', zorder=3)
    ax.scatter(30, 18, color=hcol, s=3500, edgecolor=line_color, linewidth=2, marker='h', zorder=3)
    ax.scatter(75, 50, color=acol, s=3500, edgecolor=line_color, linewidth=2, marker='h', zorder=3)
    ax.scatter(75, 18, color=acol, s=3500, edgecolor=line_color, linewidth=2, marker='h', zorder=3)
    ax.text(30, 50, f"{hl}", color=line_color, fontsize=30, ha='center', va='center', fontweight='bold')
    ax.text(30, 18, f"{hr}", color=line_color, fontsize=30, ha='center', va='center', fontweight='bold')
    ax.text(75, 50, f"{al}", color=line_color, fontsize=30, ha='center', va='center', fontweight='bold')
    ax.text(75, 18, f"{ar}", color=line_color, fontsize=30, ha='center', va='center', fontweight='bold')
    ax.set_title(f"{hteamName}: {ht} entradas", color=hcol, fontsize=18, fontweight='bold', loc='left')
    ax.set_title(f"{ateamName}: {at} entradas", color=acol, fontsize=18, fontweight='bold', loc='right')

def zone14hs(ax, team_name, col):
    dfhp = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful') & (~df['qualifiers'].str.contains('CornerTaken|Freekick', na=False))]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if team_name == ateamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    z14 = hs = 0
    for _, row in dfhp.iterrows():
        if row['endX']>=70 and row['endX']<=88.54 and row['endY']>=22.66 and row['endY']<=45.32:
            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color='#38dacc', comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
            ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor='#38dacc', zorder=4)
            z14 += 1
        if row['endX']>=70 and ((row['endY']>=11.33 and row['endY']<=22.66) or (row['endY']>=45.32 and row['endY']<=56.95)):
            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col, comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
            ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=col, zorder=4)
            hs += 1
    
    ax.fill([70, 88.54, 88.54, 70], [22.66, 22.66, 45.32, 45.32], '#38dacc', alpha=0.2)
    ax.fill([70, 105, 105, 70], [11.33, 11.33, 22.66, 22.66], col, alpha=0.2)
    ax.fill([70, 105, 105, 70], [45.32, 45.32, 56.95, 56.95], col, alpha=0.2)
    
    ax.scatter(16.46, 13.85, color=col, s=10000, edgecolor=line_color, linewidth=2, marker='h')
    ax.scatter(16.46, 54.15, color='#38dacc', s=10000, edgecolor=line_color, linewidth=2, marker='h')
    ax.text(16.46, 13.85-4, "Carril int", fontsize=14, color=line_color, ha='center')
    ax.text(16.46, 54.15-4, "Zona14", fontsize=14, color=line_color, ha='center')
    ax.text(16.46, 13.85+2, str(hs), fontsize=35, color=line_color, ha='center', va='center', fontweight='bold')
    ax.text(16.46, 54.15+2, str(z14), fontsize=35, color=line_color, ha='center', va='center', fontweight='bold')
    ax.set_title(f"{team_name}\nPase Zona 14 y carril interior", color=line_color, fontsize=25, fontweight='bold')

def Crosses(ax):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    ax.set_ylim(-0.5, 68.5)
    
    # Obtener centros de ambos equipos
    hcrosses = df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Cross', na=False)) & (~df['qualifiers'].str.contains('CornerTaken', na=False))]
    acrosses = df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['qualifiers'].str.contains('Cross', na=False)) & (~df['qualifiers'].str.contains('CornerTaken', na=False))]
    
    # Dibujar centros del equipo local (invertidos)
    for _, row in hcrosses.iterrows():
        col_use = hcol if row['outcomeType'] == 'Successful' else 'gray'
        pitch.lines(105 - row['x'], 68 - row['y'], 105 - row['endX'], 68 - row['endY'], color=col_use, lw=2, comet=True, alpha=0.6, ax=ax)
        ax.scatter(105 - row['endX'], 68 - row['endY'], s=30, color=col_use, edgecolor=line_color, zorder=3, alpha=0.8)
    
    # Dibujar centros del equipo visitante
    for _, row in acrosses.iterrows():
        col_use = acol if row['outcomeType'] == 'Successful' else 'gray'
        pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col_use, lw=2, comet=True, alpha=0.6, ax=ax)
        ax.scatter(row['endX'], row['endY'], s=30, color=col_use, edgecolor=line_color, zorder=3, alpha=0.8)
    
    # Estadísticas
    hs = len(hcrosses[hcrosses['outcomeType']=='Successful'])
    ht = len(hcrosses)
    hl = len(hcrosses[hcrosses['y']>=45.33])
    hr = len(hcrosses[hcrosses['y']<22.67])
    
    asuc = len(acrosses[acrosses['outcomeType']=='Successful'])
    at = len(acrosses)
    al = len(acrosses[acrosses['y']>=45.33])
    ar = len(acrosses[acrosses['y']<22.67])
    
    # Textos
    ax.text(20, 55, f"Centros desde\nla Derecha: {hr}", color=hcol, fontsize=14, ha='center', fontweight='bold')
    ax.text(20, 13, f"Centros desde\nla Izquierda: {hl}", color=hcol, fontsize=14, ha='center', fontweight='bold')
    ax.text(85, 55, f"Centros desde\nla Izquierda: {al}", color=acol, fontsize=14, ha='center', fontweight='bold')
    ax.text(85, 13, f"Centros desde\nLa Derecha: {ar}", color=acol, fontsize=14, ha='center', fontweight='bold')
    ax.set_title(f"{hteamName} <---Centros", color=hcol, fontsize=20, fontweight='bold', loc='left')
    ax.set_title(f"{ateamName} Centros--->", color=acol, fontsize=20, fontweight='bold', loc='right')
    ax.text(20, -3, f"Acertados: {hs}\nFallados: {ht-hs}", color=hcol, fontsize=12, ha='center')
    ax.text(85, -3, f"Acertados: {asuc}\nFallados: {at-asuc}", color=acol, fontsize=12, ha='center')

def Pass_end_zone(ax, team_name, cm):
    pez = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful')]
    pitch = Pitch(pitch_type='uefa', line_color=line_color, corner_arcs=True, pitch_color=bg_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if team_name == ateamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    if not pez.empty:
        bs = pitch.bin_statistic(pez['endX'], pez['endY'], statistic='count', bins=(6, 5))
        pitch.heatmap(bs, ax=ax, cmap=cm, edgecolors=line_color, linewidth=0.5, alpha=0.7)
        bs['statistic'] = (bs['statistic']/bs['statistic'].sum()*100).round(0).astype(int)
        pitch.label_heatmap(bs, ax=ax, str_format='{:.0f}%', color=line_color, fontsize=12, va='center', ha='center')
    ax.set_title(f"{team_name}\nZona de finalización pase", color=line_color, fontsize=25, fontweight='bold')

def HighTO(ax):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    def get_turnovers(team):
        t = df[df['teamName']==team]
        to = t[(t['type']=='Dispossessed') | ((t['type']=='Pass') & (t['outcomeType']=='Unsuccessful'))]
        return len(to[to['x']>=70])
    
    hto = get_turnovers(hteamName)
    ato = get_turnovers(ateamName)
    
    ax.fill([0, 35, 35, 0], [0, 0, 68, 68], hcol, alpha=0.15)
    ax.fill([70, 105, 105, 70], [0, 0, 68, 68], acol, alpha=0.15)
    ax.set_title(f"{hteamName} Pérdidas zona alta: {hto}", color=hcol, fontsize=18, fontweight='bold', loc='left')
    ax.set_title(f"{ateamName} Pérdidas zona alta: {ato}", color=acol, fontsize=18, fontweight='bold', loc='right')

def Chance_creating_zone(ax, team_name, cm, col):
    ccp = df[(df['qualifiers'].str.contains('KeyPass', na=False)) & (df['teamName']==team_name)]
    pitch = Pitch(pitch_type='uefa', line_color=line_color, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if team_name == ateamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    cc = 0
    if not ccp.empty:
        bs = pitch.bin_statistic(ccp.x, ccp.y, bins=(6, 5), statistic='count', normalize=False)
        pitch.heatmap(bs, ax=ax, cmap=cm, edgecolors='#f8f8f8')
        pitch.label_heatmap(bs, color=line_color, fontsize=25, ax=ax, ha='center', va='center', str_format='{:.0f}', path_effects=path_eff)
    
    for _, row in ccp.iterrows():
        if 'IntentionalGoalAssist' in str(row['qualifiers']):
            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=green, comet=True, lw=3, zorder=3, ax=ax)
            ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=green, zorder=4)
        else:
            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, comet=True, lw=3, zorder=3, ax=ax)
            ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=violet, zorder=4)
        cc += 1
    
    if col == hcol:
        ax.text(105, -3.5, "Morado = Pase clave\nVerde = Asistencia", color=col, size=12, ha='right')
        ax.text(52.5, 70, f"Total Oportunidades creadas = {cc}", color=col, fontsize=15, ha='center')
    else:
        ax.text(105, 71.5, "Morado = Pase clave\nVerde = Asistencia", color=col, size=12, ha='left')
        ax.text(52.5, -2, f"Total Oportunidades creadas = {cc}", color=col, fontsize=15, ha='center')
    ax.set_title(f"{team_name}\nZona Oportunidades creadas", color=line_color, fontsize=25, fontweight='bold')

def plot_congestion(ax):
    pcmap = LinearSegmentedColormap.from_list("c", [acol, 'gray', hcol], N=20)
    df1 = df[(df['teamName']==hteamName) & (df['isTouch']==True) & (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn', na=False))].copy()
    df2 = df[(df['teamName']==ateamName) & (df['isTouch']==True) & (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn', na=False))].copy()
    df2['x'] = 105 - df2['x']
    df2['y'] = 68 - df2['y']
    
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, line_zorder=6)
    pitch.draw(ax=ax)
    ax.set_ylim(-0.5, 68.5)
    ax.set_xlim(-0.5, 105.5)
    
    if df1.empty or df2.empty:
        ax.set_title("Zona de dominio equipo", color=line_color, fontsize=30, fontweight='bold', y=1.075)
        return
    
    bs1 = pitch.bin_statistic(df1.x, df1.y, bins=(6, 5), statistic='count', normalize=False)
    bs2 = pitch.bin_statistic(df2.x, df2.y, bins=(6, 5), statistic='count', normalize=False)
    
    cx = np.array([[8.75, 26.25, 43.75, 61.25, 78.75, 96.25]]*5)
    cy = np.array([[61.2]*6, [47.6]*6, [34.0]*6, [20.4]*6, [6.8]*6])
    df_cong = pd.DataFrame({'cx': cx.flatten(), 'cy': cy.flatten()})
    
    hd_values = []
    for i in range(bs1['statistic'].shape[0]):
        for j in range(bs1['statistic'].shape[1]):
            s1, s2 = bs1['statistic'][i, j], bs2['statistic'][i, j]
            total = s1 + s2
            if total == 0:
                hd_values.append(0.5)
            elif (s1/total) > 0.55:
                hd_values.append(1)
            elif (s1/total) < 0.45:
                hd_values.append(0)
            else:
                hd_values.append(0.5)
    
    df_cong['hd'] = hd_values
    bs = pitch.bin_statistic(df_cong.cx, df_cong.cy, bins=(6, 5), values=df_cong['hd'], statistic='sum', normalize=False)
    pitch.heatmap(bs, ax=ax, cmap=pcmap, edgecolors='#000000', lw=0, zorder=3, alpha=0.85)
    
    try:
        ax_text(52.5, 71, s=f"<{hteamName}>  |  Disputado  |  <{ateamName}>", highlight_textprops=[{'color':hcol}, {'color':acol}], color='gray', fontsize=18, ha='center', va='center', ax=ax)
    except:
        ax.text(52.5, 71, f"{hteamName} | Disputado | {ateamName}", color='gray', fontsize=18, ha='center')
    
    ax.set_title("Zona de dominio equipo", color=line_color, fontsize=30, fontweight='bold', y=1.075)
    ax.text(0, -3, 'Direccion Ataque--->', color=hcol, fontsize=13, ha='left')
    ax.text(105, -3, '<---Direccion Ataque', color=acol, fontsize=13, ha='right')
    
    for v in range(1, 6):
        ax.vlines(v*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
    for h in range(1, 5):
        ax.hlines(h*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)

# ========== FUNCIONES DE VISUALIZACIÓN - DASHBOARD 3 ==========
def home_player_passmap(ax):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    ax.set_ylim(-0.5, 68.5)
    
    if home_progressor_df is None or home_progressor_df.empty:
        ax.set_title("Sin datos", color=line_color, fontsize=25, fontweight='bold')
        return
    
    home_player_name = home_progressor_df['name'].iloc[0]
    acc_pass = df[(df['name']==home_player_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful')]
    pro_pass = acc_pass[(acc_pass['prog_pass']>=9.11) & (acc_pass['x']>=35) & (~acc_pass['qualifiers'].str.contains('CornerTaken|Freekick', na=False))]
    pro_carry = df[(df['name']==home_player_name) & (df['prog_carry']>=9.11) & (df['endX']>=35)]
    key_pass = acc_pass[acc_pass['qualifiers'].str.contains('KeyPass', na=False)]
    g_assist = acc_pass[acc_pass['qualifiers'].str.contains('GoalAssist', na=False)]
    
    pitch.lines(acc_pass.x, acc_pass.y, acc_pass.endX, acc_pass.endY, color=line_color, lw=2, alpha=0.15, comet=True, zorder=2, ax=ax)
    pitch.lines(pro_pass.x, pro_pass.y, pro_pass.endX, pro_pass.endY, color=hcol, lw=3, alpha=1, comet=True, zorder=3, ax=ax)
    pitch.lines(key_pass.x, key_pass.y, key_pass.endX, key_pass.endY, color=violet, lw=4, alpha=1, comet=True, zorder=4, ax=ax)
    pitch.lines(g_assist.x, g_assist.y, g_assist.endX, g_assist.endY, color='green', lw=4, alpha=1, comet=True, zorder=5, ax=ax)
    
    ax.scatter(acc_pass.endX, acc_pass.endY, s=30, color=bg_color, edgecolor='gray', alpha=1, zorder=2)
    ax.scatter(pro_pass.endX, pro_pass.endY, s=40, color=bg_color, edgecolor=hcol, alpha=1, zorder=3)
    ax.scatter(key_pass.endX, key_pass.endY, s=50, color=bg_color, edgecolor=violet, alpha=1, zorder=4)
    ax.scatter(g_assist.endX, g_assist.endY, s=50, color=bg_color, edgecolor='green', alpha=1, zorder=5)
    
    for _, row in pro_carry.iterrows():
        arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=hcol, zorder=4, mutation_scale=20, alpha=0.9, linewidth=2, linestyle='--')
        ax.add_patch(arrow)
    
    home_name_show = home_progressor_df['shortName'].iloc[0]
    ax.set_title(f"{home_name_show} Mapa de pases", color=hcol, fontsize=25, fontweight='bold', y=1.03)
    ax.text(0, -3, f'Pase Prog: {len(pro_pass)}          Conduccion Prog: {len(pro_carry)}', fontsize=15, color=hcol, ha='left', va='center')

def away_player_passmap(ax):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    ax.set_ylim(-0.5, 68.5)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    if away_progressor_df is None or away_progressor_df.empty:
        ax.set_title("Sin datos", color=line_color, fontsize=25, fontweight='bold')
        return
    
    away_player_name = away_progressor_df['name'].iloc[0]
    acc_pass = df[(df['name']==away_player_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful')]
    pro_pass = acc_pass[(acc_pass['prog_pass']>=9.11) & (acc_pass['x']>=35) & (~acc_pass['qualifiers'].str.contains('CornerTaken|Freekick', na=False))]
    pro_carry = df[(df['name']==away_player_name) & (df['prog_carry']>=9.11) & (df['endX']>=35)]
    key_pass = acc_pass[acc_pass['qualifiers'].str.contains('KeyPass', na=False)]
    g_assist = acc_pass[acc_pass['qualifiers'].str.contains('GoalAssist', na=False)]
    
    pitch.lines(acc_pass.x, acc_pass.y, acc_pass.endX, acc_pass.endY, color=line_color, lw=2, alpha=0.15, comet=True, zorder=2, ax=ax)
    pitch.lines(pro_pass.x, pro_pass.y, pro_pass.endX, pro_pass.endY, color=acol, lw=3, alpha=1, comet=True, zorder=3, ax=ax)
    pitch.lines(key_pass.x, key_pass.y, key_pass.endX, key_pass.endY, color=violet, lw=4, alpha=1, comet=True, zorder=4, ax=ax)
    pitch.lines(g_assist.x, g_assist.y, g_assist.endX, g_assist.endY, color='green', lw=4, alpha=1, comet=True, zorder=5, ax=ax)
    
    ax.scatter(acc_pass.endX, acc_pass.endY, s=30, color=bg_color, edgecolor='gray', alpha=1, zorder=2)
    ax.scatter(pro_pass.endX, pro_pass.endY, s=40, color=bg_color, edgecolor=acol, alpha=1, zorder=3)
    ax.scatter(key_pass.endX, key_pass.endY, s=50, color=bg_color, edgecolor=violet, alpha=1, zorder=4)
    ax.scatter(g_assist.endX, g_assist.endY, s=50, color=bg_color, edgecolor='green', alpha=1, zorder=5)
    
    for _, row in pro_carry.iterrows():
        arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=acol, zorder=4, mutation_scale=20, alpha=0.9, linewidth=2, linestyle='--')
        ax.add_patch(arrow)
    
    away_name_show = away_progressor_df['shortName'].iloc[0]
    ax.set_title(f"{away_name_show} Mapa de pases", color=acol, fontsize=25, fontweight='bold', y=1.03)
    ax.text(0, 71, f'Pase Prog: {len(pro_pass)}          Conduccion Prog: {len(pro_carry)}', fontsize=15, color=acol, ha='right', va='center')

def home_passes_recieved(ax):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    ax.set_ylim(-0.5, 68.5)
    
    if home_average_locs_and_count_df is None or home_average_locs_and_count_df.empty:
        ax.set_title("Sin datos", color=line_color, fontsize=25, fontweight='bold')
        return
    
    fws = home_average_locs_and_count_df[home_average_locs_and_count_df['position']=='FW'] if 'position' in home_average_locs_and_count_df.columns else home_average_locs_and_count_df
    if fws.empty:
        fws = home_average_locs_and_count_df.nlargest(1, 'pass_avg_x')
    name = fws['name'].iloc[0] if len(fws) > 0 else ''
    name_show = get_short_name(name)
    
    filtered_rows = df[(df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['name'].shift(-1)==name)]
    pr = len(filtered_rows)
    
    pitch.lines(filtered_rows.x, filtered_rows.y, filtered_rows.endX, filtered_rows.endY, lw=3, transparent=True, comet=True, color=hcol, ax=ax, alpha=0.5)
    pitch.scatter(filtered_rows.endX, filtered_rows.endY, s=30, edgecolor=hcol, linewidth=1, color=bg_color, zorder=2, ax=ax)
    
    ax.set_title(f"{name_show} Pases Recibidos", color=hcol, fontsize=25, fontweight='bold', y=1.03)
    ax.text(52.5, -3, f'Pases recibidos: {pr}', color=line_color, fontsize=15, ha='center', va='center')

def away_passes_recieved(ax):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    ax.set_ylim(-0.5, 68.5)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    if away_average_locs_and_count_df is None or away_average_locs_and_count_df.empty:
        ax.set_title("Sin datos", color=line_color, fontsize=25, fontweight='bold')
        return
    
    fws = away_average_locs_and_count_df[away_average_locs_and_count_df['position']=='FW'] if 'position' in away_average_locs_and_count_df.columns else away_average_locs_and_count_df
    if fws.empty:
        fws = away_average_locs_and_count_df.nlargest(1, 'pass_avg_x')
    name = fws['name'].iloc[0] if len(fws) > 0 else ''
    name_show = get_short_name(name)
    
    filtered_rows = df[(df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['name'].shift(-1)==name)]
    pr = len(filtered_rows)
    
    pitch.lines(filtered_rows.x, filtered_rows.y, filtered_rows.endX, filtered_rows.endY, lw=3, transparent=True, comet=True, color=acol, ax=ax, alpha=0.5)
    pitch.scatter(filtered_rows.endX, filtered_rows.endY, s=30, edgecolor=acol, linewidth=1, color=bg_color, zorder=2, ax=ax)
    
    ax.set_title(f"{name_show} Pases Recibidos", color=acol, fontsize=25, fontweight='bold', y=1.03)
    ax.text(52.5, 71, f'Pases recibidos: {pr}', color=line_color, fontsize=15, ha='center', va='center')

def home_player_def_acts(ax):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    ax.set_ylim(-12, 68.5)
    
    if home_defender_df is None or home_defender_df.empty:
        ax.set_title("Sin datos", color=line_color, fontsize=25, fontweight='bold')
        return
    
    home_player_name = home_defender_df['name'].iloc[0]
    hp = df[df['name']==home_player_name]
    
    hp_tk = hp[hp['type']=='Tackle']
    hp_intc = hp[(hp['type']=='Interception') | (hp['type']=='BlockedPass')]
    hp_cl = hp[hp['type']=='Clearance']
    
    pitch.scatter(hp_tk.x, hp_tk.y, s=250, c=hcol, lw=2.5, edgecolor=hcol, marker='+', ax=ax)
    pitch.scatter(hp_intc.x, hp_intc.y, s=250, c='None', lw=2.5, edgecolor=hcol, marker='s', ax=ax)
    pitch.scatter(hp_cl.x, hp_cl.y, s=250, c='None', lw=2.5, edgecolor=hcol, marker='d', ax=ax)
    
    ax.text(5, -3, f"Entradas: {len(hp_tk)}  Intercepciones: {len(hp_intc)}  Despejes: {len(hp_cl)}", color=hcol, ha='left', va='top', fontsize=13)
    ax.set_title(f"{home_defender_df['shortName'].iloc[0]} Acciones Defensivas", color=hcol, fontsize=25, fontweight='bold')

def away_player_def_acts(ax):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    ax.set_ylim(-0.5, 80)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    if away_defender_df is None or away_defender_df.empty:
        ax.set_title("Sin datos", color=line_color, fontsize=25, fontweight='bold')
        return
    
    away_player_name = away_defender_df['name'].iloc[0]
    ap = df[df['name']==away_player_name]
    
    ap_tk = ap[ap['type']=='Tackle']
    ap_intc = ap[(ap['type']=='Interception') | (ap['type']=='BlockedPass')]
    ap_cl = ap[ap['type']=='Clearance']
    
    pitch.scatter(ap_tk.x, ap_tk.y, s=250, c=acol, lw=2.5, edgecolor=acol, marker='+', ax=ax)
    pitch.scatter(ap_intc.x, ap_intc.y, s=250, c='None', lw=2.5, edgecolor=acol, marker='s', ax=ax)
    pitch.scatter(ap_cl.x, ap_cl.y, s=250, c='None', lw=2.5, edgecolor=acol, marker='d', ax=ax)
    
    ax.text(100, 71, f"Entradas: {len(ap_tk)}  Intercepciones: {len(ap_intc)}  Despejes: {len(ap_cl)}", color=acol, ha='right', va='top', fontsize=13)
    ax.set_title(f"{away_defender_df['shortName'].iloc[0]} Acciones Defensivas", color=acol, fontsize=25, fontweight='bold')

def home_gk(ax):
    df_gk = df[(df['teamName']==hteamName) & (df['position']=='GK')]
    if df_gk.empty:
        ax.set_facecolor(bg_color)
        ax.set_title("Sin datos PT", color=hcol, fontsize=25, fontweight='bold')
        return
    
    gk_pass = df_gk[df_gk['type']=='Pass']
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    ax.set_ylim(-0.5, 68.5)
    gk_name = df_gk['shortName'].iloc[0] if 'shortName' in df_gk.columns else get_short_name(df_gk['name'].iloc[0])
    
    for _, row in gk_pass.iterrows():
        col_use = hcol if row['outcomeType']=='Successful' else 'gray'
        pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col_use, lw=3, comet=True, alpha=0.5, zorder=2, ax=ax)
        ax.scatter(row['endX'], row['endY'], s=30, color=col_use, edgecolor=line_color, zorder=3)
    
    ax.set_title(f'{gk_name} Mapa Pases PT', color=hcol, fontsize=25, fontweight='bold', y=1.03)

def away_gk(ax):
    df_gk = df[(df['teamName']==ateamName) & (df['position']=='GK')]
    if df_gk.empty:
        ax.set_facecolor(bg_color)
        ax.set_title("Sin datos PT", color=acol, fontsize=25, fontweight='bold')
        return
    
    gk_pass = df_gk[df_gk['type']=='Pass']
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    ax.set_ylim(-0.5, 68.5)
    ax.invert_xaxis()
    ax.invert_yaxis()
    gk_name = df_gk['shortName'].iloc[0] if 'shortName' in df_gk.columns else get_short_name(df_gk['name'].iloc[0])
    
    for _, row in gk_pass.iterrows():
        col_use = acol if row['outcomeType']=='Successful' else 'gray'
        pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col_use, lw=3, comet=True, alpha=0.5, zorder=2, ax=ax)
        ax.scatter(row['endX'], row['endY'], s=30, color=col_use, edgecolor=line_color, zorder=3)
    
    ax.set_title(f'{gk_name} Mapa Pases PT', color=acol, fontsize=25, fontweight='bold', y=1.03)

def sh_sq_bar(ax):
    if sh_sq_df is None or sh_sq_df.empty:
        ax.set_facecolor(bg_color)
        ax.set_title("Sin datos", color=line_color, fontsize=25, fontweight='bold')
        return
    
    top10 = sh_sq_df.nsmallest(10, 'total')['shortName'].tolist()
    shsq_sh = sh_sq_df.nsmallest(10, 'total')['Shots'].tolist()
    shsq_sa = sh_sq_df.nsmallest(10, 'total')['Shot Assist'].tolist()
    shsq_bs = sh_sq_df.nsmallest(10, 'total')['Buildup to shot'].tolist()
    
    left1 = [w + x for w, x in zip(shsq_sh, shsq_sa)]
    ax.barh(top10, shsq_sh, label='Tiro', color=col1, left=0)
    ax.barh(top10, shsq_sa, label='Asistencia tiro', color=violet, left=shsq_sh)
    ax.barh(top10, shsq_bs, label='Construcción tiro', color=col2, left=left1)
    
    ax.set_facecolor(bg_color)
    ax.tick_params(axis='x', colors=line_color, labelsize=15)
    ax.tick_params(axis='y', colors=line_color, labelsize=15)
    for spine in ax.spines.values():
        spine.set_edgecolor(bg_color)
    ax.set_title("Participación en secuencias de tiros", color=line_color, fontsize=25, fontweight='bold')
    ax.legend()

def passer_bar(ax):
    if progressor_df is None or progressor_df.empty:
        ax.set_facecolor(bg_color)
        ax.set_title("Sin datos", color=line_color, fontsize=25, fontweight='bold')
        return
    
    top10 = progressor_df.nsmallest(10, 'total')['shortName'].tolist()
    pp = progressor_df.nsmallest(10, 'total')['Progressive Passes'].tolist()
    pc = progressor_df.nsmallest(10, 'total')['Progressive Carries'].tolist()
    lb = progressor_df.nsmallest(10, 'total')['LineBreaking Pass'].tolist()
    
    left1 = [w + x for w, x in zip(pp, pc)]
    ax.barh(top10, pp, label='Pases Progresivos', color=col1, left=0)
    ax.barh(top10, pc, label='Conducciones progresivas', color=col2, left=pp)
    ax.barh(top10, lb, label='Pase filtrado', color=violet, left=left1)
    
    ax.set_facecolor(bg_color)
    ax.tick_params(axis='x', colors=line_color, labelsize=15)
    ax.tick_params(axis='y', colors=line_color, labelsize=15)
    for spine in ax.spines.values():
        spine.set_edgecolor(bg_color)
    ax.set_title("Top 10 progresores de balón", color=line_color, fontsize=25, fontweight='bold')
    ax.legend()

def defender_bar(ax):
    if defender_df is None or defender_df.empty:
        ax.set_facecolor(bg_color)
        ax.set_title("Sin datos", color=line_color, fontsize=25, fontweight='bold')
        return
    
    top10 = defender_df.nsmallest(10, 'total')['shortName'].tolist()
    tk = defender_df.nsmallest(10, 'total')['Tackles'].tolist()
    int_ = defender_df.nsmallest(10, 'total')['Interceptions'].tolist()
    cl = defender_df.nsmallest(10, 'total')['Clearance'].tolist()
    
    left1 = [w + x for w, x in zip(tk, int_)]
    ax.barh(top10, tk, label='Entradas', color=col1, left=0)
    ax.barh(top10, int_, label='Intercepciones', color=violet, left=tk)
    ax.barh(top10, cl, label='Despejes', color=col2, left=left1)
    
    ax.set_facecolor(bg_color)
    ax.tick_params(axis='x', colors=line_color, labelsize=15)
    ax.tick_params(axis='y', colors=line_color, labelsize=15)
    for spine in ax.spines.values():
        spine.set_edgecolor(bg_color)
    ax.set_title("Top 10 Defensores", color=line_color, fontsize=25, fontweight='bold')
    ax.legend()

def threat_creators(ax):
    if xT_df is None or xT_df.empty:
        ax.set_facecolor(bg_color)
        ax.set_title("Sin datos xT", color=line_color, fontsize=25, fontweight='bold')
        return
    
    top10 = xT_df.nsmallest(10, 'total')['shortName'].tolist()
    xtp = xT_df.nsmallest(10, 'total')['xT from Pass'].tolist()
    xtc = xT_df.nsmallest(10, 'total')['xT from Carry'].tolist()
    
    ax.barh(top10, xtp, label='xT Pases', color=col1, left=0)
    ax.barh(top10, xtc, label='xT Conducción', color=violet, left=xtp)
    
    ax.set_facecolor(bg_color)
    ax.tick_params(axis='x', colors=line_color, labelsize=15)
    ax.tick_params(axis='y', colors=line_color, labelsize=15)
    for spine in ax.spines.values():
        spine.set_edgecolor(bg_color)
    ax.set_title("Top 10 jugadores más peligrosos", color=line_color, fontsize=25, fontweight='bold')
    ax.legend()

# ========== GENERADORES DE DASHBOARDS ==========
def generate_dashboard_1():
    fig, axs = plt.subplots(4, 3, figsize=(35, 35), facecolor=bg_color)
    
    pass_network_visualization(axs[0,0], home_passes_between_df, home_average_locs_and_count_df, hcol, hteamName)
    plot_shotmap(axs[0,1])
    pass_network_visualization(axs[0,2], away_passes_between_df, away_average_locs_and_count_df, acol, ateamName)
    
    defensive_block(axs[1,0], defensive_home_average_locs_and_count_df, hteamName, hcol)
    plot_goalPost(axs[1,1])
    defensive_block(axs[1,2], defensive_away_average_locs_and_count_df, ateamName, acol)
    
    draw_progressive_pass_map(axs[2,0], hteamName, hcol)
    plot_Momentum(axs[2,1])
    draw_progressive_pass_map(axs[2,2], ateamName, acol)
    
    draw_progressive_carry_map(axs[3,0], hteamName, hcol)
    plotting_match_stats(axs[3,1])
    draw_progressive_carry_map(axs[3,2], ateamName, acol)
    
    # Heading
    highlight_text = [{'color':hcol}, {'color':acol}]
    fig_text(0.5, 0.98, f"<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>", color=line_color, fontsize=70, fontweight='bold',
             highlight_textprops=highlight_text, ha='center', va='center', ax=fig)
    fig.text(0.5, 0.95, "Post Partido Informe-1", color=line_color, fontsize=30, ha='center', va='center')
    fig.text(0.5, 0.93, "Analista John Triguero", color=line_color, fontsize=22.5, ha='center', va='center')
    fig.text(0.125, 0.1, 'Direccion Ataque ------->', color=hcol, fontsize=25, ha='left', va='center')
    fig.text(0.9, 0.1, '<------- Direccion Ataque', color=acol, fontsize=25, ha='right', va='center')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig

def generate_dashboard_2():
    fig, axs = plt.subplots(4, 3, figsize=(35, 35), facecolor=bg_color)
    
    Final_third_entry(axs[0,0], hteamName, hcol)
    box_entry(axs[0,1])
    Final_third_entry(axs[0,2], ateamName, acol)
    
    zone14hs(axs[1,0], hteamName, hcol)
    Crosses(axs[1,1])
    zone14hs(axs[1,2], ateamName, acol)
    
    Pass_end_zone(axs[2,0], hteamName, pearl_earring_cmaph)
    HighTO(axs[2,1])
    Pass_end_zone(axs[2,2], ateamName, pearl_earring_cmapa)
    
    Chance_creating_zone(axs[3,0], hteamName, pearl_earring_cmaph, hcol)
    plot_congestion(axs[3,1])
    Chance_creating_zone(axs[3,2], ateamName, pearl_earring_cmapa, acol)
    
    # Heading
    highlight_text = [{'color':hcol}, {'color':acol}]
    fig_text(0.5, 0.98, f"<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>", color=line_color, fontsize=70, fontweight='bold',
             highlight_textprops=highlight_text, ha='center', va='center', ax=fig)
    fig.text(0.5, 0.95, "Post Partido Informe-2", color=line_color, fontsize=30, ha='center', va='center')
    fig.text(0.5, 0.93, "Analista John Triguero", color=line_color, fontsize=22.5, ha='center', va='center')
    fig.text(0.125, 0.1, 'Direccion Ataque ------->', color=hcol, fontsize=25, ha='left', va='center')
    fig.text(0.9, 0.1, '<------- Direccion Ataque', color=acol, fontsize=25, ha='right', va='center')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig

def generate_dashboard_3():
    fig, axs = plt.subplots(4, 3, figsize=(35, 35), facecolor=bg_color)
    
    home_player_passmap(axs[0,0])
    passer_bar(axs[0,1])
    away_player_passmap(axs[0,2])
    
    home_passes_recieved(axs[1,0])
    sh_sq_bar(axs[1,1])
    away_passes_recieved(axs[1,2])
    
    home_player_def_acts(axs[2,0])
    defender_bar(axs[2,1])
    away_player_def_acts(axs[2,2])
    
    home_gk(axs[3,0])
    threat_creators(axs[3,1])
    away_gk(axs[3,2])
    
    # Heading
    highlight_text = [{'color':hcol}, {'color':acol}]
    fig_text(0.5, 0.98, f"<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>", color=line_color, fontsize=70, fontweight='bold',
             highlight_textprops=highlight_text, ha='center', va='center', ax=fig)
    fig.text(0.5, 0.95, "Top Jugadores del Partido", color=line_color, fontsize=30, ha='center', va='center')
    fig.text(0.5, 0.93, "Analista John Triguero", color=line_color, fontsize=22.5, ha='center', va='center')
    fig.text(0.125, 0.097, 'Direccion Ataque ------->', color=hcol, fontsize=25, ha='left', va='center')
    fig.text(0.9, 0.097, '<------- Direccion Ataque', color=acol, fontsize=25, ha='right', va='center')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig

# ========== INTERFAZ STREAMLIT ==========
def main():
    global hcol, acol, bg_color, line_color
    
    st.title("⚽ Zona del Analista")
    st.markdown("**Dashboard de análisis de partidos WhoScored**")
    
    with st.sidebar:
        st.header("Configuración")
        uploaded_file = st.file_uploader("Sube archivo HTML de WhoScored", type=['html'])
        
        st.subheader("Colores")
        hcol = st.color_picker("Color Equipo Local", "#ff4b44")
        acol = st.color_picker("Color Equipo Visitante", "#00FFD5")
        bg_color = st.color_picker("Color Fondo", "#363d4d")
        line_color = st.color_picker("Color Líneas", "#ffffff")
    
    if uploaded_file is not None:
        try:
            html_content = uploaded_file.read().decode('utf-8')
            json_data_txt = extract_json_from_html(html_content)
            data = json.loads(json_data_txt)
            events_dict, players_df, teams_dict = extract_data_from_dict(data)
            
            with st.spinner("Procesando datos..."):
                process_all_data(events_dict, players_df, teams_dict)
            
            st.success(f"✅ Partido cargado: **{hteamName} {hgoal_count} - {agoal_count} {ateamName}**")
            
            tab1, tab2, tab3 = st.tabs(["📊 Dashboard 1", "📈 Dashboard 2", "👥 Dashboard 3"])
            
            with tab1:
                st.subheader("Informe Post-Partido 1")
                with st.spinner("Generando Dashboard 1..."):
                    fig1 = generate_dashboard_1()
                    st.pyplot(fig1)
                    buf1 = BytesIO()
                    fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight', facecolor=bg_color)
                    st.download_button("📥 Descargar Dashboard 1", buf1.getvalue(), f"{hteamName}_vs_{ateamName}_Dashboard1.png", "image/png")
                    plt.close(fig1)
            
            with tab2:
                st.subheader("Informe Post-Partido 2")
                with st.spinner("Generando Dashboard 2..."):
                    fig2 = generate_dashboard_2()
                    st.pyplot(fig2)
                    buf2 = BytesIO()
                    fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor=bg_color)
                    st.download_button("📥 Descargar Dashboard 2", buf2.getvalue(), f"{hteamName}_vs_{ateamName}_Dashboard2.png", "image/png")
                    plt.close(fig2)
            
            with tab3:
                st.subheader("Top Jugadores del Partido")
                with st.spinner("Generando Dashboard 3..."):
                    fig3 = generate_dashboard_3()
                    st.pyplot(fig3)
                    buf3 = BytesIO()
                    fig3.savefig(buf3, format='png', dpi=150, bbox_inches='tight', facecolor=bg_color)
                    st.download_button("📥 Descargar Dashboard 3", buf3.getvalue(), f"{hteamName}_vs_{ateamName}_Dashboard3.png", "image/png")
                    plt.close(fig3)
                    
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("👈 Sube un archivo HTML de WhoScored para comenzar")
        st.markdown("""
        ### Instrucciones:
        1. Ve a WhoScored.com y abre un partido
        2. Guarda la página como HTML (Ctrl+S)
        3. Sube el archivo aquí
        4. ¡Genera tus dashboards!
        """)

if __name__ == "__main__":
    main()
