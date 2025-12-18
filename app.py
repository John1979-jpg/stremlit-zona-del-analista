# -*- coding: utf-8 -*-
"""
Zona del Analista - WhoScored Dashboard Generator
Genera 3 dashboards completos de análisis de partidos exactamente como las imágenes de referencia
"""

import streamlit as st
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import matplotlib.patches as patches
from io import BytesIO
from mplsoccer import Pitch, VerticalPitch, add_image
import matplotlib.patheffects as path_effects
from unidecode import unidecode
from datetime import date
from PIL import Image
from urllib.request import urlopen
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Zona del Analista", page_icon="⚽", layout="wide")

# ============== FUNCIONES DE EXTRACCIÓN ==============
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

def process_dataframe(df, teams_dict):
    df['type'] = df['type'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    df['outcomeType'] = df['outcomeType'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    df['period'] = df['period'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    df['period'] = df['period'].replace({'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3, 'SecondPeriodOfExtraTime': 4, 'PenaltyShootout': 5})
    df['period'] = pd.to_numeric(df['period'], errors='coerce')
    df['teamName'] = df['teamId'].map(teams_dict)
    
    df['x'] = pd.to_numeric(df['x'], errors='coerce') * 1.05
    df['y'] = pd.to_numeric(df['y'], errors='coerce') * 0.68
    df['endX'] = pd.to_numeric(df.get('endX'), errors='coerce') * 1.05 if 'endX' in df.columns else np.nan
    df['endY'] = pd.to_numeric(df.get('endY'), errors='coerce') * 0.68 if 'endY' in df.columns else np.nan
    
    if 'qualifiers' in df.columns:
        df['qualifiers'] = df['qualifiers'].astype(str)
    
    df['pass_or_carry_angle'] = np.degrees(np.arctan2(df['endY'] - df['y'], df['endX'] - df['x']))
    df['prog_pass'] = np.where((df['type'] == 'Pass'),
                               np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['prog_carry'] = np.where((df['type'] == 'Carry'),
                                np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    
    df['cumulative_mins'] = df['minute'] + (1/60) * df['second']
    for period in df['period'].unique():
        if pd.notna(period) and int(period) > 1:
            mask_prev = df['period'] == period - 1
            mask_curr = df['period'] == period
            if mask_prev.any() and mask_curr.any():
                t_delta = df.loc[mask_prev, 'cumulative_mins'].max() - df.loc[mask_curr, 'cumulative_mins'].min()
                df.loc[mask_curr, 'cumulative_mins'] += t_delta
    return df

def insert_ball_carries(events_df):
    match_events = events_df.reset_index(drop=True)
    match_carries = pd.DataFrame()
    
    for idx in range(len(match_events) - 1):
        match_event = match_events.loc[idx]
        next_evt = match_events.loc[idx + 1]
        
        if 'endX' not in match_event or pd.isna(match_event.get('endX')):
            continue
        
        same_team = match_event['teamId'] == next_evt['teamId']
        not_ball_touch = next_evt['type'] not in ['BallTouch', 'TakeOn', 'Foul']
        dx = next_evt['x'] - match_event['endX']
        dy = next_evt['y'] - match_event['endY']
        far_enough = dx ** 2 + dy ** 2 >= 9
        not_too_far = dx ** 2 + dy ** 2 <= 3600
        dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
        same_phase = dt < 10 and dt >= 1
        same_period = match_event['period'] == next_evt['period']
        
        if same_team and not_ball_touch and far_enough and not_too_far and same_phase and same_period:
            carry = pd.DataFrame([{
                'minute': match_event['minute'], 'second': match_event['second'],
                'teamId': next_evt['teamId'], 'teamName': next_evt['teamName'],
                'x': match_event['endX'], 'y': match_event['endY'],
                'endX': next_evt['x'], 'endY': next_evt['y'],
                'type': 'Carry', 'outcomeType': 'Successful',
                'period': next_evt['period'], 'playerId': next_evt.get('playerId'),
                'name': next_evt.get('name'), 'shortName': next_evt.get('shortName'),
                'cumulative_mins': (match_event['cumulative_mins'] + next_evt['cumulative_mins']) / 2,
                'prog_carry': np.sqrt((105 - match_event['endX'])**2 + (34 - match_event['endY'])**2) - 
                              np.sqrt((105 - next_evt['x'])**2 + (34 - next_evt['y'])**2)
            }])
            match_carries = pd.concat([match_carries, carry], ignore_index=True)
    
    events_out = pd.concat([match_carries, match_events], ignore_index=True)
    events_out = events_out.sort_values(['period', 'cumulative_mins']).reset_index(drop=True)
    return events_out

def get_passes_df(df):
    df1 = df[~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card', na=False)]
    df1 = df1.copy()
    df1.loc[:, "receiver"] = df1["playerId"].shift(-1)
    passes_ids = df1.index[df1['type'] == 'Pass']
    df_passes = df1.loc[passes_ids, ["x", "y", "endX", "endY", "teamName", "playerId", "receiver", "type", "outcomeType", "pass_or_carry_angle"]]
    return df_passes

def get_passes_between_df(teamName, passes_df, df, players_df):
    passes_df = passes_df[(passes_df["teamName"] == teamName)].copy()
    dfteam = df[(df['teamName'] == teamName) & (~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card', na=False))]
    passes_df = passes_df.merge(players_df[["playerId", "isFirstEleven"]], on='playerId', how='left')
    
    average_locs_and_count_df = (dfteam.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']}))
    average_locs_and_count_df.columns = ['pass_avg_x', 'pass_avg_y', 'count']
    average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left')
    average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')
    
    passes_player_ids_df = passes_df.loc[:, ['playerId', 'receiver', 'teamName']].copy()
    passes_player_ids_df['pos_max'] = passes_player_ids_df[['playerId', 'receiver']].max(axis='columns')
    passes_player_ids_df['pos_min'] = passes_player_ids_df[['playerId', 'receiver']].min(axis='columns')
    passes_between_df = passes_player_ids_df.groupby(['pos_min', 'pos_max']).size().reset_index(name='pass_count')
    
    passes_between_df = passes_between_df.merge(average_locs_and_count_df[['pass_avg_x', 'pass_avg_y', 'name']], left_on='pos_min', right_index=True)
    passes_between_df = passes_between_df.merge(average_locs_and_count_df[['pass_avg_x', 'pass_avg_y', 'name']], left_on='pos_max', right_index=True, suffixes=['', '_end'])
    
    return passes_between_df, average_locs_and_count_df

def get_defensive_action_df(df):
    defensive_actions_ids = df.index[
        ((df['type'] == 'Aerial') & (df['qualifiers'].str.contains('Defensive', na=False))) |
        (df['type'] == 'BallRecovery') | (df['type'] == 'BlockedPass') |
        (df['type'] == 'Challenge') | (df['type'] == 'Clearance') |
        (df['type'] == 'Foul') | (df['type'] == 'Interception') | (df['type'] == 'Tackle')
    ]
    df_defensive_actions = df.loc[defensive_actions_ids, ["x", "y", "teamName", "playerId", "type", "outcomeType"]]
    return df_defensive_actions

def get_da_count_df(team_name, defensive_actions_df, players_df):
    defensive_actions_df = defensive_actions_df[defensive_actions_df["teamName"] == team_name]
    defensive_actions_df = defensive_actions_df.merge(players_df[["playerId", "isFirstEleven"]], on='playerId', how='left')
    average_locs_and_count_df = defensive_actions_df.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']})
    average_locs_and_count_df.columns = ['x', 'y', 'count']
    average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left')
    average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')
    return average_locs_and_count_df

# ============== FUNCIONES DASHBOARD 1 ==============
def pass_network_visualization(ax, passes_between_df, average_locs_and_count_df, col, teamName, hteamName, bg_color, line_color, passes_df):
    MAX_LINE_WIDTH = 15
    MAX_MARKER_SIZE = 3000
    passes_between_df = passes_between_df.copy()
    passes_between_df['width'] = (passes_between_df.pass_count / passes_between_df.pass_count.max() * MAX_LINE_WIDTH)
    
    color = np.array(to_rgba(col))
    color = np.tile(color, (len(passes_between_df), 1))
    c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
    c_transparency = (c_transparency * 0.8) + 0.1
    color[:, 3] = c_transparency

    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)

    pitch.lines(passes_between_df.pass_avg_x, passes_between_df.pass_avg_y, 
               passes_between_df.pass_avg_x_end, passes_between_df.pass_avg_y_end,
               lw=passes_between_df.width, color=color, zorder=1, ax=ax)

    for index, row in average_locs_and_count_df.iterrows():
        marker = 'o' if row.get('isFirstEleven', True) else 's'
        alpha = 1 if row.get('isFirstEleven', True) else 0.75
        pitch.scatter(row['pass_avg_x'], row['pass_avg_y'], s=1000, marker=marker, 
                     color=bg_color, edgecolor=line_color, linewidth=2, alpha=alpha, ax=ax)
        pitch.annotate(str(int(row["shirtNo"])) if pd.notna(row.get("shirtNo")) else '', 
                      xy=(row.pass_avg_x, row.pass_avg_y), c=col, ha='center', va='center', size=18, ax=ax)

    avgph = round(average_locs_and_count_df['pass_avg_x'].median(), 2)
    ax.axvline(x=avgph, color='gray', linestyle='--', alpha=0.75, linewidth=2)
    
    center_backs = average_locs_and_count_df[average_locs_and_count_df['position'] == 'DC']
    def_line_h = round(center_backs['pass_avg_x'].median(), 2) if not center_backs.empty else avgph
    forwards = average_locs_and_count_df[average_locs_and_count_df['isFirstEleven'] == 1].nlargest(2, 'pass_avg_x')
    fwd_line_h = round(forwards['pass_avg_x'].mean(), 2) if not forwards.empty else avgph
    
    ax.fill([def_line_h, fwd_line_h, fwd_line_h, def_line_h], [0, 0, 68, 68], col, alpha=0.1)
    
    team_passes = passes_df[passes_df["teamName"] == teamName].copy()
    team_passes['pass_or_carry_angle'] = team_passes['pass_or_carry_angle'].abs()
    team_passes = team_passes[(team_passes['pass_or_carry_angle'] >= 0) & (team_passes['pass_or_carry_angle'] <= 90)]
    verticality = round((1 - team_passes['pass_or_carry_angle'].median() / 90) * 100, 2) if not team_passes.empty else 0

    if teamName != hteamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.text(avgph - 1, 73, f"{avgph}m", fontsize=15, color=line_color, ha='left')
        ax.text(105, 73, f"verticalidad: {verticality}%", fontsize=15, color=line_color, ha='left')
        ax.text(2, 2, "circulo = Titulares\ncuadro= suplentes", color=col, size=12, ha='right', va='top')
        ax.set_title(f"{teamName}\nRed de pases", color=line_color, size=25, fontweight='bold')
    else:
        ax.text(avgph - 1, -5, f"{avgph}m", fontsize=15, color=line_color, ha='right')
        ax.text(105, -5, f"verticalidad: {verticality}%", fontsize=15, color=line_color, ha='right')
        ax.text(2, 66, "circulo = Titulares\ncuadro= suplentes", color=col, size=12, ha='left', va='top')
        ax.set_title(f"{teamName}\nRed de Pases", color=line_color, size=25, fontweight='bold')

def plot_shotmap(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, linewidth=2, line_color=line_color)
    pitch.draw(ax=ax)
    ax.set_ylim(-0.5, 68.5)
    ax.set_xlim(-0.5, 105.5)
    
    shot_types = ['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost']
    hShotsdf = df[(df['teamName'] == hteamName) & (df['type'].isin(shot_types))]
    aShotsdf = df[(df['teamName'] == ateamName) & (df['type'].isin(shot_types))]
    
    for shots, col, flip in [(hShotsdf, hcol, True), (aShotsdf, acol, False)]:
        goals = shots[shots['type'] == 'Goal']
        posts = shots[shots['type'] == 'ShotOnPost']
        saves = shots[(shots['type'] == 'SavedShot') & (~shots['qualifiers'].str.contains('BigChance', na=False))]
        misses = shots[(shots['type'] == 'MissedShots') & (~shots['qualifiers'].str.contains('BigChance', na=False))]
        
        if flip:
            pitch.scatter((105 - goals.x), (68 - goals.y), s=350, edgecolors='white', c='None', marker='football', zorder=3, ax=ax)
            pitch.scatter((105 - posts.x), (68 - posts.y), s=200, edgecolors=col, c=col, marker='o', ax=ax)
            pitch.scatter((105 - saves.x), (68 - saves.y), s=200, edgecolors=col, c='None', hatch='///////', marker='o', ax=ax)
            pitch.scatter((105 - misses.x), (68 - misses.y), s=200, edgecolors=col, c='None', marker='o', ax=ax)
        else:
            pitch.scatter(goals.x, goals.y, s=350, edgecolors='white', c='None', marker='football', zorder=3, ax=ax)
            pitch.scatter(posts.x, posts.y, s=200, edgecolors=col, c=col, marker='o', ax=ax)
            pitch.scatter(saves.x, saves.y, s=200, edgecolors=col, c='None', hatch='///////', marker='o', ax=ax)
            pitch.scatter(misses.x, misses.y, s=200, edgecolors=col, c='None', marker='o', ax=ax)
    
    ax.text(0, 70, f"{hteamName}\n<---Tiros", color=hcol, size=25, ha='left', fontweight='bold')
    ax.text(105, 70, f"{ateamName}\nTiros--->", color=acol, size=25, ha='right', fontweight='bold')

def defensive_block(ax, average_locs_and_count_df, defensive_actions_df, team_name, col, hteamName, bg_color, line_color):
    defensive_actions_team_df = defensive_actions_df[defensive_actions_df["teamName"] == team_name]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, line_zorder=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_facecolor(bg_color)
    ax.set_xlim(-0.5, 105.5)

    MAX_MARKER_SIZE = 3500
    average_locs_and_count_df = average_locs_and_count_df.copy()
    average_locs_and_count_df['marker_size'] = average_locs_and_count_df['count'] / average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE
    
    flamingo_cmap = LinearSegmentedColormap.from_list("custom", [bg_color, col], N=500)
    if len(defensive_actions_team_df) > 2:
        try:
            pitch.kdeplot(defensive_actions_team_df.x, defensive_actions_team_df.y, ax=ax, fill=True, levels=5000, thresh=0.02, cut=4, cmap=flamingo_cmap)
        except:
            pass

    for index, row in average_locs_and_count_df.iterrows():
        marker = 'o' if row.get('isFirstEleven', True) else 's'
        pitch.scatter(row['x'], row['y'], s=row['marker_size'] + 100, marker=marker,
                     color=bg_color, edgecolor=line_color, linewidth=1, alpha=1, zorder=3, ax=ax)
        pitch.annotate(str(int(row["shirtNo"])) if pd.notna(row.get("shirtNo")) else '',
                      xy=(row.x, row.y), c=line_color, ha='center', va='center', size=14, ax=ax)

    dah = round(average_locs_and_count_df['x'].mean(), 2)
    ax.axvline(x=dah, color='gray', linestyle='--', alpha=0.75, linewidth=2)
    
    center_backs = average_locs_and_count_df[average_locs_and_count_df['position'] == 'DC']
    def_line_h = round(center_backs['x'].median(), 2) if not center_backs.empty else dah
    forwards = average_locs_and_count_df[average_locs_and_count_df['isFirstEleven'] == 1].nlargest(2, 'x')
    fwd_line_h = round(forwards['x'].mean(), 2) if not forwards.empty else dah
    compactness = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2) if fwd_line_h != def_line_h else 0

    if team_name != hteamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.text(dah - 1, 73, f"{dah}m", fontsize=15, color=line_color, ha='left', va='center')
        ax.text(105, 73, f'Defensa Compacta: {compactness}%', fontsize=15, color=line_color, ha='left', va='center')
        ax.text(2, 2, "círculo = titular\ncuadro = suplente", color='gray', size=12, ha='right', va='top')
        ax.set_title(f"{team_name}\nBloque Defensivo", color=line_color, fontsize=25, fontweight='bold')
    else:
        ax.text(dah - 1, -5, f"{dah}m", fontsize=15, color=line_color, ha='right', va='center')
        ax.text(105, -5, f'Defensa Compacta: {compactness}%', fontsize=15, color=line_color, ha='right', va='center')
        ax.text(2, 66, "círculo = titular\ncuadro = suplente", color='gray', size=12, ha='left', va='top')
        ax.set_title(f"{team_name}\nBloque Defensivo", color=line_color, fontsize=25, fontweight='bold')

def plot_goalPost(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=bg_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_ylim(-0.5, 68.5)
    ax.set_xlim(-0.5, 105.5)

    # Portería visitante (arriba)
    ax.plot([7.5, 7.5], [0, 30], color=line_color, linewidth=5)
    ax.plot([7.5, 97.5], [30, 30], color=line_color, linewidth=5)
    ax.plot([97.5, 97.5], [30, 0], color=line_color, linewidth=5)
    ax.plot([0, 105], [0, 0], color=line_color, linewidth=3)
    for y in np.arange(0, 6) * 6:
        ax.plot([7.5, 97.5], [y, y], color=line_color, linewidth=2, alpha=0.2)
    for x in (np.arange(0, 11) * 9) + 7.5:
        ax.plot([x, x], [0, 30], color=line_color, linewidth=2, alpha=0.2)
    
    # Portería local (abajo)
    ax.plot([7.5, 7.5], [38, 68], color=line_color, linewidth=5)
    ax.plot([7.5, 97.5], [68, 68], color=line_color, linewidth=5)
    ax.plot([97.5, 97.5], [68, 38], color=line_color, linewidth=5)
    ax.plot([0, 105], [38, 38], color=line_color, linewidth=3)
    for y in (np.arange(0, 6) * 6) + 38:
        ax.plot([7.5, 97.5], [y, y], color=line_color, linewidth=2, alpha=0.2)
    for x in (np.arange(0, 11) * 9) + 7.5:
        ax.plot([x, x], [38, 68], color=line_color, linewidth=2, alpha=0.2)

    shot_types = ['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost']
    hShotsdf = df[(df['teamName'] == hteamName) & (df['type'].isin(shot_types))].copy()
    aShotsdf = df[(df['teamName'] == ateamName) & (df['type'].isin(shot_types))].copy()
    
    hSaves = len(hShotsdf[(hShotsdf['type'] == 'SavedShot')])
    aSaves = len(aShotsdf[(aShotsdf['type'] == 'SavedShot')])
    hGoals = len(hShotsdf[hShotsdf['type'] == 'Goal'])
    aGoals = len(aShotsdf[aShotsdf['type'] == 'Goal'])

    ax.text(52.5, 70, f"{hteamName} PT Paradas", color=hcol, fontsize=25, ha='center', fontweight='bold')
    ax.text(52.5, -2, f"{ateamName} PT Paradas", color=acol, fontsize=25, ha='center', va='top', fontweight='bold')
    ax.text(100, 68, f"Paradas = {aSaves}\n\nGoles Evitados:\n{aSaves}", color=hcol, fontsize=14, va='top', ha='left')
    ax.text(100, 2, f"Paradas = {hSaves}\n\nGoles Evitados:\n{hSaves}", color=acol, fontsize=14, va='bottom', ha='left')

def draw_progressive_pass_map(ax, df, team_name, col, hteamName, bg_color, line_color):
    dfpro = df[(df['teamName'] == team_name) & (df['prog_pass'] >= 9.11) & 
               (~df['qualifiers'].str.contains('CornerTaken|Freekick', na=False)) &
               (df['x'] >= 35) & (df['outcomeType'] == 'Successful')]
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)

    if team_name != hteamName:
        ax.invert_xaxis()
        ax.invert_yaxis()

    pro_count = len(dfpro)
    if pro_count == 0:
        ax.set_title(f"{team_name}\n0 Pases Progresivos", color=line_color, fontsize=25, fontweight='bold')
        return

    left_pro = len(dfpro[dfpro['y'] >= 45.33])
    mid_pro = len(dfpro[(dfpro['y'] >= 22.67) & (dfpro['y'] < 45.33)])
    right_pro = len(dfpro[dfpro['y'] < 22.67])

    ax.hlines(22.67, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.hlines(45.33, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)

    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    ax.text(8, 11.335, f'{right_pro}\n({round(right_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 34, f'{mid_pro}\n({round(mid_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 56.675, f'{left_pro}\n({round(left_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)

    pitch.lines(dfpro.x, dfpro.y, dfpro.endX, dfpro.endY, lw=3.5, comet=True, color=col, ax=ax, alpha=0.5)
    pitch.scatter(dfpro.endX, dfpro.endY, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2, ax=ax)

    ax.set_title(f"{team_name}\n{pro_count} Pases Progresivos", color=line_color, fontsize=25, fontweight='bold')

def draw_progressive_carry_map(ax, df, team_name, col, hteamName, bg_color, line_color):
    dfpro = df[(df['teamName'] == team_name) & (df['type'] == 'Carry') & (df['prog_carry'] >= 9.11) & (df['endX'] >= 35)]
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)

    if team_name != hteamName:
        ax.invert_xaxis()
        ax.invert_yaxis()

    pro_count = len(dfpro)
    if pro_count == 0:
        ax.set_title(f"{team_name}\n0 Avance Progresivo con Balón", color=line_color, fontsize=25, fontweight='bold')
        return

    left_pro = len(dfpro[dfpro['y'] >= 45.33])
    mid_pro = len(dfpro[(dfpro['y'] >= 22.67) & (dfpro['y'] < 45.33)])
    right_pro = len(dfpro[dfpro['y'] < 22.67])

    ax.hlines(22.67, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.hlines(45.33, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)

    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    ax.text(8, 11.335, f'{right_pro}\n({round(right_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 34, f'{mid_pro}\n({round(mid_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 56.675, f'{left_pro}\n({round(left_pro/pro_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)

    for _, row in dfpro.iterrows():
        if pd.notna(row['x']) and pd.notna(row['endX']):
            arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']),
                                           arrowstyle='->', color=col, zorder=4, mutation_scale=20, alpha=0.9, linewidth=2, linestyle='--')
            ax.add_patch(arrow)

    ax.set_title(f"{team_name}\n{pro_count} Avance Progresivo con Balón", color=line_color, fontsize=25, fontweight='bold')

def plotting_match_stats(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 105)
    ax.set_ylim(-5, 65)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    def get_stats(team):
        t = df[df['teamName'] == team]
        passes = t[t['type'] == 'Pass']
        acc = passes[passes['outcomeType'] == 'Successful']
        longballs = passes[passes['qualifiers'].str.contains('Longball', na=False)] if 'qualifiers' in passes.columns else pd.DataFrame()
        longballs_acc = longballs[longballs['outcomeType'] == 'Successful']
        corners = passes[passes['qualifiers'].str.contains('CornerTaken', na=False)] if 'qualifiers' in passes.columns else pd.DataFrame()
        tackles = t[t['type'] == 'Tackle']
        tackles_won = tackles[tackles['outcomeType'] == 'Successful']
        interceptions = t[t['type'] == 'Interception']
        clearances = t[t['type'] == 'Clearance']
        aerials = t[t['type'] == 'Aerial']
        aerials_won = aerials[aerials['outcomeType'] == 'Successful']
        return len(passes), len(acc), len(longballs), len(longballs_acc), len(corners), len(tackles), len(tackles_won), len(interceptions), len(clearances), len(aerials), len(aerials_won)

    hp, hac, hlb, hlba, hcor, htk, htkw, hint, hcl, har, harw = get_stats(hteamName)
    ap, aac, alb, alba, acor, atk, atkw, aint, acl, aar, aarw = get_stats(ateamName)
    
    total = hp + ap
    hposs = round(hp / total * 100) if total > 0 else 50
    aposs = 100 - hposs

    path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
    stats = ['Posesion', 'Pases (Acc.)', 'Balones Largos (Acc.)', 'Corners', 'Entradas (ganadas)', 'Intercepciones', 'Despejes', 'Duelos aéreos (ganados)']
    h_vals = [f"{hposs}%", f"{hp}({hac})", f"{hlb}({hlba})", str(hcor), f"{htk}({htkw})", str(hint), str(hcl), f"{har}({harw})"]
    a_vals = [f"{aposs}%", f"{ap}({aac})", f"{alb}({alba})", str(acor), f"{atk}({atkw})", str(aint), str(acl), f"{aar}({aarw})"]

    ax.set_title("Estadisticas Partido", color=line_color, fontsize=25, fontweight='bold')
    for i, (stat, hv, av) in enumerate(zip(stats, h_vals, a_vals)):
        y = 55 - i * 7
        ax.text(52.5, y, stat, color=bg_color, fontsize=15, ha='center', va='center', fontweight='bold', path_effects=path_eff)
        ax.text(5, y, hv, color=line_color, fontsize=17, ha='left', va='center', fontweight='bold')
        ax.text(100, y, av, color=line_color, fontsize=17, ha='right', va='center', fontweight='bold')

# ============== FUNCIONES DASHBOARD 2 ==============
def Final_third_entry(ax, df, team_name, col, hteamName, bg_color, line_color):
    dfpass = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['x'] < 70) & (df['endX'] >= 70) & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Freekick', na=False))]
    dfcarry = df[(df['teamName'] == team_name) & (df['type'] == 'Carry') & (df['x'] < 70) & (df['endX'] >= 70)]
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)

    if team_name != hteamName:
        ax.invert_xaxis()
        ax.invert_yaxis()

    pass_count = len(dfpass) + len(dfcarry)
    if pass_count == 0:
        ax.set_title(f"{team_name}\n0 Entradas al último tercio", color=line_color, fontsize=25, fontweight='bold')
        return

    left_entry = len(dfpass[dfpass['y'] >= 45.33]) + len(dfcarry[dfcarry['y'] >= 45.33])
    mid_entry = len(dfpass[(dfpass['y'] >= 22.67) & (dfpass['y'] < 45.33)]) + len(dfcarry[(dfcarry['y'] >= 22.67) & (dfcarry['y'] < 45.33)])
    right_entry = len(dfpass[dfpass['y'] < 22.67]) + len(dfcarry[dfcarry['y'] < 22.67])

    ax.hlines(22.67, xmin=0, xmax=70, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.hlines(45.33, xmin=0, xmax=70, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.vlines(70, ymin=-2, ymax=70, colors=line_color, linestyle='dashed', alpha=0.55)

    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    ax.text(8, 11.335, f'{right_entry}\n({round(right_entry/pass_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 34, f'{mid_entry}\n({round(mid_entry/pass_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 56.675, f'{left_entry}\n({round(left_entry/pass_count*100)}%)', color=col, fontsize=24, va='center', ha='center', bbox=bbox_props)

    pitch.lines(dfpass.x, dfpass.y, dfpass.endX, dfpass.endY, lw=2, comet=True, color=col, ax=ax, alpha=0.5)
    for _, row in dfcarry.iterrows():
        if pd.notna(row['x']) and pd.notna(row['endX']):
            arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']),
                                           arrowstyle='->', color=col, zorder=4, mutation_scale=15, alpha=0.7, linewidth=1.5, linestyle='--')
            ax.add_patch(arrow)

    ax.set_title(f"{team_name}\n{pass_count} Entradas al último tercio", color=line_color, fontsize=25, fontweight='bold')
    
    if team_name == hteamName:
        ax.text(23, -5, f'Entrada por pase: {len(dfpass)}', fontsize=13, color=line_color, ha='center')
        ax.text(65, -5, f'Entrada por conducción: {len(dfcarry)}', fontsize=13, color=line_color, ha='center')
    else:
        ax.text(33, 73, f'Entrada por pase: {len(dfpass)}', fontsize=13, color=line_color, ha='center')
        ax.text(76, 73, f'Entrada por conducción: {len(dfcarry)}', fontsize=13, color=line_color, ha='center')

def box_entry(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    def get_entries(team):
        passes = df[(df['teamName'] == team) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') &
                   (df['x'] < 88.5) & (df['endX'] >= 88.5) & (df['endY'] >= 13.85) & (df['endY'] <= 54.15)]
        carries = df[(df['teamName'] == team) & (df['type'] == 'Carry') &
                    (df['x'] < 88.5) & (df['endX'] >= 88.5) & (df['endY'] >= 13.85) & (df['endY'] <= 54.15)]
        left_p = len(passes[passes['y'] >= 45.33]) + len(carries[carries['y'] >= 45.33])
        right_p = len(passes[passes['y'] < 22.67]) + len(carries[carries['y'] < 22.67])
        return len(passes), len(carries), left_p, right_p

    hp, hc, hl, hr = get_entries(hteamName)
    ap, ac, al, ar = get_entries(ateamName)

    ax.text(30, 50, f"{hl}", color=hcol, fontsize=30, ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, edgecolor=hcol, linewidth=2))
    ax.text(30, 18, f"{hr}", color=hcol, fontsize=30, ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, edgecolor=hcol, linewidth=2))
    ax.text(75, 50, f"{al}", color=acol, fontsize=30, ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, edgecolor=acol, linewidth=2))
    ax.text(75, 18, f"{ar}", color=acol, fontsize=30, ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, edgecolor=acol, linewidth=2))

    ax.set_title(f"{hteamName}\nEntradas al área: {hp+hc}", color=hcol, fontsize=20, fontweight='bold', loc='left')
    ax.set_title(f"{ateamName}\nEntradas al área: {ap+ac}", color=acol, fontsize=20, fontweight='bold', loc='right')

def zone14hs(ax, df, team_name, col, hteamName, bg_color, line_color):
    dfhp = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('CornerTaken|Freekick', na=False))]
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if team_name != hteamName:
        ax.invert_xaxis()
        ax.invert_yaxis()

    z14 = hs = 0
    for _, row in dfhp.iterrows():
        if row['endX'] >= 70 and row['endX'] <= 88.54 and row['endY'] >= 22.66 and row['endY'] <= 45.32:
            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color='#38dacc', comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
            ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor='#38dacc', zorder=4)
            z14 += 1
        if row['endX'] >= 70 and ((row['endY'] >= 11.33 and row['endY'] <= 22.66) or (row['endY'] >= 45.32 and row['endY'] <= 56.95)):
            pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col, comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
            ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=col, zorder=4)
            hs += 1

    ax.fill([70, 88.54, 88.54, 70], [22.66, 22.66, 45.32, 45.32], '#38dacc', alpha=0.2)
    ax.fill([70, 105, 105, 70], [11.33, 11.33, 22.66, 22.66], col, alpha=0.2)
    ax.fill([70, 105, 105, 70], [45.32, 45.32, 56.95, 56.95], col, alpha=0.2)

    ax.scatter(16.46, 13.85, color=col, s=10000, edgecolor=line_color, linewidth=2, marker='h')
    ax.scatter(16.46, 54.15, color='#38dacc', s=10000, edgecolor=line_color, linewidth=2, marker='h')
    ax.text(16.46, 13.85 - 3.5, "Carril int", fontsize=16, color=line_color, ha='center', va='center')
    ax.text(16.46, 54.15 - 3.5, "Zona14", fontsize=16, color=line_color, ha='center', va='center')
    ax.text(16.46, 13.85 + 2, str(hs), fontsize=35, color=line_color, ha='center', va='center', fontweight='bold')
    ax.text(16.46, 54.15 + 2, str(z14), fontsize=35, color=line_color, ha='center', va='center', fontweight='bold')

    ax.set_title(f"{team_name}\nPase Zona 14 y carril interior", color=line_color, fontsize=25, fontweight='bold')

def Crosses(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    def get_crosses(team):
        crosses = df[(df['teamName'] == team) & (df['type'] == 'Pass') & (df['qualifiers'].str.contains('Cross', na=False)) & (~df['qualifiers'].str.contains('CornerTaken', na=False))]
        succ = len(crosses[crosses['outcomeType'] == 'Successful'])
        left = len(crosses[crosses['y'] >= 45.33])
        right = len(crosses[crosses['y'] < 22.67])
        return len(crosses), succ, left, right

    ht, hs, hl, hr = get_crosses(hteamName)
    at, asuc, al, ar = get_crosses(ateamName)

    ax.text(20, 55, f"Centros desde\nla Derecha: {hr}", color=hcol, fontsize=14, ha='center')
    ax.text(20, 13, f"Centros desde\nla Izquierda: {hl}", color=hcol, fontsize=14, ha='center')
    ax.text(85, 55, f"Centros desde\nla Izquierda: {al}", color=acol, fontsize=14, ha='center')
    ax.text(85, 13, f"Centros desde\nLa Derecha: {ar}", color=acol, fontsize=14, ha='center')

    ax.set_title(f"{hteamName}\n<---Centros", color=hcol, fontsize=20, fontweight='bold', loc='left')
    ax.set_title(f"{ateamName}\nCentros--->", color=acol, fontsize=20, fontweight='bold', loc='right')
    ax.text(20, -3, f"Acertados: {hs}\nFallados: {ht - hs}", color=hcol, fontsize=12, ha='center')
    ax.text(85, -3, f"Acertados: {asuc}\nFallados: {at - asuc}", color=acol, fontsize=12, ha='center')

def Pass_end_zone(ax, df, team_name, col, hteamName, bg_color, line_color):
    pez = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')]
    pitch = Pitch(pitch_type='uefa', line_color=line_color, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if team_name != hteamName:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    cm = LinearSegmentedColormap.from_list("custom", [bg_color, col], N=20)
    
    if not pez.empty:
        bin_statistic = pitch.bin_statistic(pez['endX'], pez['endY'], statistic='count', bins=(6, 5))
        pitch.heatmap(bin_statistic, ax=ax, cmap=cm, edgecolors=line_color, linewidth=0.5, alpha=0.7)
        
        bin_statistic['statistic'] = (bin_statistic['statistic'] / bin_statistic['statistic'].sum() * 100).round(0).astype(int)
        labels = pitch.label_heatmap(bin_statistic, ax=ax, str_format='{:.0f}%', color=line_color, fontsize=12, va='center', ha='center')
    
    ax.set_title(f"{team_name}\nZona de finalización pase", color=line_color, fontsize=25, fontweight='bold')

def HighTO(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    def get_turnovers(team):
        t = df[df['teamName'] == team]
        to = t[(t['type'].isin(['Dispossessed'])) | ((t['type'] == 'Pass') & (t['outcomeType'] == 'Unsuccessful'))]
        return len(to[to['x'] >= 70])

    h_to = get_turnovers(hteamName)
    a_to = get_turnovers(ateamName)

    ax.fill([0, 35, 35, 0], [0, 0, 68, 68], hcol, alpha=0.15)
    ax.fill([70, 105, 105, 70], [0, 0, 68, 68], acol, alpha=0.15)

    ax.set_title(f"{hteamName}\nPérdidas zona alta: {h_to}", color=hcol, fontsize=18, fontweight='bold', loc='left')
    ax.set_title(f"{ateamName}\nPérdidas zona alta: {a_to}", color=acol, fontsize=18, fontweight='bold', loc='right')
    ax.text(52.5, -3, "<---Direccion Ataque", color=hcol, fontsize=12, ha='center')
    ax.text(52.5, 71, "Direccion Ataque--->", color=acol, fontsize=12, ha='center')

# ============== FUNCIONES DASHBOARD 3 ==============
def get_progressor_df(df, players_df):
    prog_passes = df[(df['type'] == 'Pass') & (df['prog_pass'] >= 9.11) & (df['outcomeType'] == 'Successful') & (df['x'] >= 35)]
    prog_carries = df[(df['type'] == 'Carry') & (df['prog_carry'] >= 9.11) & (df['endX'] >= 35)]
    
    pp_count = prog_passes.groupby('playerId').size().reset_index(name='Progressive Passes')
    pc_count = prog_carries.groupby('playerId').size().reset_index(name='Progressive Carries')
    
    merged = pp_count.merge(pc_count, on='playerId', how='outer').fillna(0)
    merged['total'] = merged['Progressive Passes'] + merged['Progressive Carries']
    merged = merged.merge(players_df[['playerId', 'name', 'teamId']], on='playerId', how='left')
    merged['shortName'] = merged['name'].apply(lambda x: x.split()[-1] if isinstance(x, str) else str(x))
    return merged.sort_values('total', ascending=True)

def get_defender_df(df, players_df):
    tackles = df[df['type'] == 'Tackle'].groupby('playerId').size().reset_index(name='Tackles')
    interceptions = df[df['type'] == 'Interception'].groupby('playerId').size().reset_index(name='Interceptions')
    clearances = df[df['type'] == 'Clearance'].groupby('playerId').size().reset_index(name='Clearance')
    
    merged = tackles.merge(interceptions, on='playerId', how='outer').merge(clearances, on='playerId', how='outer').fillna(0)
    merged['total'] = merged['Tackles'] + merged['Interceptions'] + merged['Clearance']
    merged = merged.merge(players_df[['playerId', 'name', 'teamId']], on='playerId', how='left')
    merged['shortName'] = merged['name'].apply(lambda x: x.split()[-1] if isinstance(x, str) else str(x))
    return merged.sort_values('total', ascending=True)

def passer_bar(ax, progressor_df, col1, col2, bg_color, line_color):
    top10 = progressor_df.nlargest(10, 'total').sort_values('total', ascending=True)
    names = top10['shortName'].tolist()
    pp = top10['Progressive Passes'].tolist()
    pc = top10['Progressive Carries'].tolist()
    
    ax.barh(names, pp, label='Pases Progresivos', color=col1)
    ax.barh(names, pc, left=pp, label='Conducciones progresivas', color=col2)
    
    for i, (p, c) in enumerate(zip(pp, pc)):
        if p > 0: ax.text(p/2, i, str(int(p)), ha='center', va='center', color=line_color, fontsize=11, fontweight='bold')
        if c > 0: ax.text(p + c/2, i, str(int(c)), ha='center', va='center', color=line_color, fontsize=11, fontweight='bold')
    
    ax.set_facecolor(bg_color)
    ax.tick_params(colors=line_color, labelsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor(bg_color)
    ax.set_title("Top 10 progresores de balón", color=line_color, fontsize=25, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, facecolor=bg_color, labelcolor=line_color)

def defender_bar(ax, defender_df, col1, col2, bg_color, line_color):
    top10 = defender_df.nlargest(10, 'total').sort_values('total', ascending=True)
    names = top10['shortName'].tolist()
    tk = top10['Tackles'].tolist()
    intc = top10['Interceptions'].tolist()
    cl = top10['Clearance'].tolist()
    
    left1 = [t + i for t, i in zip(tk, intc)]
    ax.barh(names, tk, label='Entradas', color=col1)
    ax.barh(names, intc, left=tk, label='Intercepciones', color='#a369ff')
    ax.barh(names, cl, left=left1, label='Despejes', color=col2)
    
    ax.set_facecolor(bg_color)
    ax.tick_params(colors=line_color, labelsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor(bg_color)
    ax.set_title("Top10 Defensores", color=line_color, fontsize=25, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, facecolor=bg_color, labelcolor=line_color)

def player_passmap(ax, df, player_name, col, bg_color, line_color, is_away=False):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    acc_pass = df[(df['name'] == player_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')]
    pro_pass = acc_pass[(acc_pass['prog_pass'] >= 9.11) & (acc_pass['x'] >= 35) & (~acc_pass['qualifiers'].str.contains('CornerTaken|Freekick', na=False))]
    pro_carry = df[(df['name'] == player_name) & (df['type'] == 'Carry') & (df['prog_carry'] >= 9.11) & (df['endX'] >= 35)]
    key_pass = acc_pass[acc_pass['qualifiers'].str.contains('KeyPass', na=False)]
    
    pitch.lines(acc_pass.x, acc_pass.y, acc_pass.endX, acc_pass.endY, color=line_color, lw=2, alpha=0.15, comet=True, zorder=2, ax=ax)
    pitch.lines(pro_pass.x, pro_pass.y, pro_pass.endX, pro_pass.endY, color=col, lw=3, alpha=1, comet=True, zorder=3, ax=ax)
    pitch.lines(key_pass.x, key_pass.y, key_pass.endX, key_pass.endY, color='#a369ff', lw=4, alpha=1, comet=True, zorder=4, ax=ax)
    
    ax.scatter(acc_pass.endX, acc_pass.endY, s=30, color=bg_color, edgecolor='gray', alpha=1, zorder=2)
    ax.scatter(pro_pass.endX, pro_pass.endY, s=40, color=bg_color, edgecolor=col, alpha=1, zorder=3)
    
    for _, row in pro_carry.iterrows():
        if pd.notna(row['x']) and pd.notna(row['endX']):
            arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']),
                                           arrowstyle='->', color=col, zorder=4, mutation_scale=20, alpha=0.9, linewidth=2, linestyle='--')
            ax.add_patch(arrow)
    
    short_name = player_name.split()[-1] if isinstance(player_name, str) else str(player_name)
    ax.set_title(f"{short_name} Mapa de pases", color=col, fontsize=25, fontweight='bold', y=1.03)
    
    if is_away:
        ax.text(105, 71, f'Pase Prog: {len(pro_pass)}          Conduccion Prog: {len(pro_carry)}', fontsize=13, color=col, ha='left')
    else:
        ax.text(0, -3, f'Pase Prog: {len(pro_pass)}          Conduccion Prog: {len(pro_carry)}', fontsize=13, color=col, ha='left')

# ============== GENERACIÓN DE DASHBOARDS ==============
def generate_dashboard_1(df, players_df, hteamName, ateamName, hcol, acol, bg_color, line_color, titulo, subtitulo, analyst_name, analyst_color, hgoals, agoals):
    fig, axs = plt.subplots(4, 3, figsize=(35, 35), facecolor=bg_color)
    
    passes_df = get_passes_df(df)
    home_passes_between, home_avg_locs = get_passes_between_df(hteamName, passes_df, df, players_df)
    away_passes_between, away_avg_locs = get_passes_between_df(ateamName, passes_df, df, players_df)
    
    defensive_actions_df = get_defensive_action_df(df)
    home_da_locs = get_da_count_df(hteamName, defensive_actions_df, players_df)
    away_da_locs = get_da_count_df(ateamName, defensive_actions_df, players_df)
    home_da_locs = home_da_locs[home_da_locs['position'] != 'GK']
    away_da_locs = away_da_locs[away_da_locs['position'] != 'GK']
    
    pass_network_visualization(axs[0, 0], home_passes_between, home_avg_locs, hcol, hteamName, hteamName, bg_color, line_color, passes_df)
    plot_shotmap(axs[0, 1], df, hteamName, ateamName, hcol, acol, bg_color, line_color)
    pass_network_visualization(axs[0, 2], away_passes_between, away_avg_locs, acol, ateamName, hteamName, bg_color, line_color, passes_df)
    
    defensive_block(axs[1, 0], home_da_locs, defensive_actions_df, hteamName, hcol, hteamName, bg_color, line_color)
    plot_goalPost(axs[1, 1], df, hteamName, ateamName, hcol, acol, bg_color, line_color)
    defensive_block(axs[1, 2], away_da_locs, defensive_actions_df, ateamName, acol, hteamName, bg_color, line_color)
    
    draw_progressive_pass_map(axs[2, 0], df, hteamName, hcol, hteamName, bg_color, line_color)
    axs[2, 1].set_facecolor(bg_color)
    axs[2, 1].axis('off')
    axs[2, 1].text(0.5, 0.5, "Momento del partido por xT\n(Requiere datos FotMob)", color=line_color, fontsize=20, ha='center', va='center', transform=axs[2, 1].transAxes)
    draw_progressive_pass_map(axs[2, 2], df, ateamName, acol, hteamName, bg_color, line_color)
    
    draw_progressive_carry_map(axs[3, 0], df, hteamName, hcol, hteamName, bg_color, line_color)
    plotting_match_stats(axs[3, 1], df, hteamName, ateamName, hcol, acol, bg_color, line_color)
    draw_progressive_carry_map(axs[3, 2], df, ateamName, acol, hteamName, bg_color, line_color)
    
    fig.text(0.5, 0.98, f"{hteamName} {hgoals} - {agoals} {ateamName}", color=line_color, fontsize=60, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.95, titulo, color=line_color, fontsize=28, ha='center', va='top')
    fig.text(0.5, 0.93, f"{subtitulo} | Analista {analyst_name}", color=analyst_color, fontsize=22, ha='center', va='top')
    fig.text(0.125, 0.08, 'Direccion Ataque ------->', color=hcol, fontsize=22, ha='left')
    fig.text(0.875, 0.08, '<------- Direccion Ataque', color=acol, fontsize=22, ha='right')
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    return fig

def generate_dashboard_2(df, players_df, hteamName, ateamName, hcol, acol, bg_color, line_color, titulo, subtitulo, analyst_name, analyst_color, hgoals, agoals):
    fig, axs = plt.subplots(4, 3, figsize=(35, 35), facecolor=bg_color)
    
    Final_third_entry(axs[0, 0], df, hteamName, hcol, hteamName, bg_color, line_color)
    box_entry(axs[0, 1], df, hteamName, ateamName, hcol, acol, bg_color, line_color)
    Final_third_entry(axs[0, 2], df, ateamName, acol, hteamName, bg_color, line_color)
    
    zone14hs(axs[1, 0], df, hteamName, hcol, hteamName, bg_color, line_color)
    Crosses(axs[1, 1], df, hteamName, ateamName, hcol, acol, bg_color, line_color)
    zone14hs(axs[1, 2], df, ateamName, acol, hteamName, bg_color, line_color)
    
    Pass_end_zone(axs[2, 0], df, hteamName, hcol, hteamName, bg_color, line_color)
    HighTO(axs[2, 1], df, hteamName, ateamName, hcol, acol, bg_color, line_color)
    Pass_end_zone(axs[2, 2], df, ateamName, acol, hteamName, bg_color, line_color)
    
    for i, (team, col) in enumerate([(hteamName, hcol), (ateamName, acol)]):
        ax = axs[3, 0] if i == 0 else axs[3, 2]
        ax.set_facecolor(bg_color)
        ax.axis('off')
        ax.text(0.5, 0.5, f"{team}\nZona Oportunidades creadas\n(Requiere cálculo adicional)", color=col, fontsize=16, ha='center', va='center', transform=ax.transAxes)
    axs[3, 1].set_facecolor(bg_color)
    axs[3, 1].axis('off')
    axs[3, 1].text(0.5, 0.5, "Zona de dominio equipo\n(Requiere cálculo territorial)", color=line_color, fontsize=16, ha='center', va='center', transform=axs[3, 1].transAxes)
    
    fig.text(0.5, 0.98, f"{hteamName} {hgoals} - {agoals} {ateamName}", color=line_color, fontsize=60, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.95, titulo, color=line_color, fontsize=28, ha='center', va='top')
    fig.text(0.5, 0.93, f"{subtitulo} | Analista {analyst_name}", color=analyst_color, fontsize=22, ha='center', va='top')
    fig.text(0.125, 0.08, 'Direccion Ataque ------->', color=hcol, fontsize=22, ha='left')
    fig.text(0.875, 0.08, '<------- Direccion Ataque', color=acol, fontsize=22, ha='right')
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    return fig

def generate_dashboard_3(df, players_df, hteamName, ateamName, hcol, acol, bg_color, line_color, titulo, subtitulo, analyst_name, analyst_color, hgoals, agoals, teams_dict):
    fig, axs = plt.subplots(4, 3, figsize=(35, 35), facecolor=bg_color)
    
    progressor_df = get_progressor_df(df, players_df)
    defender_df = get_defender_df(df, players_df)
    
    h_team_id = [k for k, v in teams_dict.items() if v == hteamName][0]
    a_team_id = [k for k, v in teams_dict.items() if v == ateamName][0]
    
    h_prog = progressor_df[progressor_df['teamId'] == h_team_id].nlargest(1, 'total')
    a_prog = progressor_df[progressor_df['teamId'] == a_team_id].nlargest(1, 'total')
    
    h_player = h_prog['name'].iloc[0] if not h_prog.empty else None
    a_player = a_prog['name'].iloc[0] if not a_prog.empty else None
    
    if h_player: player_passmap(axs[0, 0], df, h_player, hcol, bg_color, line_color, False)
    passer_bar(axs[0, 1], progressor_df, hcol, acol, bg_color, line_color)
    if a_player: player_passmap(axs[0, 2], df, a_player, acol, bg_color, line_color, True)
    
    for row in range(1, 4):
        for col_idx in range(3):
            axs[row, col_idx].set_facecolor(bg_color)
            axs[row, col_idx].axis('off')
            if col_idx == 1:
                titles = ["Participación en secuencias de tiros", "Top10 Defensores", "Top 10 jugadores más peligrosos"]
                if row == 2:
                    defender_bar(axs[row, col_idx], defender_df, hcol, acol, bg_color, line_color)
                    axs[row, col_idx].axis('on')
                else:
                    axs[row, col_idx].text(0.5, 0.5, f"{titles[row-1]}\n(Requiere datos adicionales)", color=line_color, fontsize=16, ha='center', va='center', transform=axs[row, col_idx].transAxes)
            else:
                team = hteamName if col_idx == 0 else ateamName
                col = hcol if col_idx == 0 else acol
                titles = [["Pases Recibidos", "Acciones Defensivas", "Mapa de Pases Portero"], ["Pases Recibidos", "Acciones Defensivas", "Mapa de Pases Portero"]]
                axs[row, col_idx].text(0.5, 0.5, f"{team}\n{titles[0][row-1]}\n(Requiere datos adicionales)", color=col, fontsize=14, ha='center', va='center', transform=axs[row, col_idx].transAxes)
    
    fig.text(0.5, 0.98, f"{hteamName} {hgoals} - {agoals} {ateamName}", color=line_color, fontsize=60, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.95, titulo, color=line_color, fontsize=28, ha='center', va='top')
    fig.text(0.5, 0.93, f"{subtitulo} | Analista {analyst_name}", color=analyst_color, fontsize=22, ha='center', va='top')
    fig.text(0.125, 0.08, 'Direccion Ataque ------->', color=hcol, fontsize=22, ha='left')
    fig.text(0.875, 0.08, '<------- Direccion Ataque', color=acol, fontsize=22, ha='right')
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    return fig

# ============== APLICACIÓN PRINCIPAL ==============
def main():
    st.title("⚽ Zona del Analista - Dashboard Generator")
    
    with st.sidebar:
        st.header("📁 Cargar Datos")
        uploaded_file = st.file_uploader("HTML de WhoScored", type=['html', 'htm'])
        
        st.divider()
        st.header("📝 Información")
        titulo = st.text_input("Título (Liga/Jornada)", "Champions 25/26 Jornada 5| Post Partido Informe-1")
        subtitulo = st.text_input("Subtítulo (Fecha)", "Miercoles 26/11/25")
        analyst_name = st.text_input("Analista", "John Triguero")
        
        st.divider()
        st.header("🎨 Colores")
        bg_color = st.color_picker("Fondo", '#363d4d')
        line_color = st.color_picker("Líneas", '#ffffff')
        home_color = st.color_picker("Equipo Local", '#ff4b44')
        away_color = st.color_picker("Equipo Visitante", '#00FFD5')
        analyst_color = st.color_picker("Nombre Analista", '#ffffff')
    
    if uploaded_file:
        try:
            html_content = uploaded_file.read().decode('utf-8')
            
            with st.spinner('Procesando datos...'):
                json_data_txt = extract_json_from_html(html_content)
                data = json.loads(json_data_txt)
                events_dict, players_df, teams_dict = extract_data_from_dict(data)
                
                df = pd.DataFrame(events_dict)
                df = process_dataframe(df, teams_dict)
                df = insert_ball_carries(df)
                
                team_ids = list(teams_dict.keys())
                hteamName = teams_dict[team_ids[0]]
                ateamName = teams_dict[team_ids[1]]
                
                homedf = df[df['teamName'] == hteamName]
                awaydf = df[df['teamName'] == ateamName]
                hgoals = len(homedf[(homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal', na=False))])
                hgoals += len(awaydf[(awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal', na=False))])
                agoals = len(awaydf[(awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal', na=False))])
                agoals += len(homedf[(homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal', na=False))])
            
            st.success(f"✅ {hteamName} {hgoals} - {agoals} {ateamName}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Dashboard 1\n(Informe General)", use_container_width=True):
                    with st.spinner('Generando...'):
                        fig = generate_dashboard_1(df, players_df, hteamName, ateamName, home_color, away_color, bg_color, line_color, titulo, subtitulo, analyst_name, analyst_color, hgoals, agoals)
                        buf = BytesIO()
                        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=bg_color)
                        buf.seek(0)
                        st.download_button("📥 Descargar", buf, f"{hteamName}_vs_{ateamName}_Dashboard1.png", "image/png")
                        st.pyplot(fig)
                        plt.close()
            
            with col2:
                if st.button("📊 Dashboard 2\n(Zonas)", use_container_width=True):
                    with st.spinner('Generando...'):
                        fig = generate_dashboard_2(df, players_df, hteamName, ateamName, home_color, away_color, bg_color, line_color, titulo.replace("Informe-1", "Informe-2"), subtitulo, analyst_name, analyst_color, hgoals, agoals)
                        buf = BytesIO()
                        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=bg_color)
                        buf.seek(0)
                        st.download_button("📥 Descargar", buf, f"{hteamName}_vs_{ateamName}_Dashboard2.png", "image/png")
                        st.pyplot(fig)
                        plt.close()
            
            with col3:
                if st.button("📊 Dashboard 3\n(Top Jugadores)", use_container_width=True):
                    with st.spinner('Generando...'):
                        fig = generate_dashboard_3(df, players_df, hteamName, ateamName, home_color, away_color, bg_color, line_color, titulo.replace("Post Partido Informe-1", "Top Jugadores del Partido"), subtitulo, analyst_name, analyst_color, hgoals, agoals, teams_dict)
                        buf = BytesIO()
                        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=bg_color)
                        buf.seek(0)
                        st.download_button("📥 Descargar", buf, f"{hteamName}_vs_{ateamName}_Dashboard3.png", "image/png")
                        st.pyplot(fig)
                        plt.close()
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("👆 Sube un archivo HTML de WhoScored para comenzar")

if __name__ == "__main__":
    main()
