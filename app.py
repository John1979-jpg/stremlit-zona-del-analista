import streamlit as st
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import matplotlib.patches as patches
from io import BytesIO
import matplotlib as mpl
from mplsoccer import Pitch, VerticalPitch
import matplotlib.patheffects as path_effects
from unidecode import unidecode
from datetime import date
import requests
import warnings
warnings.filterwarnings('ignore')

# ============== CONFIGURACIÓN DE LA PÁGINA ==============
st.set_page_config(
    page_title="Zona del Analista - WhoScored",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== ESTILOS ==============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #e94560;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .match-info {
        background-color: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 15px;
        padding-right: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ============== COLORES POR DEFECTO ==============
DEFAULT_COLORS = {
    'bg_color': '#363d4d',
    'line_color': '#38dacc',
    'home_color': '#ffffff',
    'away_color': '#cf2740'
}

# ============== FUNCIONES DE EXTRACCIÓN DE DATOS ==============

def extract_json_from_html(html_content):
    """Extrae los datos JSON del HTML de WhoScored"""
    regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
    data_txt = re.findall(regex_pattern, html_content)[0]
    
    data_txt = data_txt.replace('matchId', '"matchId"')
    data_txt = data_txt.replace('matchCentreData', '"matchCentreData"')
    data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
    data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"')
    data_txt = data_txt.replace('};', '}')
    
    return data_txt

def extract_data_from_dict(data):
    """Extrae eventos, jugadores y equipos del diccionario de datos"""
    events_dict = data["matchCentreData"]["events"]
    teams_dict = {
        data["matchCentreData"]['home']['teamId']: data["matchCentreData"]['home']['name'],
        data["matchCentreData"]['away']['teamId']: data["matchCentreData"]['away']['name']
    }
    
    players_home_df = pd.DataFrame(data["matchCentreData"]['home']['players'])
    players_home_df["teamId"] = data["matchCentreData"]['home']['teamId']
    players_away_df = pd.DataFrame(data["matchCentreData"]['away']['players'])
    players_away_df["teamId"] = data["matchCentreData"]['away']['teamId']
    players_df = pd.concat([players_home_df, players_away_df])
    
    return events_dict, players_df, teams_dict

def process_events_dataframe(df, teams_dict):
    """Procesa el dataframe de eventos"""
    # Extraer displayName de los diccionarios
    df['type'] = df['type'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    df['outcomeType'] = df['outcomeType'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    df['period'] = df['period'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    
    df['period'] = df['period'].replace({
        'FirstHalf': 1, 'SecondHalf': 2, 
        'FirstPeriodOfExtraTime': 3, 'SecondPeriodOfExtraTime': 4,
        'PenaltyShootout': 5, 'PostGame': 14, 'PreMatch': 16
    })
    
    # Calcular minutos acumulativos
    df['cumulative_mins'] = df['minute'] + df['second'] / 60
    for period in df['period'].unique():
        if period > 1:
            prev_max = df[df['period'] == period - 1]['cumulative_mins'].max()
            curr_min = df[df['period'] == period]['cumulative_mins'].min()
            df.loc[df['period'] == period, 'cumulative_mins'] += prev_max - curr_min
    
    df['teamName'] = df['teamId'].map(teams_dict)
    team_names = list(teams_dict.values())
    opposition_dict = {team_names[i]: team_names[1-i] for i in range(len(team_names))}
    df['oppositionTeamName'] = df['teamName'].map(opposition_dict)
    
    # Escalar coordenadas a campo UEFA (105x68)
    df['x'] = df['x'] * 1.05
    df['y'] = df['y'] * 0.68
    df['endX'] = df['endX'] * 1.05 if 'endX' in df.columns else np.nan
    df['endY'] = df['endY'] * 0.68 if 'endY' in df.columns else np.nan
    df['goalMouthY'] = df['goalMouthY'] * 0.68 if 'goalMouthY' in df.columns else np.nan
    
    # Convertir qualifiers a string
    if 'qualifiers' in df.columns:
        df['qualifiers'] = df['qualifiers'].astype(str)
    
    # Calcular pases y conducciones progresivas
    df['prog_pass'] = np.where((df['type'] == 'Pass'),
                               np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - 
                               np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['prog_carry'] = np.where((df['type'] == 'Carry'),
                                np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - 
                                np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    
    return df

def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=60, min_carry_duration=1, max_carry_duration=10):
    """Inserta eventos de carry (conducción) entre eventos"""
    events_out = pd.DataFrame()
    match_events = events_df.reset_index(drop=True)
    match_carries = pd.DataFrame()
    
    for idx, match_event in match_events.iterrows():
        if idx == len(match_events) - 1:
            break
            
        next_evt = match_events.loc[idx + 1]
        
        if 'endX' not in match_event or pd.isna(match_event.get('endX')):
            continue
            
        same_team = match_event['teamId'] == next_evt['teamId']
        not_ball_touch = next_evt['type'] not in ['BallTouch', 'TakeOn', 'Foul']
        
        dx = next_evt['x'] - match_event['endX']
        dy = next_evt['y'] - match_event['endY']
        far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
        not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
        
        dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
        min_time = dt >= min_carry_duration
        same_phase = dt < max_carry_duration
        same_period = match_event['period'] == next_evt['period']
        
        valid_carry = same_team and not_ball_touch and far_enough and not_too_far and min_time and same_phase and same_period
        
        if valid_carry:
            carry = pd.DataFrame([{
                'eventId': match_event.get('eventId', idx) + 0.5,
                'minute': (match_event['minute'] + next_evt['minute']) / 2,
                'second': (match_event['second'] + next_evt['second']) / 2,
                'teamId': next_evt['teamId'],
                'teamName': next_evt['teamName'],
                'x': match_event['endX'],
                'y': match_event['endY'],
                'endX': next_evt['x'],
                'endY': next_evt['y'],
                'type': 'Carry',
                'outcomeType': 'Successful',
                'period': next_evt['period'],
                'playerId': next_evt['playerId'],
                'cumulative_mins': (match_event['cumulative_mins'] + next_evt['cumulative_mins']) / 2
            }])
            match_carries = pd.concat([match_carries, carry], ignore_index=True)
    
    events_out = pd.concat([match_carries, match_events], ignore_index=True)
    events_out = events_out.sort_values(['period', 'cumulative_mins']).reset_index(drop=True)
    
    return events_out

def get_short_name(full_name):
    """Obtiene nombre corto del jugador"""
    if pd.isna(full_name):
        return full_name
    parts = str(full_name).split()
    if len(parts) == 1:
        return full_name
    elif len(parts) == 2:
        return parts[0][0] + ". " + parts[1]
    else:
        return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])

# ============== FUNCIONES AUXILIARES ==============

def add_footer(fig, analyst_name, match_date, fotmob_id):
    """Añade el pie de página con información del analista"""
    footer_text = f"Análisis: {analyst_name} | Fecha: {match_date} | FotMob ID: {fotmob_id} | Datos: WhoScored"
    fig.text(0.5, 0.02, footer_text, ha='center', va='center', fontsize=10, 
             color='gray', style='italic', transform=fig.transFigure)

def get_passes_df(df):
    """Obtiene dataframe de pases con receptor"""
    df1 = df[~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card', na=False)]
    df1 = df1.copy()
    df1.loc[:, "receiver"] = df1["playerId"].shift(-1)
    passes_ids = df1.index[df1['type'] == 'Pass']
    df_passes = df1.loc[passes_ids, ["x", "y", "endX", "endY", "teamName", "playerId", "receiver", "type", "outcomeType"]]
    return df_passes

# ============== FUNCIONES DE VISUALIZACIÓN ==============

def get_passes_between_df(team_name, passes_df, players_df, df):
    """Calcula los pases entre jugadores para el pass network"""
    passes_team = passes_df[passes_df['teamName'] == team_name].copy()
    dfteam = df[(df['teamName'] == team_name) & (~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card', na=False))]
    
    if passes_team.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    passes_team = passes_team.merge(players_df[["playerId", "isFirstEleven"]], on='playerId', how='left')
    
    # Calcular posiciones medias
    average_locs_and_count_df = dfteam.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']})
    average_locs_and_count_df.columns = ['pass_avg_x', 'pass_avg_y', 'count']
    average_locs_and_count_df = average_locs_and_count_df.merge(
        players_df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], 
        on='playerId', how='left'
    )
    average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')
    
    if 'name' in average_locs_and_count_df.columns:
        average_locs_and_count_df['name'] = average_locs_and_count_df['name'].apply(
            lambda x: unidecode(str(x)) if pd.notna(x) else x
        )
    
    # Calcular pases entre jugadores
    passes_player_ids_df = passes_team[['playerId', 'receiver', 'teamName']].dropna()
    passes_player_ids_df['pos_max'] = passes_player_ids_df[['playerId', 'receiver']].max(axis=1)
    passes_player_ids_df['pos_min'] = passes_player_ids_df[['playerId', 'receiver']].min(axis=1)
    
    passes_between_df = passes_player_ids_df.groupby(['pos_min', 'pos_max']).size().reset_index(name='pass_count')
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_min', right_index=True)
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_max', right_index=True, suffixes=['', '_end'])
    
    return passes_between_df, average_locs_and_count_df

def plot_pass_network(ax, passes_between_df, average_locs_and_count_df, team_color, team_name, bg_color, line_color, is_away=False):
    """Dibuja el pass network de un equipo"""
    if passes_between_df.empty or average_locs_and_count_df.empty:
        pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
        pitch.draw(ax=ax)
        ax.text(52.5, 34, "No hay datos suficientes", ha='center', va='center', fontsize=14, color=line_color)
        ax.set_title(f"{team_name}\nRed de Pases", color=line_color, fontsize=18, fontweight='bold')
        return {}
    
    MAX_LINE_WIDTH = 15
    passes_between_df['width'] = (passes_between_df['pass_count'] / passes_between_df['pass_count'].max() * MAX_LINE_WIDTH)
    
    MIN_TRANSPARENCY = 0.05
    MAX_TRANSPARENCY = 0.85
    color = np.array(to_rgba(team_color))
    color = np.tile(color, (len(passes_between_df), 1))
    c_transparency = passes_between_df['pass_count'] / passes_between_df['pass_count'].max()
    c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency
    
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    # Dibujar líneas entre jugadores
    pitch.lines(
        passes_between_df['pass_avg_x'], passes_between_df['pass_avg_y'],
        passes_between_df['pass_avg_x_end'], passes_between_df['pass_avg_y_end'],
        lw=passes_between_df['width'], color=color, zorder=1, ax=ax
    )
    
    # Dibujar nodos de jugadores
    for idx, row in average_locs_and_count_df.iterrows():
        is_starter = row.get('isFirstEleven', True)
        marker = 'o' if is_starter else 's'
        alpha = 1 if is_starter else 0.75
        pitch.scatter(row['pass_avg_x'], row['pass_avg_y'], s=1000, marker=marker, 
                     color=bg_color, edgecolor=line_color, linewidth=2, alpha=alpha, ax=ax)
        
        shirt_no = row.get('shirtNo', '')
        ax.annotate(str(int(shirt_no)) if pd.notna(shirt_no) else '', 
                   xy=(row['pass_avg_x'], row['pass_avg_y']), 
                   ha='center', va='center', fontsize=14, color=team_color, fontweight='bold')
    
    # Línea de altura media
    avgph = average_locs_and_count_df['pass_avg_x'].median()
    ax.axvline(x=avgph, color='gray', linestyle='--', alpha=0.75, linewidth=2)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.text(avgph - 1, 73, f"{avgph:.1f}m", fontsize=12, color=line_color, ha='left')
    else:
        ax.text(avgph - 1, -5, f"{avgph:.1f}m", fontsize=12, color=line_color, ha='right')
    
    ax.set_title(f"{team_name}\nRed de Pases", color=line_color, fontsize=18, fontweight='bold')
    ax.text(2, 66 if not is_away else 2, "○ = titular\n□ = suplente", color=team_color, fontsize=10, ha='left', va='top')
    
    return {
        'Team': team_name,
        'Avg_Pass_Height': round(avgph, 2),
        'Total_Passes': int(average_locs_and_count_df['count'].sum())
    }

def plot_defensive_block(ax, df, team_name, team_color, bg_color, line_color, players_df, is_away=False):
    """Dibuja el bloque defensivo con heatmap"""
    # Filtrar acciones defensivas
    defensive_types = ['Aerial', 'BallRecovery', 'BlockedPass', 'Challenge', 
                       'Clearance', 'Error', 'Foul', 'Interception', 'Tackle']
    
    df_def = df[(df['teamName'] == team_name) & (df['type'].isin(defensive_types))].copy()
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    if df_def.empty:
        ax.set_title(f"{team_name}\nBloque Defensivo", color=line_color, fontsize=18, fontweight='bold')
        return {}
    
    # Calcular posiciones medias por jugador
    average_locs = df_def.groupby('playerId').agg({'x': 'median', 'y': 'median'}).reset_index()
    average_locs.columns = ['playerId', 'x', 'y']
    da_count = df_def.groupby('playerId').size().reset_index(name='count')
    average_locs = average_locs.merge(da_count, on='playerId')
    average_locs = average_locs.merge(players_df[['playerId', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left')
    
    # Excluir portero
    average_locs = average_locs[average_locs['position'] != 'GK']
    
    # Heatmap de acciones defensivas
    flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo", [bg_color, team_color], N=500)
    pitch.kdeplot(df_def['x'], df_def['y'], ax=ax, fill=True, levels=5000, thresh=0.02, cut=4, cmap=flamingo_cmap)
    
    # Dibujar nodos de jugadores
    MAX_MARKER_SIZE = 3500
    if not average_locs.empty:
        average_locs['marker_size'] = (average_locs['count'] / average_locs['count'].max() * MAX_MARKER_SIZE)
        
        for _, row in average_locs.iterrows():
            is_starter = row.get('isFirstEleven', True)
            marker = 'o' if is_starter else 's'
            pitch.scatter(row['x'], row['y'], s=row['marker_size']+100, marker=marker, 
                         color=bg_color, edgecolor=line_color, linewidth=1, alpha=1, zorder=3, ax=ax)
            
            shirt_no = row.get('shirtNo', '')
            ax.annotate(str(int(shirt_no)) if pd.notna(shirt_no) else '', 
                       xy=(row['x'], row['y']), ha='center', va='center', fontsize=12, color=line_color)
    
    # Dispersión de acciones
    pitch.scatter(df_def['x'], df_def['y'], s=10, marker='x', color='yellow', alpha=0.2, ax=ax)
    
    # Altura media defensiva
    dah = average_locs['x'].mean() if not average_locs.empty else 0
    ax.axvline(x=dah, color='gray', linestyle='--', alpha=0.75, linewidth=2)
    
    if is_away:
        ax.text(dah - 1, 73, f"DAH: {dah:.1f}m", fontsize=12, color=line_color, ha='left')
    else:
        ax.text(dah - 1, -5, f"DAH: {dah:.1f}m", fontsize=12, color=line_color, ha='right')
    
    ax.set_title(f"{team_name}\nBloque Defensivo", color=line_color, fontsize=18, fontweight='bold')
    
    return {
        'Team': team_name,
        'Defensive_Actions': len(df_def),
        'Avg_Defensive_Height': round(dah, 2)
    }

def plot_progressive_passes(ax, df, team_name, team_color, bg_color, line_color, is_away=False):
    """Dibuja el mapa de pases progresivos"""
    dfpro = df[(df['teamName'] == team_name) & 
               (df['prog_pass'] >= 9.11) & 
               (df['x'] >= 35) & 
               (df['outcomeType'] == 'Successful') &
               (df['type'] == 'Pass')].copy()
    
    # Excluir corners y tiros libres si hay qualifiers
    if 'qualifiers' in dfpro.columns:
        dfpro = dfpro[~dfpro['qualifiers'].str.contains('CornerTaken|Freekick', na=False)]
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    pro_count = len(dfpro)
    
    if pro_count == 0:
        ax.set_title(f"{team_name}\n0 Pases Progresivos", color=line_color, fontsize=18, fontweight='bold')
        return {'Team': team_name, 'Progressive_Passes': 0}
    
    # Calcular por zonas
    left_pro = len(dfpro[dfpro['y'] >= 45.33])
    mid_pro = len(dfpro[(dfpro['y'] >= 22.67) & (dfpro['y'] < 45.33)])
    right_pro = len(dfpro[dfpro['y'] < 22.67])
    
    left_pct = round((left_pro/pro_count)*100)
    mid_pct = round((mid_pro/pro_count)*100)
    right_pct = round((right_pro/pro_count)*100)
    
    # Líneas divisorias
    ax.hlines(22.67, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.hlines(45.33, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
    
    # Textos por zona
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    ax.text(8, 11.335, f'{right_pro}\n({right_pct}%)', color=team_color, fontsize=18, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 34, f'{mid_pro}\n({mid_pct}%)', color=team_color, fontsize=18, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 56.675, f'{left_pro}\n({left_pct}%)', color=team_color, fontsize=18, va='center', ha='center', bbox=bbox_props)
    
    # Dibujar pases
    pitch.lines(dfpro['x'], dfpro['y'], dfpro['endX'], dfpro['endY'], lw=3.5, comet=True, color=team_color, ax=ax, alpha=0.5)
    pitch.scatter(dfpro['endX'], dfpro['endY'], s=35, edgecolor=team_color, linewidth=1, color=bg_color, zorder=2, ax=ax)
    
    ax.set_title(f"{team_name}\n{pro_count} Pases Progresivos", color=line_color, fontsize=18, fontweight='bold')
    
    return {
        'Team': team_name,
        'Progressive_Passes': pro_count,
        'Left': left_pro,
        'Center': mid_pro,
        'Right': right_pro
    }

def plot_progressive_carries(ax, df, team_name, team_color, bg_color, line_color, is_away=False):
    """Dibuja el mapa de conducciones progresivas"""
    dfpro = df[(df['teamName'] == team_name) & 
               (df['prog_carry'] >= 9.11) & 
               (df['endX'] >= 35) &
               (df['type'] == 'Carry')].copy()
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    pro_count = len(dfpro)
    
    if pro_count == 0:
        ax.set_title(f"{team_name}\n0 Conducciones Progresivas", color=line_color, fontsize=18, fontweight='bold')
        return {'Team': team_name, 'Progressive_Carries': 0}
    
    # Calcular por zonas
    left_pro = len(dfpro[dfpro['y'] >= 45.33])
    mid_pro = len(dfpro[(dfpro['y'] >= 22.67) & (dfpro['y'] < 45.33)])
    right_pro = len(dfpro[dfpro['y'] < 22.67])
    
    left_pct = round((left_pro/pro_count)*100)
    mid_pct = round((mid_pro/pro_count)*100)
    right_pct = round((right_pro/pro_count)*100)
    
    # Líneas divisorias
    ax.hlines(22.67, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.hlines(45.33, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
    
    # Textos por zona
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    ax.text(8, 11.335, f'{right_pro}\n({right_pct}%)', color=team_color, fontsize=18, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 34, f'{mid_pro}\n({mid_pct}%)', color=team_color, fontsize=18, va='center', ha='center', bbox=bbox_props)
    ax.text(8, 56.675, f'{left_pro}\n({left_pct}%)', color=team_color, fontsize=18, va='center', ha='center', bbox=bbox_props)
    
    # Dibujar conducciones con flechas
    for _, row in dfpro.iterrows():
        arrow = patches.FancyArrowPatch(
            (row['x'], row['y']), (row['endX'], row['endY']),
            arrowstyle='->', color=team_color, zorder=4, mutation_scale=20,
            alpha=0.7, linewidth=2, linestyle='--'
        )
        ax.add_patch(arrow)
    
    ax.set_title(f"{team_name}\n{pro_count} Conducciones Progresivas", color=line_color, fontsize=18, fontweight='bold')
    
    return {
        'Team': team_name,
        'Progressive_Carries': pro_count,
        'Left': left_pro,
        'Center': mid_pro,
        'Right': right_pro
    }

def plot_shotmap_combined(ax, df, home_team, away_team, home_color, away_color, bg_color, line_color):
    """Dibuja el mapa de tiros combinado (ambos equipos en un campo)"""
    shot_types = ['Goal', 'SavedShot', 'AttemptSaved', 'MissedShots', 'MissedShot', 'Miss', 
                  'ShotOnPost', 'Post', 'BlockedShot']
    shots_df = df[df['type'].isin(shot_types)].copy()
    
    shots_df['type'] = shots_df['type'].replace({
        'AttemptSaved': 'SavedShot',
        'MissedShot': 'MissedShots',
        'Miss': 'MissedShots',
        'Post': 'ShotOnPost',
        'BlockedShot': 'MissedShots'
    })
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_ylim(-0.5, 68.5)
    ax.set_xlim(-0.5, 105.5)
    
    if shots_df.empty:
        ax.set_title("Mapa de Tiros\n(Sin datos)", color=line_color, fontsize=20, fontweight='bold')
        return {home_team: {'Goals': 0, 'Shots': 0}, away_team: {'Goals': 0, 'Shots': 0}}
    
    home_shots = shots_df[shots_df['teamName'] == home_team]
    away_shots = shots_df[shots_df['teamName'] == away_team]
    
    # Tiros equipo local (invertidos para mostrar en la izquierda)
    hGoalData = home_shots[home_shots['type'] == 'Goal']
    hPostData = home_shots[home_shots['type'] == 'ShotOnPost']
    hSaveData = home_shots[home_shots['type'] == 'SavedShot']
    hMissData = home_shots[home_shots['type'] == 'MissedShots']
    
    pitch.scatter((105 - hGoalData['x']), (68 - hGoalData['y']), s=350, edgecolors='white', c='None', marker='football', zorder=3, ax=ax)
    pitch.scatter((105 - hPostData['x']), (68 - hPostData['y']), s=200, edgecolors=home_color, c=home_color, marker='o', ax=ax)
    pitch.scatter((105 - hSaveData['x']), (68 - hSaveData['y']), s=200, edgecolors=home_color, c='None', hatch='///////', marker='o', ax=ax)
    pitch.scatter((105 - hMissData['x']), (68 - hMissData['y']), s=200, edgecolors=home_color, c='None', marker='o', ax=ax)
    
    # Tiros equipo visitante
    aGoalData = away_shots[away_shots['type'] == 'Goal']
    aPostData = away_shots[away_shots['type'] == 'ShotOnPost']
    aSaveData = away_shots[away_shots['type'] == 'SavedShot']
    aMissData = away_shots[away_shots['type'] == 'MissedShots']
    
    pitch.scatter(aGoalData['x'], aGoalData['y'], s=350, edgecolors='white', c='None', marker='football', zorder=3, ax=ax)
    pitch.scatter(aPostData['x'], aPostData['y'], s=200, edgecolors=away_color, c=away_color, marker='o', ax=ax)
    pitch.scatter(aSaveData['x'], aSaveData['y'], s=200, edgecolors=away_color, c='None', hatch='///////', marker='o', ax=ax)
    pitch.scatter(aMissData['x'], aMissData['y'], s=200, edgecolors=away_color, c='None', marker='o', ax=ax)
    
    # Títulos
    ax.text(0, 70, f"{home_team}\n<---Tiros ({len(home_shots)})", color=home_color, size=18, ha='left', fontweight='bold')
    ax.text(105, 70, f"{away_team}\nTiros---> ({len(away_shots)})", color=away_color, size=18, ha='right', fontweight='bold')
    
    return {
        home_team: {'Goals': len(hGoalData), 'Shots': len(home_shots)},
        away_team: {'Goals': len(aGoalData), 'Shots': len(away_shots)}
    }

def plot_shotmap_individual(ax, df, team_name, team_color, bg_color, line_color):
    """Dibuja el mapa de tiros individual de un equipo (media cancha vertical)"""
    shot_types = ['Goal', 'SavedShot', 'AttemptSaved', 'MissedShots', 'MissedShot', 'Miss',
                  'ShotOnPost', 'Post', 'BlockedShot']
    shots_df = df[(df['type'].isin(shot_types)) & (df['teamName'] == team_name)].copy()
    
    shots_df['type'] = shots_df['type'].replace({
        'AttemptSaved': 'SavedShot',
        'MissedShot': 'MissedShots',
        'Miss': 'MissedShots',
        'Post': 'ShotOnPost',
        'BlockedShot': 'MissedShots'
    })
    
    pitch = VerticalPitch(pitch_type='uefa', half=True, pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    
    if shots_df.empty:
        ax.set_title(f"{team_name}\n0 Tiros", color=line_color, fontsize=18, fontweight='bold')
        return {'Team': team_name, 'Goals': 0, 'Shots': 0, 'Shots_on_Target': 0, 'Avg_Distance': 0}
    
    goals = shots_df[shots_df['type'] == 'Goal']
    saved = shots_df[shots_df['type'] == 'SavedShot']
    missed = shots_df[shots_df['type'] == 'MissedShots']
    post = shots_df[shots_df['type'] == 'ShotOnPost']
    
    if not goals.empty:
        pitch.scatter(goals['x'], goals['y'], s=400, marker='football', 
                     color=team_color, edgecolors='white', ax=ax, zorder=3)
    if not saved.empty:
        pitch.scatter(saved['x'], saved['y'], s=250, marker='o',
                     color='None', edgecolors=team_color, hatch='///////', linewidths=2, ax=ax, zorder=2)
    if not missed.empty:
        pitch.scatter(missed['x'], missed['y'], s=250, marker='o',
                     color='None', edgecolors=team_color, linewidths=2, ax=ax, zorder=2)
    if not post.empty:
        pitch.scatter(post['x'], post['y'], s=250, marker='o',
                     color=team_color, edgecolors='orange', linewidths=2, ax=ax, zorder=2)
    
    # Calcular distancia media
    given_point = (105, 34)
    shot_distances = np.sqrt((shots_df['x'] - given_point[0])**2 + (shots_df['y'] - given_point[1])**2)
    avg_distance = round(shot_distances.mean(), 1) if not shot_distances.empty else 0
    
    total_shots = len(shots_df)
    total_goals = len(goals)
    shots_on_target = len(saved) + total_goals
    
    ax.set_title(f"{team_name}\n{total_shots} Tiros | {total_goals} Goles", color=line_color, fontsize=18, fontweight='bold')
    ax.text(52.5, 50, f"A puerta: {shots_on_target} | Dist. media: {avg_distance}m", 
            color=team_color, fontsize=11, ha='center', fontweight='bold')
    
    return {
        'Team': team_name,
        'Goals': total_goals,
        'Shots': total_shots,
        'Shots_on_Target': shots_on_target,
        'Avg_Distance': avg_distance
    }

def plot_final_third_entries(ax, df, team_name, team_color, bg_color, line_color, is_away=False):
    """Dibuja las entradas al último tercio"""
    # Filtrar pases y conducciones que terminan en el último tercio
    passes_final = df[(df['teamName'] == team_name) & 
                      (df['type'] == 'Pass') & 
                      (df['outcomeType'] == 'Successful') &
                      (df['x'] < 70) & (df['endX'] >= 70)].copy()
    
    carries_final = df[(df['teamName'] == team_name) & 
                       (df['type'] == 'Carry') & 
                       (df['x'] < 70) & (df['endX'] >= 70)].copy()
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    # Línea del último tercio
    ax.axvline(x=70, color='yellow', linestyle='--', alpha=0.5, linewidth=2)
    
    # Dibujar pases
    if not passes_final.empty:
        pitch.lines(passes_final['x'], passes_final['y'], passes_final['endX'], passes_final['endY'], 
                   lw=2, comet=True, color=team_color, ax=ax, alpha=0.6)
    
    # Dibujar conducciones
    for _, row in carries_final.iterrows():
        arrow = patches.FancyArrowPatch(
            (row['x'], row['y']), (row['endX'], row['endY']),
            arrowstyle='->', color='yellow', zorder=4, mutation_scale=15,
            alpha=0.6, linewidth=1.5, linestyle='--'
        )
        ax.add_patch(arrow)
    
    total_entries = len(passes_final) + len(carries_final)
    pass_entries = len(passes_final)
    carry_entries = len(carries_final)
    
    ax.set_title(f"{team_name}\n{total_entries} Entradas al Último Tercio", color=line_color, fontsize=18, fontweight='bold')
    
    # Leyenda
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    ax.text(5, 60, f"Pases: {pass_entries}\nConducciones: {carry_entries}", 
            color=line_color, fontsize=12, va='top', ha='left', bbox=bbox_props)
    
    return {
        'Team': team_name,
        'Total_Entries': total_entries,
        'Pass_Entries': pass_entries,
        'Carry_Entries': carry_entries
    }

def plot_box_entries(ax, df, team_name, team_color, bg_color, line_color, is_away=False):
    """Dibuja las entradas al área"""
    # Área: x >= 88.5, y entre 13.85 y 54.15 (área grande)
    passes_box = df[(df['teamName'] == team_name) & 
                    (df['type'] == 'Pass') & 
                    (df['outcomeType'] == 'Successful') &
                    (df['x'] < 88.5) & (df['endX'] >= 88.5) &
                    (df['endY'] >= 13.85) & (df['endY'] <= 54.15)].copy()
    
    carries_box = df[(df['teamName'] == team_name) & 
                     (df['type'] == 'Carry') & 
                     (df['x'] < 88.5) & (df['endX'] >= 88.5) &
                     (df['endY'] >= 13.85) & (df['endY'] <= 54.15)].copy()
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    # Dibujar pases
    if not passes_box.empty:
        pitch.lines(passes_box['x'], passes_box['y'], passes_box['endX'], passes_box['endY'], 
                   lw=2, comet=True, color=team_color, ax=ax, alpha=0.6)
        pitch.scatter(passes_box['endX'], passes_box['endY'], s=50, color=team_color, ax=ax, zorder=3)
    
    # Dibujar conducciones
    for _, row in carries_box.iterrows():
        arrow = patches.FancyArrowPatch(
            (row['x'], row['y']), (row['endX'], row['endY']),
            arrowstyle='->', color='yellow', zorder=4, mutation_scale=15,
            alpha=0.6, linewidth=1.5, linestyle='--'
        )
        ax.add_patch(arrow)
    
    total_entries = len(passes_box) + len(carries_box)
    
    ax.set_title(f"{team_name}\n{total_entries} Entradas al Área", color=line_color, fontsize=18, fontweight='bold')
    
    return {
        'Team': team_name,
        'Box_Entries': total_entries,
        'Pass_Entries': len(passes_box),
        'Carry_Entries': len(carries_box)
    }

def plot_zone14_passes(ax, df, team_name, team_color, bg_color, line_color, is_away=False):
    """Dibuja los pases en la Zona 14 y carriles interiores"""
    # Zona 14: aproximadamente x entre 70-88.5, y entre 22.67-45.33
    zone14_passes = df[(df['teamName'] == team_name) & 
                       (df['type'] == 'Pass') & 
                       (df['outcomeType'] == 'Successful') &
                       (df['x'] >= 70) & (df['x'] <= 88.5) &
                       (df['y'] >= 22.67) & (df['y'] <= 45.33)].copy()
    
    # Half-spaces (carriles interiores): y entre 13.85-22.67 o entre 45.33-54.15
    halfspace_passes = df[(df['teamName'] == team_name) & 
                          (df['type'] == 'Pass') & 
                          (df['outcomeType'] == 'Successful') &
                          (df['x'] >= 70) &
                          (((df['y'] >= 13.85) & (df['y'] < 22.67)) | 
                           ((df['y'] > 45.33) & (df['y'] <= 54.15)))].copy()
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    # Destacar zonas
    zone14_rect = patches.Rectangle((70, 22.67), 18.5, 22.66, linewidth=2, 
                                     edgecolor='yellow', facecolor='yellow', alpha=0.2)
    ax.add_patch(zone14_rect)
    
    # Dibujar pases Zona 14 en color del equipo
    if not zone14_passes.empty:
        pitch.lines(zone14_passes['x'], zone14_passes['y'], zone14_passes['endX'], zone14_passes['endY'], 
                   lw=2, comet=True, color=team_color, ax=ax, alpha=0.7)
    
    # Dibujar pases half-spaces en otro color
    if not halfspace_passes.empty:
        pitch.lines(halfspace_passes['x'], halfspace_passes['y'], halfspace_passes['endX'], halfspace_passes['endY'], 
                   lw=2, comet=True, color='cyan', ax=ax, alpha=0.7)
    
    z14_count = len(zone14_passes)
    hs_count = len(halfspace_passes)
    
    ax.set_title(f"{team_name}\nZona 14: {z14_count} | Half-spaces: {hs_count}", color=line_color, fontsize=16, fontweight='bold')
    
    return {
        'Team': team_name,
        'Zone14_Passes': z14_count,
        'Halfspace_Passes': hs_count
    }

def plot_high_turnovers(ax, df, team_name, team_color, bg_color, line_color, is_away=False):
    """Dibuja las pérdidas en zona alta"""
    # Pérdidas: pases fallados, conducciones perdidas, desposesiones
    turnover_types = ['Pass', 'Carry', 'Dispossessed', 'OffsidePass']
    
    turnovers = df[(df['teamName'] == team_name) & 
                   (df['type'].isin(turnover_types)) & 
                   (df['outcomeType'] == 'Unsuccessful') &
                   (df['x'] >= 70)].copy()
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    # Línea del último tercio
    ax.axvline(x=70, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    
    if not turnovers.empty:
        pitch.scatter(turnovers['x'], turnovers['y'], s=150, color='red', 
                     edgecolors='white', linewidth=1, alpha=0.7, ax=ax)
    
    total_turnovers = len(turnovers)
    
    ax.set_title(f"{team_name}\n{total_turnovers} Pérdidas en Zona Alta", color=line_color, fontsize=18, fontweight='bold')
    
    return {
        'Team': team_name,
        'High_Turnovers': total_turnovers
    }

def plot_crosses(ax, df, team_name, team_color, bg_color, line_color, is_away=False):
    """Dibuja los centros al área"""
    # Centros: pases desde zonas laterales hacia el área
    crosses = df[(df['teamName'] == team_name) & 
                 (df['type'] == 'Pass') &
                 ((df['y'] < 20) | (df['y'] > 48)) &  # Desde las bandas
                 (df['endX'] >= 88.5) &  # Hacia el área
                 (df['endY'] >= 13.85) & (df['endY'] <= 54.15)].copy()
    
    # También buscar en qualifiers si contiene "Cross"
    if 'qualifiers' in df.columns:
        crosses_qual = df[(df['teamName'] == team_name) & 
                          (df['type'] == 'Pass') &
                          (df['qualifiers'].str.contains('Cross', na=False))].copy()
        crosses = pd.concat([crosses, crosses_qual]).drop_duplicates()
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    successful = crosses[crosses['outcomeType'] == 'Successful']
    unsuccessful = crosses[crosses['outcomeType'] == 'Unsuccessful']
    
    # Dibujar centros exitosos
    if not successful.empty:
        pitch.lines(successful['x'], successful['y'], successful['endX'], successful['endY'], 
                   lw=2, comet=True, color=team_color, ax=ax, alpha=0.7)
    
    # Dibujar centros fallados
    if not unsuccessful.empty:
        pitch.lines(unsuccessful['x'], unsuccessful['y'], unsuccessful['endX'], unsuccessful['endY'], 
                   lw=1.5, comet=True, color='gray', ax=ax, alpha=0.4)
    
    total_crosses = len(crosses)
    successful_crosses = len(successful)
    
    ax.set_title(f"{team_name}\n{total_crosses} Centros ({successful_crosses} exitosos)", 
                 color=line_color, fontsize=18, fontweight='bold')
    
    return {
        'Team': team_name,
        'Total_Crosses': total_crosses,
        'Successful_Crosses': successful_crosses
    }

def plot_defensive_actions(ax, df, team_name, team_color, bg_color, line_color, is_away=False):
    """Dibuja las acciones defensivas simples"""
    defensive_types = ['Tackle', 'Interception', 'Clearance', 'BlockedPass', 'Aerial', 
                       'BallRecovery', 'Challenge']
    df_def = df[(df['teamName'] == team_name) & (df['type'].isin(defensive_types))].copy()
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    
    if is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    if df_def.empty:
        ax.set_title(f"{team_name}\nAcciones Defensivas", color=line_color, fontsize=18, fontweight='bold')
        return {}
    
    pitch.scatter(df_def['x'], df_def['y'], s=100, color=team_color, alpha=0.6, ax=ax)
    
    avg_def_height = df_def['x'].mean()
    ax.axvline(x=avg_def_height, color=team_color, linestyle='--', alpha=0.75, linewidth=2)
    
    total_actions = len(df_def)
    ax.set_title(f"{team_name}\n{total_actions} Acciones Defensivas", color=line_color, fontsize=18, fontweight='bold')
    
    if is_away:
        ax.text(avg_def_height - 1, 73, f"{avg_def_height:.1f}m", fontsize=12, color=line_color, ha='left')
    else:
        ax.text(avg_def_height - 1, -5, f"{avg_def_height:.1f}m", fontsize=12, color=line_color, ha='right')
    
    return {
        'Team': team_name,
        'Defensive_Actions': total_actions,
        'Avg_Defensive_Height': round(avg_def_height, 2)
    }

# ============== APLICACIÓN PRINCIPAL ==============

def main():
    st.markdown('<p class="main-header">⚽ Zona del Analista</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis completo de partidos con datos de WhoScored</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Cargar Datos")
        st.markdown("""
        **Instrucciones:**
        1. Ve a [WhoScored.com](https://www.whoscored.com)
        2. Abre el partido que quieres analizar
        3. Guarda la página (Ctrl+S) como HTML
        4. Sube el archivo aquí
        """)
        
        uploaded_file = st.file_uploader("Sube el archivo HTML de WhoScored", type=['html', 'htm'])
        
        st.divider()
        st.header("📝 Información del Partido")
        
        analyst_name = st.text_input("Nombre del Analista", value="", placeholder="Tu nombre")
        match_date = st.date_input("Fecha del Partido", value=date.today())
        fotmob_id = st.text_input("FotMob Match ID", value="", placeholder="Ej: 4621571")
        
        st.divider()
        st.header("🎨 Personalización")
        
        bg_color = st.color_picker("Color de fondo", DEFAULT_COLORS['bg_color'])
        line_color = st.color_picker("Color de líneas", DEFAULT_COLORS['line_color'])
        home_color = st.color_picker("Color equipo local", DEFAULT_COLORS['home_color'])
        away_color = st.color_picker("Color equipo visitante", DEFAULT_COLORS['away_color'])
    
    if uploaded_file is not None:
        try:
            html_content = uploaded_file.read().decode('utf-8')
            
            with st.spinner('Extrayendo datos del partido...'):
                json_data_txt = extract_json_from_html(html_content)
                data = json.loads(json_data_txt)
                events_dict, players_df, teams_dict = extract_data_from_dict(data)
                
                df = pd.DataFrame(events_dict)
                df = process_events_dataframe(df, teams_dict)
                df = insert_ball_carries(df)
                
                home_team_id = list(teams_dict.keys())[0]
                away_team_id = list(teams_dict.keys())[1]
                home_team = teams_dict[home_team_id]
                away_team = teams_dict[away_team_id]
                
                # Crear columna de nombre corto
                if 'name' not in df.columns:
                    df = df.merge(players_df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], 
                                 on='playerId', how='left')
                
                df['shortName'] = df['name'].apply(get_short_name) if 'name' in df.columns else ''
                
                # Obtener pases con receptor
                passes_df = get_passes_df(df)
            
            # Mostrar información del partido
            st.success(f"✅ Datos cargados: {len(df)} eventos")
            
            # Info card del partido
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.markdown(f"### 🏠 {home_team}")
            with col2:
                st.markdown("### VS")
            with col3:
                st.markdown(f"### ✈️ {away_team}")
            
            # Mostrar metadatos
            if analyst_name or fotmob_id:
                meta_text = []
                if analyst_name:
                    meta_text.append(f"📊 Analista: **{analyst_name}**")
                if fotmob_id:
                    meta_text.append(f"🔗 FotMob ID: **{fotmob_id}**")
                meta_text.append(f"📅 Fecha: **{match_date}**")
                st.markdown(" | ".join(meta_text))
            
            st.divider()
            
            # Tabs de visualizaciones
            tabs = st.tabs([
                "🔗 Red de Pases", 
                "🛡️ Bloque Defensivo",
                "📈 Pases Progresivos",
                "🏃 Conducciones",
                "⚽ Mapa de Tiros",
                "🎯 Tiros Individual",
                "🚀 Entradas Último Tercio",
                "📦 Entradas al Área",
                "🎯 Zona 14",
                "❌ Pérdidas Zona Alta",
                "⚔️ Centros",
                "🛡️ Acciones Defensivas",
                "📊 Datos Raw"
            ])
            
            # Tab 1: Pass Network
            with tabs[0]:
                st.subheader("Red de Pases")
                
                home_passes_between, home_avg_locs = get_passes_between_df(home_team, passes_df, players_df, df)
                away_passes_between, away_avg_locs = get_passes_between_df(away_team, passes_df, players_df, df)
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
                
                plot_pass_network(axes[0], home_passes_between, home_avg_locs, 
                                 home_color, home_team, bg_color, line_color, is_away=False)
                plot_pass_network(axes[1], away_passes_between, away_avg_locs, 
                                 away_color, away_team, bg_color, line_color, is_away=True)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 2: Bloque Defensivo
            with tabs[1]:
                st.subheader("Bloque Defensivo")
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
                
                plot_defensive_block(axes[0], df, home_team, home_color, bg_color, line_color, players_df, is_away=False)
                plot_defensive_block(axes[1], df, away_team, away_color, bg_color, line_color, players_df, is_away=True)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 3: Pases Progresivos
            with tabs[2]:
                st.subheader("Pases Progresivos")
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
                
                plot_progressive_passes(axes[0], df, home_team, home_color, bg_color, line_color, is_away=False)
                plot_progressive_passes(axes[1], df, away_team, away_color, bg_color, line_color, is_away=True)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 4: Conducciones Progresivas
            with tabs[3]:
                st.subheader("Conducciones Progresivas")
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
                
                plot_progressive_carries(axes[0], df, home_team, home_color, bg_color, line_color, is_away=False)
                plot_progressive_carries(axes[1], df, away_team, away_color, bg_color, line_color, is_away=True)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 5: Mapa de Tiros Combinado
            with tabs[4]:
                st.subheader("Mapa de Tiros (Ambos Equipos)")
                
                fig, ax = plt.subplots(figsize=(14, 10), facecolor=bg_color)
                
                plot_shotmap_combined(ax, df, home_team, away_team, home_color, away_color, bg_color, line_color)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 6: Mapa de Tiros Individual
            with tabs[5]:
                st.subheader("Mapa de Tiros (Por Equipo)")
                
                fig, axes = plt.subplots(1, 2, figsize=(16, 10), facecolor=bg_color)
                
                home_shot_stats = plot_shotmap_individual(axes[0], df, home_team, home_color, bg_color, line_color)
                away_shot_stats = plot_shotmap_individual(axes[1], df, away_team, away_color, bg_color, line_color)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Mostrar estadísticas
                col1, col2 = st.columns(2)
                with col1:
                    if home_shot_stats:
                        st.markdown(f"**{home_team}**")
                        st.write(f"- Tiros: {home_shot_stats.get('Shots', 0)}")
                        st.write(f"- Goles: {home_shot_stats.get('Goals', 0)}")
                        st.write(f"- A puerta: {home_shot_stats.get('Shots_on_Target', 0)}")
                        st.write(f"- Distancia media: {home_shot_stats.get('Avg_Distance', 0)}m")
                with col2:
                    if away_shot_stats:
                        st.markdown(f"**{away_team}**")
                        st.write(f"- Tiros: {away_shot_stats.get('Shots', 0)}")
                        st.write(f"- Goles: {away_shot_stats.get('Goals', 0)}")
                        st.write(f"- A puerta: {away_shot_stats.get('Shots_on_Target', 0)}")
                        st.write(f"- Distancia media: {away_shot_stats.get('Avg_Distance', 0)}m")
            
            # Tab 7: Entradas Último Tercio
            with tabs[6]:
                st.subheader("Entradas al Último Tercio")
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
                
                plot_final_third_entries(axes[0], df, home_team, home_color, bg_color, line_color, is_away=False)
                plot_final_third_entries(axes[1], df, away_team, away_color, bg_color, line_color, is_away=True)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 8: Entradas al Área
            with tabs[7]:
                st.subheader("Entradas al Área")
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
                
                plot_box_entries(axes[0], df, home_team, home_color, bg_color, line_color, is_away=False)
                plot_box_entries(axes[1], df, away_team, away_color, bg_color, line_color, is_away=True)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 9: Zona 14
            with tabs[8]:
                st.subheader("Pases en Zona 14 y Half-Spaces")
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
                
                plot_zone14_passes(axes[0], df, home_team, home_color, bg_color, line_color, is_away=False)
                plot_zone14_passes(axes[1], df, away_team, away_color, bg_color, line_color, is_away=True)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 10: Pérdidas Zona Alta
            with tabs[9]:
                st.subheader("Pérdidas en Zona Alta")
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
                
                plot_high_turnovers(axes[0], df, home_team, home_color, bg_color, line_color, is_away=False)
                plot_high_turnovers(axes[1], df, away_team, away_color, bg_color, line_color, is_away=True)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 11: Centros
            with tabs[10]:
                st.subheader("Centros al Área")
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
                
                plot_crosses(axes[0], df, home_team, home_color, bg_color, line_color, is_away=False)
                plot_crosses(axes[1], df, away_team, away_color, bg_color, line_color, is_away=True)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 12: Acciones Defensivas
            with tabs[11]:
                st.subheader("Acciones Defensivas")
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
                
                plot_defensive_actions(axes[0], df, home_team, home_color, bg_color, line_color, is_away=False)
                plot_defensive_actions(axes[1], df, away_team, away_color, bg_color, line_color, is_away=True)
                
                if analyst_name:
                    add_footer(fig, analyst_name, match_date, fotmob_id if fotmob_id else "N/A")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 13: Datos Raw
            with tabs[12]:
                st.subheader("Datos del Partido")
                
                # Mostrar tipos de eventos disponibles
                with st.expander("🔍 Tipos de eventos en el partido"):
                    event_types = df['type'].value_counts()
                    st.dataframe(event_types.reset_index())
                
                st.markdown("**Eventos del partido (primeros 100):**")
                cols_to_show = ['minute', 'second', 'teamName', 'type', 'outcomeType', 'x', 'y', 'endX', 'endY']
                available_cols = [c for c in cols_to_show if c in df.columns]
                st.dataframe(df[available_cols].head(100))
                
                st.markdown("**Jugadores:**")
                st.dataframe(players_df)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar eventos (CSV)",
                    data=csv,
                    file_name=f"{home_team}_vs_{away_team}_events.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
            st.info("Asegúrate de que el archivo HTML es de un partido de WhoScored y está completo.")
            with st.expander("Ver detalles del error"):
                st.exception(e)
    
    else:
        st.info("👆 Sube un archivo HTML de WhoScored para comenzar el análisis")
        
        with st.expander("📖 ¿Cómo obtener el archivo HTML?"):
            st.markdown("""
            1. Ve a [WhoScored.com](https://www.whoscored.com)
            2. Busca el partido que quieres analizar
            3. Abre la página del partido (Live/Match Centre)
            4. Espera a que carguen todos los datos
            5. Presiona **Ctrl+S** (o Cmd+S en Mac)
            6. Guarda como "Página web completa" o "HTML solamente"
            7. Sube el archivo .html aquí
            """)
        
        with st.expander("🔧 Visualizaciones disponibles"):
            st.markdown("""
            - **Red de Pases**: Conexiones entre jugadores con posiciones medias
            - **Bloque Defensivo**: Heatmap de acciones defensivas con posiciones
            - **Pases Progresivos**: Pases que avanzan >9m hacia portería
            - **Conducciones Progresivas**: Carreras con balón progresivas
            - **Mapa de Tiros (Combinado)**: Ambos equipos en un campo
            - **Mapa de Tiros (Individual)**: Cada equipo por separado
            - **Entradas al Último Tercio**: Pases y conducciones al tercio final
            - **Entradas al Área**: Penetraciones al área grande
            - **Zona 14 y Half-Spaces**: Pases en zonas peligrosas
            - **Pérdidas en Zona Alta**: Balones perdidos en campo rival
            - **Centros**: Centros al área desde las bandas
            - **Acciones Defensivas**: Tackles, intercepciones, despejes
            - **Datos Raw**: Acceso a todos los eventos del partido
            """)

if __name__ == "__main__":
    main()
