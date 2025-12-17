import streamlit as st
import json
import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Diagnóstico WhoScored", layout="wide")

st.title("🔍 Diagnóstico de Archivo WhoScored")

uploaded_file = st.file_uploader("Sube el archivo HTML de WhoScored", type=['html', 'htm'])

if uploaded_file is not None:
    try:
        html_content = uploaded_file.read().decode('utf-8')
        st.success(f"✅ Archivo cargado: {len(html_content)} caracteres")
        
        # Intentar extraer JSON
        st.subheader("1. Extracción de JSON")
        
        regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
        matches = re.findall(regex_pattern, html_content)
        
        if matches:
            st.success(f"✅ Se encontró el patrón JSON ({len(matches)} coincidencias)")
            data_txt = matches[0]
            
            # Mostrar primeros caracteres
            st.text("Primeros 500 caracteres del JSON encontrado:")
            st.code(data_txt[:500])
            
            # Procesar JSON
            data_txt = data_txt.replace('matchId', '"matchId"')
            data_txt = data_txt.replace('matchCentreData', '"matchCentreData"')
            data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
            data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"')
            data_txt = data_txt.replace('};', '}')
            
            try:
                data = json.loads(data_txt)
                st.success("✅ JSON parseado correctamente")
                
                # Mostrar estructura
                st.subheader("2. Estructura de Datos")
                st.write("**Claves principales:**", list(data.keys()))
                
                if "matchCentreData" in data:
                    mcd = data["matchCentreData"]
                    st.write("**Claves en matchCentreData:**", list(mcd.keys()))
                    
                    # Equipos
                    st.subheader("3. Equipos")
                    if 'home' in mcd and 'away' in mcd:
                        home_name = mcd['home'].get('name', 'N/A')
                        away_name = mcd['away'].get('name', 'N/A')
                        home_id = mcd['home'].get('teamId', 'N/A')
                        away_id = mcd['away'].get('teamId', 'N/A')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Equipo Local", home_name)
                            st.write(f"ID: {home_id}")
                        with col2:
                            st.metric("Equipo Visitante", away_name)
                            st.write(f"ID: {away_id}")
                    
                    # Eventos
                    st.subheader("4. Eventos")
                    if 'events' in mcd:
                        events = mcd['events']
                        st.write(f"**Total de eventos:** {len(events)}")
                        
                        if len(events) > 0:
                            df = pd.DataFrame(events)
                            st.write(f"**Columnas disponibles:** {list(df.columns)}")
                            
                            # Mostrar primeros eventos
                            st.write("**Primeros 5 eventos (raw):**")
                            st.json(events[:5])
                            
                            # Procesar type
                            st.subheader("5. Tipos de Eventos")
                            
                            # Ver estructura del campo type
                            st.write("**Estructura del campo 'type' (primer evento):**")
                            st.write(type(events[0].get('type')))
                            st.write(events[0].get('type'))
                            
                            # Extraer tipos
                            def get_type(x):
                                if isinstance(x, dict):
                                    return x.get('displayName', str(x))
                                return str(x)
                            
                            df['type_extracted'] = df['type'].apply(get_type)
                            
                            type_counts = df['type_extracted'].value_counts()
                            st.write("**Conteo de tipos de eventos:**")
                            st.dataframe(type_counts)
                            
                            # Verificar tipos de tiros
                            st.subheader("6. Verificación de Tiros")
                            shot_types = ['Goal', 'SavedShot', 'MissedShots', 'ShotOnPost', 
                                         'AttemptSaved', 'MissedShot', 'Miss', 'Post', 'BlockedShot']
                            
                            found_shots = [t for t in shot_types if t in type_counts.index]
                            missing_shots = [t for t in shot_types if t not in type_counts.index]
                            
                            if found_shots:
                                st.success(f"✅ Tipos de tiro encontrados: {found_shots}")
                            else:
                                st.warning("⚠️ No se encontraron tipos de tiro estándar")
                            
                            if missing_shots:
                                st.info(f"ℹ️ Tipos de tiro no encontrados: {missing_shots}")
                            
                            # Verificar coordenadas
                            st.subheader("7. Verificación de Coordenadas")
                            coord_cols = ['x', 'y', 'endX', 'endY']
                            for col in coord_cols:
                                if col in df.columns:
                                    valid = df[col].notna().sum()
                                    st.write(f"**{col}:** {valid} valores válidos, rango [{df[col].min():.1f}, {df[col].max():.1f}]")
                                else:
                                    st.warning(f"⚠️ Columna '{col}' no encontrada")
                            
                            # Jugadores
                            st.subheader("8. Jugadores")
                            if 'home' in mcd and 'players' in mcd['home']:
                                home_players = pd.DataFrame(mcd['home']['players'])
                                st.write(f"**Jugadores locales:** {len(home_players)}")
                                st.write("**Columnas de jugadores:**", list(home_players.columns))
                                st.dataframe(home_players[['playerId', 'name', 'shirtNo', 'position']].head(15) if all(c in home_players.columns for c in ['playerId', 'name', 'shirtNo', 'position']) else home_players.head(15))
                            
                            if 'away' in mcd and 'players' in mcd['away']:
                                away_players = pd.DataFrame(mcd['away']['players'])
                                st.write(f"**Jugadores visitantes:** {len(away_players)}")
                                st.dataframe(away_players[['playerId', 'name', 'shirtNo', 'position']].head(15) if all(c in away_players.columns for c in ['playerId', 'name', 'shirtNo', 'position']) else away_players.head(15))
                            
                            # Dataframe procesado
                            st.subheader("9. DataFrame de Eventos Procesado")
                            
                            # Procesar outcomeType
                            def get_outcome(x):
                                if isinstance(x, dict):
                                    return x.get('displayName', str(x))
                                return str(x)
                            
                            df['outcomeType_extracted'] = df['outcomeType'].apply(get_outcome) if 'outcomeType' in df.columns else 'N/A'
                            
                            # Escalar coordenadas
                            if 'x' in df.columns:
                                df['x_scaled'] = df['x'] * 1.05
                            if 'y' in df.columns:
                                df['y_scaled'] = df['y'] * 0.68
                            
                            cols_to_show = ['minute', 'second', 'type_extracted', 'outcomeType_extracted', 'x', 'y', 'x_scaled', 'y_scaled', 'endX', 'endY']
                            available = [c for c in cols_to_show if c in df.columns]
                            
                            st.dataframe(df[available].head(50))
                            
                            # Descargar CSV de diagnóstico
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "📥 Descargar CSV completo de diagnóstico",
                                csv,
                                "diagnostico_eventos.csv",
                                "text/csv"
                            )
                    else:
                        st.error("❌ No se encontró la clave 'events' en matchCentreData")
                else:
                    st.error("❌ No se encontró 'matchCentreData' en el JSON")
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ Error al parsear JSON: {e}")
                st.text("JSON problemático (primeros 1000 chars):")
                st.code(data_txt[:1000])
        else:
            st.error("❌ No se encontró el patrón de datos en el HTML")
            st.info("Esto puede significar que:")
            st.markdown("""
            - El archivo HTML no es de WhoScored
            - El archivo está incompleto
            - WhoScored cambió su formato de datos
            """)
            
            # Buscar patrones alternativos
            st.subheader("Búsqueda de patrones alternativos")
            patterns = [
                r'matchCentreData',
                r'events.*?\[',
                r'teamId',
                r'playerId'
            ]
            for pattern in patterns:
                found = re.search(pattern, html_content)
                if found:
                    st.write(f"✅ Patrón '{pattern}' encontrado en posición {found.start()}")
                else:
                    st.write(f"❌ Patrón '{pattern}' NO encontrado")
                    
    except Exception as e:
        st.error(f"❌ Error general: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
