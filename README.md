# Zona del Analista - WhoScored Match Analysis

Una aplicación de Streamlit que genera dashboards de análisis de partidos automáticamente a partir de archivos HTML de WhoScored.

## 🚀 Características

- **Pass Network**: Visualiza la red de pases de cada equipo
- **Pases Progresivos**: Mapa de pases que avanzan hacia portería
- **Conducciones Progresivas**: Carreras con balón significativas
- **Mapa de Tiros Combinado**: Ambos equipos en un campo completo
- **Mapa de Tiros Individual**: Cada equipo por separado con estadísticas detalladas
- **Acciones Defensivas**: Tackles, intercepciones, despejes
- **Metadatos del partido**: Nombre del analista, fecha, FotMob ID
- **Personalización**: Colores personalizables para equipos y fondo
- **Exportación**: Descarga los datos en CSV

## 📦 Instalación

### Opción 1: Local

```bash
# Clonar o descargar los archivos
cd zona_del_analista

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

### Opción 2: Streamlit Cloud

1. Sube los archivos a un repositorio de GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Despliega la app

## 📖 Cómo usar

### Paso 1: Obtener el archivo HTML de WhoScored

1. Ve a [WhoScored.com](https://www.whoscored.com)
2. Busca y abre el partido que quieres analizar
3. Ve a la sección "Live" o "Match Centre"
4. **Espera a que carguen todos los datos** (importante!)
5. Presiona `Ctrl+S` (Windows/Linux) o `Cmd+S` (Mac)
6. Guarda como "Página web completa" o "HTML only"

### Paso 2: Analizar el partido

1. Abre la aplicación Streamlit
2. Sube el archivo HTML en el sidebar
3. Espera a que se procesen los datos
4. Explora las diferentes visualizaciones en las pestañas

## 🎨 Personalización

En el sidebar puedes cambiar:
- Color de fondo de los gráficos
- Color de las líneas del campo
- Color del equipo local
- Color del equipo visitante

## ⚠️ Notas importantes

- La aplicación funciona **solo con archivos HTML descargados** de WhoScored
- No es posible hacer scraping directo de la URL porque WhoScored usa JavaScript dinámico
- Asegúrate de que la página esté completamente cargada antes de guardar el HTML
- Algunos partidos pueden tener datos incompletos

## 🛠️ Tecnologías utilizadas

- **Streamlit**: Framework web
- **mplsoccer**: Visualizaciones de fútbol
- **Pandas**: Procesamiento de datos
- **Matplotlib**: Gráficos

## 📝 Licencia

Este proyecto es para uso educativo y personal.

## 🤝 Créditos

Basado en el trabajo de la comunidad de football analytics, especialmente:
- [mplsoccer](https://github.com/andrewRowlinson/mplsoccer)
- [Friends of Tracking](https://github.com/Friends-of-Tracking-Data-FoTD)
