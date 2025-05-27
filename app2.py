import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import requests
import datetime
import joblib
import ee
import geemap.foliumap as geemap
import shutil
import os

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DatetimeTickFormatter
from bokeh.embed import file_html
from bokeh.resources import CDN

os.makedirs(".streamlit", exist_ok=True)
shutil.copy(
    "/content/drive/MyDrive/secrets/secrets.toml",
    ".streamlit/secrets.toml"
)

scaler_X = joblib.load("/content/drive/MyDrive/MIAA/Clases/Herramientas IA/Prácticas/balance_hidrico/model/pkl/scaler_X.pkl")
scaler_y = joblib.load("/content/drive/MyDrive/MIAA/Clases/Herramientas IA/Prácticas/balance_hidrico/model/pkl/scaler_y.pkl")
xgb_model_best = joblib.load("/content/drive/MyDrive/MIAA/Clases/Herramientas IA/Prácticas/balance_hidrico/model/pkl/modelo_xgboost_v1.pkl")

def calcPet(img):
    T  = img.select('temperature_2m').subtract(273.15)
    Td = img.select('dewpoint_temperature_2m').subtract(273.15)
    Ra = img.select('surface_net_solar_radiation').add(img.select('surface_net_thermal_radiation')).divide(1e6)
    date = img.date()
    days = ee.Image.constant(date.advance(1, 'month').difference(date, 'day'))
    Rn = Ra.divide(days)
    G = ee.Image.constant(0)
    es = T.expression('0.6108 * exp(17.27 * T / (T + 237.3))', {'T': T})
    ea = Td.expression('0.6108 * exp(17.27 * Td / (Td + 237.3))', {'Td': Td})
    delta = es.multiply(4098).divide(T.add(237.3).pow(2))
    gamma = ee.Image.constant(1.013e-3 * 101.3 / (0.622 * 2.45))
    u2m = img.select('u_component_of_wind_10m')
    v2m = img.select('v_component_of_wind_10m')
    wind_speed = u2m.pow(2).add(v2m.pow(2)).sqrt().multiply(4.87).divide(ee.Number(67.8 * 10 - 5.42).log())
    pet = delta.multiply(Rn.subtract(G)).multiply(0.408).add(
        gamma.multiply(900).divide(T.add(273)).multiply(wind_speed).multiply(es.subtract(ea))
    ).divide(
        delta.add(gamma.multiply(ee.Image.constant(1).add(wind_speed.multiply(0.34))))
    ).rename('PET')
    return pet.copyProperties(img, img.propertyNames())



def mostrar_balance_hidrico_con_capas(year, month, pais='Ecuador', provincia='Carchi'):
    ee.Initialize(project= "ee-freddyvillota")
    from datetime import datetime as dt

    roi = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1") \
        .filter(ee.Filter.eq("ADM0_NAME", pais)) \
        .filter(ee.Filter.eq("ADM1_NAME", provincia))

    era5_col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY")
    fecha_ini = f"{year}-{month:02d}-01"
    fecha_fin = f"{year}-{month:02d}-28"
    era5 = era5_col.filterDate(fecha_ini, fecha_fin)

    petCol = era5.map(calcPet)
    pet = petCol.first().clip(roi).resample('bicubic')

    precipCol = era5.select('total_precipitation') \
        .map(lambda img: img.multiply(1000).rename('precip_mm') \
             .copyProperties(img, img.propertyNames()))
    precip = precipCol.first().clip(roi).resample('bicubic')

    balance = precip.subtract(pet).rename('balance_mm').clip(roi).resample('bicubic')


        # 6) Estadísticas (para visualización automática)
    stats_bal = balance.reduceRegion(ee.Reducer.minMax(), roi.geometry(), 10000).getInfo()
    stats_pet = pet.reduceRegion(ee.Reducer.minMax(), roi.geometry(), 10000).getInfo()
    stats_precip = precip.reduceRegion(ee.Reducer.minMax(), roi.geometry(), 10000).getInfo()

    vis_bal = {'min': stats_bal['balance_mm_min'], 'max': stats_bal['balance_mm_max'],
               'palette': ['red', 'yellow', 'green', 'blue']}
    vis_precip = {'min': stats_precip['precip_mm_min'], 'max': stats_precip['precip_mm_max'],
                  'palette': ['white', 'lightblue', 'blue', 'darkblue']}
    vis_pet = {'min': stats_pet['PET_min'], 'max': stats_pet['PET_max'],
               'palette': ['lightyellow', 'orange', 'red']}

    # También mostrar en Streamlit
    st.info(f"🟢 Visualizando datos para {provincia} ({pais}) - {year}-{month:02d}")
    st.write(f"**Balance hídrico (mm):** {stats_bal['balance_mm_min']:.2f} – {stats_bal['balance_mm_max']:.2f}")
    st.write(f"**Precipitación (mm):** {stats_precip['precip_mm_min']:.2f} – {stats_precip['precip_mm_max']:.2f}")
    st.write(f"**PET (mm):** {stats_pet['PET_min']:.2f} – {stats_pet['PET_max']:.2f}")

    Map = geemap.Map()
    Map.add_basemap("HYBRID")
    Map.centerObject(roi, zoom=8)
    Map.addLayer(balance, vis_bal, f'Balance {provincia} {year}-{month:02d}')
    Map.add_colorbar(vis_params=vis_bal, label='Balance hídrico (mm)')

    Map.addLayer(precip, vis_precip, f'Precipitación {provincia}')
    Map.add_colorbar(vis_params=vis_precip, label='Precipitación (mm)')

    Map.addLayer(pet, vis_pet, f'PET {provincia}')
    Map.add_colorbar(vis_params=vis_pet, label='Evapotranspiración potencial (mm)')

    Map.addLayer(roi.style(color='black', width=2, fillColor='00000000'), {}, provincia)

    return Map



def cargar_datos_supabase():
    url = st.secrets["SUPABASE_URL"]
    api_key = st.secrets["SUPABASE_API_KEY"]
    endpoint = f"{url}/rest/v1/balance_proyectado"
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get(endpoint, headers=headers, params={"select": "*"})
    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        st.error("Error cargando datos desde Supabase")
        return None

def procesar_datos_cmip6(df_cmip6):
    df_cmip6['temp_c'] = df_cmip6['temp_c'] - 273.15
    df_cmip6['precip_mm'] = df_cmip6['precip_mm'] * 86400 * df_cmip6['date'].dt.days_in_month
    df_cmip6['solar_rad'] = df_cmip6['solar_rad'] * 3600 * 24 * df_cmip6['date'].dt.days_in_month / 1e6
    return df_cmip6


st.image("/content/drive/MyDrive/MIAA/Clases/Herramientas IA/Prácticas/miaa-final-project/images/utpl.png", width=250)
st.markdown("""
## UNIVERSIDAD TÉCNICA PARTICULAR DE LOJA  
### FACULTAD DE INGENIERÍAS Y ARQUITECTURA  
### MAESTRÍA EN INTELIGENCIA ARTIFICIAL APLICADA

**Autor:** Freddy Hernán Villota González  
**Docente:** M.Sc. Alexandra Cristina González Eras  
**Fecha:** 16 de mayo de 2025
""")

# ------------------ SIDEBAR ------------------
st.sidebar.header("🔍 Selección de parámetros")
opcion = st.sidebar.radio("¿Qué deseas visualizar?", [
    "Balance hídrico observado (ERA5)",
    "Balance hídrico histórico en Carchi",
    "Proyección futura en Carchi con CMIP6"
])

# ------------------ BALANCE HIDRICO OBSERVADO ------------------
if opcion == "Balance hídrico observado (ERA5)":
    pais = st.sidebar.text_input("País", "Ecuador")
    provincia = st.sidebar.text_input("Provincia", "Carchi")
    anio = st.sidebar.number_input("Año", min_value=1981, max_value=2023, value=2022)
    mes = st.sidebar.number_input("Mes", min_value=1, max_value=12, value=7)
    if st.sidebar.button("Visualizar mapa"):
        mapa = mostrar_balance_hidrico_con_capas(anio, mes, pais, provincia)
        with st.container():
            st.markdown("### 🗺️ Mapa de balance hídrico")
            mapa.to_streamlit(height=500)

# ------------------ BALANCE HISTÓRICO ------------------
elif opcion == "Balance hídrico histórico en Carchi":
    st.markdown("### 📊 Análisis histórico del balance hídrico mensual en Carchi (Ecuador)")
    df_hist = pd.read_csv("/content/drive/MyDrive/MIAA/Clases/Herramientas IA/Prácticas/balance_hidrico/data/era5/Carchi/df_balanceH_historico.csv")
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    df_hist['year'] = df_hist['date'].dt.year
    df_hist['month'] = df_hist['date'].dt.month

    # Selección de rango de fechas detallado
    col1, col2 = st.columns(2)
    with col1:
        anio_ini = st.selectbox("Año inicial", sorted(df_hist['year'].unique()), index=0)
        mes_ini = st.selectbox("Mes inicial", list(range(1, 13)), index=0)
    with col2:
        anio_fin = st.selectbox("Año final", sorted(df_hist['year'].unique()), index=len(df_hist['year'].unique()) - 1)
        mes_fin = st.selectbox("Mes final", list(range(1, 13)), index=11)

    fecha_ini = pd.to_datetime(f"{anio_ini}-{mes_ini:02d}-01")
    fecha_fin = pd.to_datetime(f"{anio_fin}-{mes_fin:02d}-28")
    df_filtrado = df_hist[(df_hist['date'] >= fecha_ini) & (df_hist['date'] <= fecha_fin)]

    # Mensual: Línea continua
    st.markdown("#### 📈 Balance hídrico mensual")
    source_mensual = ColumnDataSource(df_filtrado)
    p1 = figure(title="Balance hídrico mensual en Carchi (mm)", x_axis_type='datetime', width=900, height=300)
    p1.line('date', 'balance_mm', source=source_mensual, line_width=2, color='royalblue', legend_label='Balance hídrico')
    p1.xaxis.axis_label = "Fecha"
    p1.yaxis.axis_label = "Balance hídrico (mm/mes)"
    p1.grid.grid_line_alpha = 0.4
    p1.legend.location = "top_left"
    st.components.v1.html(file_html(p1, CDN, "balance_mensual"), height=350)

    # Anual: promedio por año
    st.markdown("#### 📉 Tendencia anual (promedio)")
    df_anual = df_filtrado.groupby('year')['balance_mm'].mean().reset_index()
    source_anual = ColumnDataSource(df_anual)
    p2 = figure(title=f"Tendencia del balance hídrico medio anual en Carchi ({anio_ini}–{anio_fin})", width=900, height=300)
    p2.line('year', 'balance_mm', source=source_anual, line_width=2, color='royalblue', legend_label='Balance hídrico medio anual')
    p2.circle('year', 'balance_mm', source=source_anual, size=6, color='royalblue')
    p2.xaxis.axis_label = "Año"
    p2.yaxis.axis_label = "Balance hídrico (mm/mes)"
    p2.grid.grid_line_alpha = 0.4
    p2.legend.location = "top_left"
    st.components.v1.html(file_html(p2, CDN, "balance_anual"), height=350)

    # 📊 Gráfico de anomalías históricas: barras rojas y azules
    st.markdown("#### 🔍 Anomalías respecto a la media del periodo")
    media_anual = df_anual['balance_mm'].mean()
    df_anual['anomalia'] = df_anual['balance_mm'] - media_anual
    df_anual['color'] = df_anual['anomalia'].apply(lambda x: 'red' if x < 0 else 'blue')
    source_anom = ColumnDataSource(df_anual)

    p3 = figure(title="Anomalías del balance hídrico medio mensual en Carchi", width=900, height=300)
    p3.vbar(x='year', top='anomalia', source=source_anom, width=0.7, color='color', legend_label='Anomalía balance hídrico')
    p3.xaxis.axis_label = "Año"
    p3.yaxis.axis_label = "Anomalía (mm/mes)"
    p3.grid.grid_line_alpha = 0.4
    p3.legend.location = "top_left"
    st.components.v1.html(file_html(p3, CDN, "anomalia_hist"), height=350)


# ------------------ PROYECCIÓN FUTURA ------------------
# ------------------ PROYECCIÓN FUTURA ------------------
elif opcion == "Proyección futura en Carchi con CMIP6":
    st.markdown("### 📊 Proyección del balance hídrico mensual en Carchi (Ecuador)")
    df_cmip6 = cargar_datos_supabase()
    if df_cmip6 is not None:
        df_cmip6['date'] = pd.to_datetime(df_cmip6['date'])
        df_cmip6['year'] = df_cmip6['date'].dt.year
        df_cmip6['month'] = df_cmip6['date'].dt.month

        col1, col2 = st.columns(2)
        with col1:
            anio_ini = st.selectbox("Año inicial", sorted(df_cmip6['year'].unique()), index=0)
            mes_ini = st.selectbox("Mes inicial", list(range(1, 13)), index=0)
        with col2:
            anio_fin = st.selectbox("Año final", sorted(df_cmip6['year'].unique()), index=len(df_cmip6['year'].unique()) - 1)
            mes_fin = st.selectbox("Mes final", list(range(1, 13)), index=11)

        fecha_ini = pd.to_datetime(f"{anio_ini}-{mes_ini:02d}-01")
        fecha_fin = pd.to_datetime(f"{anio_fin}-{mes_fin:02d}-28")
        df_filtrado = df_cmip6[(df_cmip6['date'] >= fecha_ini) & (df_cmip6['date'] <= fecha_fin)]

        if st.button("Generar proyección"):
            df_filtrado = procesar_datos_cmip6(df_filtrado)

            X_new = scaler_X.transform(df_filtrado[['precip_mm', 'temp_c', 'wind_u', 'wind_v', 'solar_rad']])
            y_pred_xgb_new_scaled = xgb_model_best.predict(X_new)
            y_pred_xgb_new = scaler_y.inverse_transform(y_pred_xgb_new_scaled.reshape(-1, 1))
            df_filtrado['balance_pred_xgb'] = y_pred_xgb_new

            # 📈 Gráfico mensual
            st.markdown("#### 📈 Balance hídrico mensual proyectado")
            src_month = ColumnDataSource(df_filtrado)
            p1 = figure(title="Balance hídrico mensual proyectado en Carchi", x_axis_type='datetime', width=900, height=300)
            p1.line('date', 'balance_pred_xgb', source=src_month, line_width=2, color='royalblue', legend_label='Proyección mensual')
            p1.xaxis.axis_label = "Fecha"
            p1.yaxis.axis_label = "Balance hídrico (mm/mes)"
            p1.grid.grid_line_alpha = 0.4
            p1.legend.location = "top_left"
            st.components.v1.html(file_html(p1, CDN, "proy_mensual"), height=350)

            # 📉 Promedio anual
            st.markdown("#### 📉 Tendencia anual (promedio)")
            df_anual_fut = df_filtrado.groupby('year')['balance_pred_xgb'].mean().reset_index()
            src_fut = ColumnDataSource(df_anual_fut)
            p2 = figure(title=f"Tendencia del balance hídrico medio anual proyectado ({anio_ini}–{anio_fin})", width=900, height=300)
            p2.line('year', 'balance_pred_xgb', source=src_fut, line_width=2, color='royalblue', legend_label='Balance hídrico medio anual')
            p2.circle('year', 'balance_pred_xgb', source=src_fut, size=6, color='royalblue')
            p2.xaxis.axis_label = "Año"
            p2.yaxis.axis_label = "Balance hídrico (mm/mes)"
            p2.grid.grid_line_alpha = 0.4
            p2.legend.location = "top_left"
            st.components.v1.html(file_html(p2, CDN, "proy_anual"), height=350)

            # 📊 Anomalías
            st.markdown("#### 🔍 Anomalías respecto al promedio proyectado")
            media_fut = df_anual_fut['balance_pred_xgb'].mean()
            df_anual_fut['anomalia'] = df_anual_fut['balance_pred_xgb'] - media_fut
            df_anual_fut['color'] = df_anual_fut['anomalia'].apply(lambda x: 'red' if x < 0 else 'blue')
            src_anom_fut = ColumnDataSource(df_anual_fut)
            p3 = figure(title="Anomalías del balance hídrico medio anual proyectado en Carchi", width=900, height=300)
            p3.vbar(x='year', top='anomalia', source=src_anom_fut, width=0.7, color='color', legend_label='Anomalía balance hídrico')
            p3.xaxis.axis_label = "Año"
            p3.yaxis.axis_label = "Anomalía (mm/mes)"
            p3.grid.grid_line_alpha = 0.4
            p3.legend.location = "top_left"
            st.components.v1.html(file_html(p3, CDN, "anomalia_futura"), height=350)
