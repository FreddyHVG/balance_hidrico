
# UNIVERSIDAD TÉCNICA PARTICULAR DE LOJA

<img src="https://drive.google.com/uc?id=1X5UmWVlUX9XmckJgFLmv6mTTX81GEr0c" width="300">

## FACULTAD DE INGENIERÍAS Y ARQUITECTURA  
### MAESTRÍA EN INTELIGENCIA ARTIFICIAL APLICADA

---

## Trabajo académico final escrito: Generar una aplicación de inteligencia artificial que use librerías de software libre a través de herramientas colaborativas

**Autor:** Freddy Hernán Villota González  
**Docente:** M.Sc. Alexandra Cristina González Eras  
**Fecha:** 30 de mayo de 2025  

---

# 🌎 Balance Hídrico en la Provincia de Carchi con IA y Google Earth Engine

Este proyecto implementa un análisis multitemporal del balance hídrico mensual en la provincia de Carchi (Ecuador), utilizando datos climáticos de ERA5 y proyecciones CMIP6. Se despliega una aplicación interactiva construida con Streamlit que permite la visualización del balance hídrico observado, histórico y proyectado mediante modelos de aprendizaje automático (XGBoost). También se integra Google Earth Engine y Supabase para visualización geoespacial y manejo de datos remotos.

---

## 📌 Características principales

- Visualización del **balance hídrico observado** con ERA5 desde Google Earth Engine.
- Consulta del **balance histórico (1981–2023)** usando datos procesados localmente.
- Proyección del **balance hídrico futuro (2015–2049)** con datos CMIP6 y modelo XGBoost.
- Análisis gráfico de:
  - Balance hídrico mensual
  - Promedios anuales
  - Anomalías hídricas
- Aplicación interactiva en **Streamlit**
- Acceso remoto a datos desde **Supabase**
- Integración en **Google Colab** con despliegue vía **ngrok**

---

## 🧰 Tecnologías y librerías usadas

| Tecnología/Librería | Descripción |
|---------------------|-------------|
| Google Earth Engine | Extracción y análisis de datos climáticos espaciales |
| Streamlit | Visualización e interfaz web |
| XGBoost | Modelo predictivo para proyecciones |
| Pandas / NumPy | Limpieza y manipulación de datos |
| Bokeh | Visualización de series temporales |
| Scikit-learn | Escalado y procesamiento de features |
| Supabase | Base de datos en la nube tipo PostgreSQL |
| Pyngrok | Exposición pública del dashboard desde Google Colab |

---

## 📂 Estructura del repositorio

balance_hidrico/
├── app.py                       # Aplicación principal de Streamlit
├── start_streamlit             # Script para ejecutar la app con ngrok
├── git_push.ipynb              # Script opcional para sincronización con GitHub
├── dataset_cmip6.ipynb         # Notebook para procesar datos CMIP6
├── dataset_era5.ipynb          # Notebook para procesar datos ERA5
├── training_model.ipynb        # Notebook para entrenamiento de modelos ML
├── README.md                   # Descripción general del proyecto
├── wiki.md                     # Documentación técnica completa del proyecto
│
├── data/                       # Carpeta de datos de entrada
│   ├── cmip6/
│   │   ├── Carchi/
│   │   ├── Jalisco/
│   │   └── Loja/
│   └── era5/
│       └── Carchi/
│           └── df_balanceH_historico.csv  # Datos históricos de balance hídrico
│
├── model/                      # Carpeta de modelos entrenados y seguimiento
│   ├── mlflow_tracking/        # Experimentos y métricas de entrenamiento MLflow
│   └── pkl/
│       ├── modelo_xgboost_v1.pkl  # Modelo XGBoost final
│       ├── scaler_X.pkl          # Escalador para variables X
│       └── scaler_y.pkl          # Escalador para la variable Y


---

## 🚀 Cómo ejecutar la app en Google Colab

1. Montar Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Instalar dependencias:
```bash
!pip install streamlit geemap streamlit-folium seaborn pyngrok
```

3. Ejecutar el script en segundo plano:
```python
import threading, os
def run(): os.system('streamlit run app.py')
threading.Thread(target=run).start()
```

4. Exponer con ngrok:
```python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
```

---

## 👨‍🔬 Autor

**Freddy Hernán Villota González**  
Universidad Técnica Particular de Loja (UTPL)  
Maestría en Inteligencia Artificial Aplicada

---

## 📖 Licencia

Este proyecto se distribuye bajo la licencia MIT.

---

## 🔗 Referencias clave

- [ERA5 Dataset - Copernicus Climate Data Store](https://cds.climate.copernicus.eu)
- [CMIP6 - Coupled Model Intercomparison Project Phase 6](https://esgf-node.llnl.gov/projects/cmip6/)
- [Google Earth Engine](https://developers.google.com/earth-engine)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Supabase Docs](https://supabase.com/docs)
