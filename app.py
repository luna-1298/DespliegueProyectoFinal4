import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import traceback

# --- PARCHE DE COMPATIBILIDAD DINÁMICO PARA SCIKIT-LEARN ---
try:
    import sklearn.ensemble._loss as modern_loss
    sys.modules['sklearn.ensemble._gb_losses'] = modern_loss
    sys.modules['sklearn.ensemble.losses'] = modern_loss
except Exception:
    pass

try:
    import sklearn.ensemble._gb_losses as legacy_loss
    sys.modules['sklearn.ensemble._loss'] = legacy_loss
except Exception:
    pass

# Configuración de Streamlit
st.set_page_config(page_title="Predicción de Rendimiento Estudiantil", layout="wide")

st.title("🎯 Aplicación Predictiva: Rendimiento Estudiantil")
st.markdown("Esta aplicación procesa datos de estudiantes y predice el resultado utilizando el modelo optimizado de Bagging.")

# --- CARGA DINÁMICA DE RECURSOS --- 
@st.cache_resource
def cargar_recursos():
    # Determinar el directorio absoluto en donde se aloja este archivo (app.py)
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Construir rutas absolutas locales seguras para GitHub
    modelo_path = os.path.join(dir_actual, 'optimized_bagging_model.joblib')
    scaler_path = os.path.join(dir_actual, 'min_max_scaler.joblib')
    
    # Fallback por si ejecutas localmente en este entorno de Colab
    if not os.path.exists(modelo_path):
        modelo_path = '/content/optimized_bagging_model.joblib'
    if not os.path.exists(scaler_path):
        scaler_path = '/content/min_max_scaler.joblib'
        
    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)
    return modelo, scaler

try:
    modelo_bagging, scaler = cargar_recursos()
    st.success("¡Modelo y Escalador cargados correctamente desde las rutas absolutas del repositorio!")
except Exception as e:
    st.error("❌ Error crítico al cargar los archivos serializados (.joblib):")
    st.code(traceback.format_exc())
    st.stop()

# --- FORMULARIO DE ENTRADA ---
st.header("📝 Ingresar Datos del Estudiante")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Edad (Age)", min_value=15.0, max_value=25.0, value=17.0, step=1.0)
    study_time_hours_week = st.number_input("Horas de estudio a la semana (StudyTime_hours_week)", min_value=0.0, max_value=40.0, value=12.0, step=0.5)
    failures = st.slider("Número de reprobaciones previas (Failures)", min_value=0, max_value=4, value=0)
    absences = st.number_input("Ausencias (Absences)", min_value=0.0, max_value=100.0, value=2.0, step=1.0)

with col2:
    internet = st.selectbox("¿Tiene acceso a Internet? (Internet)", options=["yes", "no"], index=0)
    free_time = st.slider("Tiempo libre después de clases (FreeTime: 1-Muy bajo, 5-Muy alto)", min_value=1, max_value=5, value=3)
    go_out = st.slider("Salidas con amigos (GoOut: 1-Muy poco, 5-Mucho)", min_value=1, max_value=5, value=2)
    health = st.slider("Estado de salud (Health: 1-Muy malo, 5-Muy bueno)", min_value=1.0, max_value=5.0, value=4.0, step=1.0)

with col3:
    mother_education = st.selectbox("Educación de la madre (MotherEducation: 0-Ninguna a 4-Superior)", options=[0, 1, 2, 3, 4], index=2)
    father_education = st.selectbox("Educación del padre (FatherEducation: 0-Ninguna a 4-Superior)", options=[0, 1, 2, 3, 4], index=2)
    travel_time = st.slider("Tiempo de viaje a la escuela (TravelTime: 1 a 4)", min_value=1, max_value=4, value=1)

# --- PROCESAMIENTO Y PREDICCIÓN ---
if st.button("🔮 Predecir Rendimiento"):
    try:
        # 1. Crear DataFrame con los datos de entrada
        datos_entrada = pd.DataFrame([{
            'Age': age,
            'StudyTime_hours_week': study_time_hours_week,
            'Failures': failures,
            'Absences': absences,
            'Internet': 1 if internet == "yes" else 0,
            'FreeTime': free_time,
            'GoOut': go_out,
            'Health': health,
            'MotherEducation': mother_education,
            'FatherEducation': father_education,
            'TravelTime': travel_time
        }])

        columnas_modelo = [
            'Age', 'StudyTime_hours_week', 'Failures', 'Absences',
            'Internet', 'FreeTime', 'GoOut', 'Health',
            'MotherEducation', 'FatherEducation', 'TravelTime'
        ]

        # 2. Normalización de características con el escalador entrenado
        datos_entrada_normalizados = datos_entrada.copy()
        datos_entrada_normalizados[columnas_modelo] = scaler.transform(datos_entrada[columnas_modelo])

        # 3. Predicción empleando el modelo de Bagging
        prediccion = modelo_bagging.predict(datos_entrada_normalizados[columnas_modelo])[0]

        # 4. Mostrar Resultados de la Predicción
        st.subheader("📊 Resultado de la Inferencia:")
        if prediccion == 1:
            st.success("🎉 **Aprobado (Pass / 1)**: Basado en las características ingresadas, se predice que el alumno aprobará satisfactoriamente.")
        else:
            st.error("⚠️ **Reprobado (Fail / 0)**: Se detectó un riesgo potencial de reprobación escolar.")
            
    except Exception as e:
        st.error("❌ Error durante el procesamiento de los datos o la predicción:")
        st.code(traceback.format_exc())
