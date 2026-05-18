import streamlit as st
import pandas as pd
import pickle
# 1. Cargar el modelo y los transformadores
modelo_knn, vars_train, le, scaler = pickle.load(open('modelo_knn.pkl', 'rb'))
# 2. Títulos de la aplicación
st.title('Predicción Estado de Situación de Generadores')
st.write("Ingrese los datos del generador para conocer su estado operativo:")
# 3. Preguntas sencillas 
anio = st.number_input('Año de Operación (Ej: 2015)', value=2015, step=1)
carga = st.number_input('Carga Mínima MW (Ej: 10.5)', value=10.5)
fuente = st.selectbox('Fuente de Energía Principal', ['NG', 'WAT', 'DFO', 'WND', 'SUN', 'SUB', 'BIT'])
# 4. Botón de predicción
if st.button("Ejecutar Predicción"):    
    # Armamos la tablita con los datos que ingresó el usuario
    datos_usuario = pd.DataFrame({
        'AniodeOperacion': [anio],
        'CargaMinimaMW': [carga],
        'FuentedeEnergia1': [fuente]
    })
    # Magia de preparación: Dummies y relleno automático de ceros para lo que falte
    df_prep = pd.get_dummies(datos_usuario, drop_first=False, dtype=int)
    df_prep = df_prep.reindex(columns=vars_train, fill_value=0)
        # Las columnas numéricas exactas usadas en el entrenamiento
    cols_num = [
        'CapacidaddeVeranoMW', 'CapacidaddeInviernoMW', 'CargaMinimaMW', 
        'MesdeOperacion', 'AniodeOperacion', 'AumentoCapacidadVeranoPlanMW', 
        'AumentoCapacidadInviernoPlanMW', 'MesAumentoPlanificado', 
        'AnioAumentoPlanificado', 'ReduccionCapacidadVeranoPlanMW', 
        'ReduccionCapacidadInviernoPlanMW', 'MesReduccionPlanificada', 
        'AnioReduccionPlanificada', 'MesRepotenciacionPlan', 
        'AnioRepotenciacionPlan', 'MesOtrasModificaciones', 'AnioOtrasModificaciones'
    ]
        # Normalizar (Regla de oro de KNN)
    df_prep[cols_num] = scaler.transform(df_prep[cols_num])
        # Generar predicción y convertirla a texto
    prediccion_num = modelo_knn.predict(df_prep)
    resultado = le.inverse_transform(prediccion_num)
        # Mostrar el resultado final en verde
    st.success(f"✨ El modelo predice que el Estado de Situación es: {resultado}")
        # Advertencia ética 
    st.warning("⚠️ Nota: Las predicciones de este modelo tienen un margen de error según lo evaluado en la metodología CRISP-DM. Úsese como apoyo a la decisión.")
