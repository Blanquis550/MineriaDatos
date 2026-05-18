## Predicción Discreta (Clasificación): Arbol de Decisión, Red Neuronal, KNN.
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Cargar el modelo KNN y los transformadores que ya guardaste en Colab
filename = 'modelo_knn.pkl'
modelo_knn, variables_entrenamiento, labelencoder, min_max_scaler = pickle.load(open(filename, 'rb'))

# 2. Interfaz Gráfica con Streamlit
st.title('Predicción de Estado de Situación de Generadores (Modelo KNN)')
st.write('Sube un archivo CSV con los datos de los generadores a los que deseas predecirles el estado. (Asegúrate de que no tenga la columna EstadoSituacion).')

# Crear el botón para subir el archivo masivo
archivo_subido = st.file_uploader("Sube tu archivo CSV de datos futuros aquí", type=['csv'])

if archivo_subido is not None:
    # Leer los datos que subió el usuario
    datos_nuevos = pd.read_csv(archivo_subido)
    st.write("Vista previa de los datos cargados:", datos_nuevos.head())
    
    # 3. Botón para ejecutar la predicción
    if st.button("Ejecutar Predicción del Modelo"):
        
        # --- PREPARACIÓN DE DATOS (Igual a Colab) ---
        datos_preparados = datos_nuevos.copy()
        
        # Lista de numéricas que usaste en tu código
        columnas_numericas = ['CapacidaddeVeranoMW', 'CapacidaddeInviernoMW', 'CargaMinimaMW', 
                              'MesdeOperacion', 'AniodeOperacion', 'AumentoCapacidadVeranoPlanMW', 
                              'AumentoCapacidadInviernoPlanMW', 'MesAumentoPlanificado', 
                              'AnioAumentoPlanificado', 'ReduccionCapacidadVeranoPlanMW', 
                              'ReduccionCapacidadInviernoPlanMW', 'MesReduccionPlanificada', 
                              'AnioReduccionPlanificada', 'MesRepotenciacionPlan', 
                              'AnioRepotenciacionPlan', 'MesOtrasModificaciones', 'AnioOtrasModificaciones']
        
        # Reemplazar nulos en los datos nuevos si los hay
        for col in columnas_numericas:
            if col in datos_preparados.columns:
                datos_preparados[col] = datos_preparados[col].replace('?', '-1').astype(float)
        
        # Aplicar Dummies (OBLIGATORIO drop_first=False en el despliegue)
        datos_preparados = pd.get_dummies(datos_preparados, drop_first=False, dtype=int)
        
        # Aplicar Reindex para rellenar con ceros las variables que falten 
        # y forzar que queden iguales al entrenamiento
        datos_preparados = datos_preparados.reindex(columns=variables_entrenamiento, fill_value=0)
        
        # Aplicar Transformación del normalizador (Regla de oro para KNN)
        datos_preparados[columnas_numericas] = min_max_scaler.transform(datos_preparados[columnas_numericas])
        
        # --- PREDICCIÓN ---
        # El modelo arroja números (ej: 0, 1, 2)
        predicciones_numericas = modelo_knn.predict(datos_preparados)
        
        # Convertimos los números de vuelta a texto real (ej: 'OP', 'SB', etc.)
        predicciones_texto = labelencoder.inverse_transform(predicciones_numericas)
        
        # Pegar las respuestas al cuadro de datos original y mostrarlo en la web
        datos_nuevos['Predicción_EstadoSituacion'] = predicciones_texto
        
        st.success("¡Predicciones generadas con éxito!")
        st.write(datos_nuevos)
        
        # Nota obligatoria de la profe: Informar el error del modelo al usuario final
        st.warning("Nota para la toma de decisiones: Este modelo tiene un porcentaje de error y precisión esperado en base a la fase de evaluación. Por favor verifique el reporte técnico.")
¿Cómo probarlo? Guarda este código en tu GitHub (haciendo 
