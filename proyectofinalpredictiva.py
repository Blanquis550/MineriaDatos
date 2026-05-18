## Predicción Discreta (Clasificación): Arbol de Decisión, Red Neuronal, KNN.
import streamlit as st
import pandas as pd
import pickle

# 1. Cargar el modelo y los transformadores ya entrenados
modelo_knn, vars_train, le, scaler = pickle.load(open('modelo_knn.pkl', 'rb'))

st.title('Predicción Estado de Situación - KNN')
st.write("Sube el archivo CSV con los generadores nuevos.")

# 2. Capturar datos masivos con un CSV
archivo = st.file_uploader("Cargar archivo CSV", type=['csv'])

if archivo is not None:
    df = pd.read_csv(archivo)
    
    if st.button("Ejecutar Predicción"):
        # 3. Preparación rápida (Dummies y Reindex para forzar que coincidan las variables)
        df_prep = pd.get_dummies(df, drop_first=False, dtype=int)
        df_prep = df_prep.reindex(columns=vars_train, fill_value=0)
        
        # 4. Normalizar solo las variables numéricas puras
        cols_num = ['CapacidaddeVeranoMW', 'CapacidaddeInviernoMW', 'CargaMinimaMW', 
                    'MesdeOperacion', 'AniodeOperacion', 'AumentoCapacidadVeranoPlanMW', 
                    'AumentoCapacidadInviernoPlanMW', 'MesAumentoPlanificado', 
                    'AnioAumentoPlanificado', 'ReduccionCapacidadVeranoPlanMW', 
                    'ReduccionCapacidadInviernoPlanMW', 'MesReduccionPlanificada', 
                    'AnioReduccionPlanificada', 'MesRepotenciacionPlan', 
                    'AnioRepotenciacionPlan', 'MesOtrasModificaciones', 'AnioOtrasModificaciones']
        
        df_prep[cols_num] = scaler.transform(df_prep[cols_num])
        
        # 5. Predecir y mostrar resultado
        df['Predicción_Estado'] = le.inverse_transform(modelo_knn.predict(df_prep))
        st.success("¡Predicción exitosa!")
        st.dataframe(df)
