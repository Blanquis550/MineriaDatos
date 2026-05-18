import streamlit as st
import pandas as pd
import pickle
# 1. Cargar el modelo y los transformadores
modelo_knn, vars_train, le, scaler = pickle.load(open('modelo_knn.pkl', 'rb'))
st.title('Predicción Estado de Situación de Generadores')
st.write("Ingrese los datos del generador para conocer su estado:")
# 2. Formulario de "preguntas sencillas" para el usuario
anio = st.number_input('Año de Operación (Ej: 2015)', value=2015, step=1)
carga = st.number_input('Carga Mínima MW (Ej: 10.5)', value=10.5)
fuente = st.selectbox('Fuente de Energía Principal', ['NG', 'WAT', 'DFO', 'WND', 'SUN', 'SUB', 'BIT']) # Puedes agregar más siglas de tu base
if st.button("Ejecutar Predicción"):
        # 3. Armar los datos capturados en una tabla oculta
    datos_usuario = pd.DataFrame({
        'AniodeOperacion': [anio],
        'CargaMinimaMW': [carga],
        'FuentedeEnergia1': [fuente]
    })
        # 4. Preparación automática
    # Convertir la variable categórica a Dummies
    df_prep = pd.get_dummies(datos_usuario, drop_first=False, dtype=int)
        # ¡Magia! Rellenar con ceros las variables que no le preguntamos al usuario
    df_prep = df_prep.reindex(columns=vars_train, fill_value=0)
        # Lista de numéricas puras que declaraste en tu Colab
    cols_num = ['CapacidaddeVeranoMW', 'CapacidaddeInviernoMW', 'CargaMinimaMW', 
                'MesdeOperacion', 'AniodeOperacion', 'AumentoCapacidadVeranoPlanMW', 
                'AumentoCapacidadInviernoPlanMW', 'MesAumentoPlanificado', 
                'AnioAumentoPlanificado', 'ReduccionCapacidadVeranoPlanMW', 
                'ReduccionCapacidadInviernoPlanMW', 'MesReduccionPlanificada', 
                'AnioReduccionPlanificada', 'MesRepotenciacionPlan', 
                'AnioRepotenciacionPlan', 'MesOtrasModificaciones', 'AnioOtrasModificaciones']
        # Normalizar usando el scaler que trajimos de Colab
    df_prep[cols_num] = scaler.transform(df_prep[cols_num])
        # 5. Predicción
    prediccion_num = modelo_knn.predict(df_prep)
    prediccion_texto = le.inverse_transform(prediccion_num)
                # 5. Predecir y mostrar resultado
        df['Predicción_Estado'] = le.inverse_transform(modelo_knn.predict(df_prep))
        st.success("¡Predicción exitosa!")
        st.dataframe(df)
