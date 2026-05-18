## Predicción Discreta (Clasificación): Arbol de Decisión, Red Neuronal, KNN.
### 1. Carga y Decodificación de Datos ARFF
#Importar librerías
import pandas as pd
from scipy.io import arff
import numpy as np
# 1. Definir el nombre del archivo a subir a GitHub
filename = 'modelo_knn.pkl'
# 2. Extraer los nombres de las variables predictoras originales (X)
variables_entrenamiento = X.columns.values
# 3. Exportar el Modelo y todas sus herramientas de transformación
with open(filename, 'wb') as archivo:
    pickle.dump([model_knn, variables_entrenamiento, labelencoder, min_max_scaler], archivo)
