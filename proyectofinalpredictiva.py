## Predicción Discreta (Clasificación): Arbol de Decisión, Red Neuronal, KNN.
### 1. Carga y Decodificación de Datos ARFF
#Importar librerías
import pandas as pd
from scipy.io import arff
import numpy as np
### 4. Creación y Evaluación de Modelos Individuales
#### C. K-Nearest Neighbors - KNN (Entrenamiento con datos normalizados)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
print("\n--- ENTRENANDO Y EVALUANDO K-NEAREST NEIGHBORS (KNN) ---")
# Aprendizaje
# Se usan datos balanceados Y normalizados (X_train_norm)
model_knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean') # Se pueden ajustar n_neighbors
model_knn.fit(X_train_norm, Y_train_bal)
# Predicción y Reporte
Y_pred_knn = model_knn.predict(X_test_norm) # X_test normalizado para KNN
print("\nReporte de Clasificación de KNN:")
print(metrics.classification_report(Y_test, Y_pred_knn, target_names=labelencoder.classes_))
# Visual: Matriz de Confusión
print("\nMatriz de Confusión de KNN:")
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(Y_test, Y_pred_knn), display_labels=labelencoder.classes_)
disp.plot(cmap=plt.cm.Reds)
plt.title("Matriz de Confusión - KNN")
plt.show()
# Visual: Curva ROC para multiclase (One-vs-Rest)
print("\nCurva ROC de KNN (One-vs-Rest) para cada clase:")
y_score_knn = model_knn.predict_proba(X_test_norm)
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(labelencoder.classes_):
    # Binarizar Y_test para la clase actual (One-vs-Rest)
    y_test_bin = (Y_test == i).astype(int)
    # Calcular FPR, TPR y umbrales
    fpr, tpr, _ = metrics.roc_curve(y_test_bin, y_score_knn[:, i])
    # Graficar la curva ROC
    plt.plot(fpr, tpr, label=f'ROC de {class_name} (AUC = {metrics.auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Clasificador aleatorio') # Línea de base
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - KNN (Multiclase One-vs-Rest)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
# 1. Definir el nombre del archivo a subir a GitHub
filename = 'modelo_knn.pkl'
# 2. Extraer los nombres de las variables predictoras originales (X)
variables_entrenamiento = X.columns.values
# 3. Exportar el Modelo y todas sus herramientas de transformación
with open(filename, 'wb') as archivo:
    pickle.dump([model_knn, variables_entrenamiento, labelencoder, min_max_scaler], archivo)
