import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# CARGA DE DATOS
# ==========================

RUTA_DATOS = "datos_procesados"

X = []
y = []

for clase in os.listdir(RUTA_DATOS):

    ruta_clase = os.path.join(RUTA_DATOS, clase)

    if not os.path.isdir(ruta_clase):
        continue

    for archivo in os.listdir(ruta_clase):

        if archivo.endswith(".npy"):

            ruta_archivo = os.path.join(ruta_clase, archivo)

            imagen = np.load(ruta_archivo)

            # vector de características
            vector = imagen.flatten()

            X.append(vector)
            y.append(clase)

X = np.array(X)
y = np.array(y)

print("Cantidad de imágenes:", len(X))
print("Dimensión original:", X.shape[1])

# ==========================
# DIVISIÓN DEL DATASET
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

# ==========================
# MODELO FINAL KNN
# ==========================

# Usamos la combinación óptima de hiperparámetros:
# K=2, distancia Manhattan y pesos basados en la distancia (evita empates y da más valor a los más cercanos)
knn = KNeighborsClassifier(
    n_neighbors=2,
    metric='manhattan',
    weights='distance'
)

knn.fit(X_train, y_train)

# ==========================
# PREDICCIÓN Y EVALUACIÓN
# ==========================

y_pred = knn.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte:")
print(classification_report(y_test, y_pred))

# ==========================
# GUARDAR MODELO
# ==========================

joblib.dump(knn, "modelo_knn_final.pkl")