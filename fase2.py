import joblib

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carpeta principal
RUTA_DATOS = "datos_procesados"

# Datos y etiquetas
X = []
y = []

# Recorrer cada clase
for clase in os.listdir(RUTA_DATOS):

    ruta_clase = os.path.join(RUTA_DATOS, clase)

    if not os.path.isdir(ruta_clase):
        continue

    for archivo in os.listdir(ruta_clase):

        if archivo.endswith(".npy"):

            ruta_archivo = os.path.join(ruta_clase, archivo)

            # Cargar imagen procesada
            imagen = np.load(ruta_archivo)

            # Convertir matriz 128x128 a vector de 16384 elementos
            vector = imagen.flatten()

            X.append(vector)
            y.append(clase)

# Convertir a numpy arrays
X = np.array(X)
y = np.array(y)

print("Cantidad de imágenes:", len(X))
print("Dimensión de cada imagen:", X.shape[1])

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

# Crear modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar
knn.fit(X_train, y_train)

# Predicciones
y_pred = knn.predict(X_test)

# Evaluación
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte:")
print(classification_report(y_test, y_pred))

joblib.dump(knn, "modelo_knn.pkl")