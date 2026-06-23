import joblib
import numpy as np
from fase1 import procesar_imagen

# Cargar el modelo entrenado
modelo = joblib.load("modelo_knn.pkl")

# Preprocesar la imagen nueva usando la función de la Fase 1
# calcetin_nuevo.jpeg ya existe en el directorio de trabajo
imagen_procesada = procesar_imagen("images.jpeg")

# Convertir la matriz de la imagen a un vector de 1 fila (1, -1) para el modelo KNN
vector = imagen_procesada.flatten().reshape(1, -1)

# Realizar la predicción
prediccion = modelo.predict(vector)

print("Clase predicha:", prediccion[0])
