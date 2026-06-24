import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importar la función de preprocesamiento de la Fase 1
from fase1 import procesar_imagen

# ==========================================
# FASE 2: CARGA DE DATOS Y ENTRENAMIENTO
# ==========================================
RUTA_DATOS = "datos_procesados"

print("Cargando vectores de características preprocesados...")
X = []
y = []

# Validar que existan los datos procesados
if not os.path.exists(RUTA_DATOS):
    print(f"Error: No se encontró la carpeta '{RUTA_DATOS}'. Por favor, ejecuta primero la Fase 1.")
    exit(1)

for clase in os.listdir(RUTA_DATOS):
    ruta_clase = os.path.join(RUTA_DATOS, clase)

    if not os.path.isdir(ruta_clase):
        continue

    for archivo in os.listdir(ruta_clase):
        if archivo.endswith(".npy"):
            ruta_archivo = os.path.join(ruta_clase, archivo)
            imagen = np.load(ruta_archivo)
            
            # Vector de características
            vector = imagen.flatten()
            X.append(vector)
            y.append(clase)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("Error: No se encontraron archivos .npy en los datos procesados.")
    exit(1)

print(f"Cantidad de imágenes cargadas: {len(X)}")
print(f"Dimensión original del vector: {X.shape[1]}")

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrenamiento: {len(X_train)} muestras")
print(f"Prueba: {len(X_test)} muestras")

# Modelo KNN con la configuración optimizada
print("\nEntrenando modelo KNN...")
knn = KNeighborsClassifier(
    n_neighbors=2,
    metric='manhattan',
    weights='distance'
)
knn.fit(X_train, y_train)

# Predicción y evaluación
y_pred = knn.predict(X_test)

print("\n" + "="*40)
print("EVALUACIÓN DEL MODELO")
print("="*40)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado
ruta_modelo = "modelo_knn_final.pkl"
joblib.dump(knn, ruta_modelo)
print(f"\nModelo guardado correctamente como: '{ruta_modelo}'")

# ==========================================
# FASE 3: INFERENCIA SOBRE UNA NUEVA IMAGEN
# ==========================================
print("\n" + "="*40)
print("FASE 3: INFERENCIA CON IMAGEN NUEVA")
print("="*40)

# Imagen de prueba
ruta_nueva_imagen = "datos/tenis/tenis63.jpeg"

if os.path.exists(ruta_nueva_imagen):
    print(f"Procesando imagen de entrada: '{ruta_nueva_imagen}'...")
    # Preprocesar usando la función optimizada de la Fase 1
    imagen_procesada = procesar_imagen(ruta_nueva_imagen)
    
    # Convertir a un vector de 1 fila (1, -1)
    vector_nuevo = imagen_procesada.flatten().reshape(1, -1)
    
    # Predicción en caliente usando el modelo knn en memoria
    prediccion = knn.predict(vector_nuevo)
    print(f"Clase predicha: {prediccion[0]}")
else:
    print(f"Aviso: No se pudo realizar la predicción porque no se encontró la imagen '{ruta_nueva_imagen}'.")
