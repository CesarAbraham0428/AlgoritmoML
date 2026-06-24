import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from PIL import Image, ImageOps

# ==========================================
# CONFIGURACIÓN Y PREPROCESAMIENTO DE IMÁGENES
# ==========================================
TAMAÑO = 128  # Tamaño objetivo de 128x128 píxeles

def procesar_imagen(ruta_imagen):
    """
    Aplica los pasos necesarios para transformar una imagen nueva:
      1. Convertir a escala de grises
      2. Mejorar contraste (autocontrast)
      3. Redimensionar manteniendo relación de aspecto con relleno (padding) blanco
      4. Normalizar píxeles a rango [0, 1]
    Retorna un numpy array listo para KNN.
    """
    # Abrir imagen
    img = Image.open(ruta_imagen)

    # PASO 1: Escala de grises
    img_gris = img.convert('L')

    # MEJORA: Autocontraste (Normaliza iluminación)
    img_gris = ImageOps.autocontrast(img_gris)

    # MEJORA: Redimensionar con relleno (padding) blanco manteniendo relación de aspecto
    img_gris.thumbnail((TAMAÑO, TAMAÑO), Image.Resampling.LANCZOS)
    
    # Lienzo blanco de 128x128
    img_padded = Image.new('L', (TAMAÑO, TAMAÑO), 255)
    
    # Centrar la imagen en el lienzo
    x = (TAMAÑO - img_gris.width) // 2
    y = (TAMAÑO - img_gris.height) // 2
    img_padded.paste(img_gris, (x, y))

    # PASO 3: Normalizar píxeles (0-255 → 0.0-1.0)
    img_array = np.array(img_padded, dtype='float32') / 255.0

    return img_array


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

# ==========================================
# VISUALIZACIÓN PCA DEL DATASET
# ==========================================

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)

plt.figure(figsize=(10,6))

for clase in np.unique(y_train):
    indices = y_train == clase

    plt.scatter(
        X_train_pca[indices, 0],
        X_train_pca[indices, 1],
        label=clase
    )

plt.title("Distribución de las prendas (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid(True)

plt.show()

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
ruta_nueva_imagen = "imagen1.jpeg"

if os.path.exists(ruta_nueva_imagen):
    print(f"Procesando imagen de entrada: '{ruta_nueva_imagen}'...")
    # Preprocesar usando la función optimizada de la Fase 1
    imagen_procesada = procesar_imagen(ruta_nueva_imagen)
    
    # Convertir a un vector de 1 fila (1, -1)
    vector_nuevo = imagen_procesada.flatten().reshape(1, -1)
    
    # Predicción en caliente usando el modelo knn en memoria
    prediccion = knn.predict(vector_nuevo)

    # Obtener vecinos más cercanos
    distancias, indices = knn.kneighbors(vector_nuevo)

    # Transformar imagen nueva al espacio PCA
    nuevo_pca = pca.transform(vector_nuevo)

    # Coordenadas de vecinos
    vecinos_pca = X_train_pca[indices[0]]

    plt.figure(figsize=(10,6))

    # Dibujar dataset completo
    for clase in np.unique(y_train):
        mask = y_train == clase

        plt.scatter(
            X_train_pca[mask,0],
            X_train_pca[mask,1],
            alpha=0.5,
            label=clase
        )

    # Dibujar vecinos
    plt.scatter(
        vecinos_pca[:,0],
        vecinos_pca[:,1],
        s=250,
        marker='s',
        edgecolors='black',
        label='Vecinos KNN'
    )

    # Dibujar imagen nueva
    plt.scatter(
        nuevo_pca[0,0],
        nuevo_pca[0,1],
        s=400,
        marker='*',
        label='Imagen nueva'
    )

    # Líneas hacia vecinos
    for vecino in vecinos_pca:
        plt.plot(
            [nuevo_pca[0,0], vecino[0]],
            [nuevo_pca[0,1], vecino[1]]
        )

    plt.title(
        f"Clasificación KNN\nPredicción: {prediccion[0]}"
    )

    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")

    plt.legend()
    plt.grid(True)

    plt.show()

    print(f"Clase predicha: {prediccion[0]}")
else:
    print(f"Aviso: No se pudo realizar la predicción porque no se encontró la imagen '{ruta_nueva_imagen}'.")
