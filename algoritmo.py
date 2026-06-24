import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from PIL import Image, ImageOps

TAMANO = 128
RUTA_DATOS = "datos_procesados"

def procesar_imagen(ruta_imagen):
    img = ImageOps.autocontrast(Image.open(ruta_imagen).convert('L'))
    img.thumbnail((TAMANO, TAMANO), Image.Resampling.LANCZOS)
    img_padded = Image.new('L', (TAMANO, TAMANO), 255)
    img_padded.paste(img, ((TAMANO - img.width) // 2, (TAMANO - img.height) // 2))
    return np.array(img_padded, dtype='float32') / 255.0

if not os.path.exists(RUTA_DATOS):
    print(f"Error: No se encontró la carpeta '{RUTA_DATOS}'. Carga el dataset para ejecutar el entrenamiento")
    exit(1)

X, y = [], []
for clase in os.listdir(RUTA_DATOS):
    ruta_clase = os.path.join(RUTA_DATOS, clase)
    if os.path.isdir(ruta_clase):
        for archivo in os.listdir(ruta_clase):
            if archivo.endswith(".npy"):
                X.append(np.load(os.path.join(ruta_clase, archivo)).flatten())
                y.append(clase)

X, y = np.array(X), np.array(y)
if len(X) == 0:
    print("Error: No se encontraron archivos .npy en los datos procesados.")
    exit(1)

print(f"Cantidad de imágenes cargadas: {len(X)}")
print(f"Dimensión original del vector: {X.shape[1]}")

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Entrenamiento: {len(X_entrenamiento)} muestras")
print(f"Prueba: {len(X_prueba)} muestras")

print("\nEntrenando modelo KNN...")
knn = KNeighborsClassifier(n_neighbors=2, metric='manhattan', weights='distance')
knn.fit(X_entrenamiento, y_entrenamiento)

pca = PCA(n_components=2)
X_entrenamiento_pca = pca.fit_transform(X_entrenamiento)

plt.figure(figsize=(10, 6))
for clase in np.unique(y_entrenamiento):
    indices = y_entrenamiento == clase
    plt.scatter(X_entrenamiento_pca[indices, 0], X_entrenamiento_pca[indices, 1], label=clase)
plt.title("Distribución de las prendas (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid(True)
plt.show()

y_prediccion = knn.predict(X_prueba)
print("EVALUACIÓN DEL MODELO")
print(f"Accuracy: {accuracy_score(y_prueba, y_prediccion):.4f}")

ruta_nueva_imagen = "imagen.jpg"
if os.path.exists(ruta_nueva_imagen):
    print(f"Procesando imagen de entrada: '{ruta_nueva_imagen}'...")
    imagen_procesada = procesar_imagen(ruta_nueva_imagen)
    vector_nuevo = imagen_procesada.flatten().reshape(1, -1)
    prediccion = knn.predict(vector_nuevo)

    distancias, indices = knn.kneighbors(vector_nuevo)
    nuevo_pca = pca.transform(vector_nuevo)
    vecinos_pca = X_entrenamiento_pca[indices[0]]

    plt.figure(figsize=(10, 6))
    for clase in np.unique(y_entrenamiento):
        mascara = y_entrenamiento == clase
        plt.scatter(X_entrenamiento_pca[mascara, 0], X_entrenamiento_pca[mascara, 1], alpha=0.5, label=clase)

    # Dibujar líneas punteadas hacia los vecinos
    for vecino in vecinos_pca:
        plt.plot([nuevo_pca[0, 0], vecino[0]], [nuevo_pca[0, 1], vecino[1]], color='gray', linestyle='--', alpha=0.8)

    # Graficar vecinos destacados con bordes
    plt.scatter(vecinos_pca[:, 0], vecinos_pca[:, 1], s=150, facecolors='none', edgecolors='black', linewidths=1.5, label='Vecinos KNN')
    # Graficar punto nuevo como un círculo rojo destacado
    plt.scatter(nuevo_pca[0, 0], nuevo_pca[0, 1], s=200, color='red', edgecolors='black', linewidths=2, label='Imagen nueva')

    plt.title(f"Clasificación KNN\nPredicción: {prediccion[0]}")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Clase predicha: {prediccion[0]}")
else:
    print(f"Aviso: No se pudo realizar la predicción porque no se encontró la imagen '{ruta_nueva_imagen}'.")
