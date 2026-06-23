# Proyecto de Clasificación con KNN - Fase 1: Preprocesamiento de Imágenes

Este repositorio contiene la implementación de la **Fase 1** para el preprocesamiento de imágenes que posteriormente serán utilizadas en un algoritmo de clasificación K-Nearest Neighbors (KNN).

El código principal de esta fase se encuentra en [fase1.py](file:///c:/Users/lopez/OneDrive/Desktop/Algoritmo%20U3/fase1.py).

---

## ¿Qué hace la Fase 1?

La Fase 1 se encarga de tomar las imágenes originales (en formatos comunes como `.jpg`, `.png`, `.jpeg`, etc.) y estandarizarlas para que el algoritmo KNN pueda procesar la información de manera eficiente y homogénea.

El proceso consta de tres pasos principales aplicados a cada imagen:

1. **Conversión a Escala de Grises**: 
   Se elimina la información de color de la imagen convirtiéndola al formato de un solo canal (`L`). Esto reduce significativamente la dimensionalidad y los requisitos de cómputo del modelo KNN, enfocando la clasificación en las formas y contrastes en lugar del color.
   
2. **Redimensionamiento Estándar**:
   Todas las imágenes se redimensionan a un tamaño fijo y cuadrado (definido por defecto en **128x128 píxeles**). Esto garantiza que todos los vectores de características de entrada tengan exactamente la misma longitud.

3. **Normalización de Píxeles**:
   Los valores de intensidad de los píxeles (que originalmente están en el rango entero `[0, 255]`) se transforman a valores de punto flotante en el rango numérico `[0.0, 1.0]` mediante la división entre `255.0`. Esto mejora la estabilidad numérica y el rendimiento de los cálculos de distancia en KNN.

---

## Resultados y Datos Procesados Entregados

Una vez ejecutado el script [fase1.py](file:///c:/Users/lopez/OneDrive/Desktop/Algoritmo%20U3/fase1.py), se generan los siguientes resultados:

### 1. Estructura de Salida
Los datos procesados se organizan automáticamente en la carpeta `datos_procesados/` manteniendo la misma estructura de carpetas por clase (etiqueta) detectada en la carpeta de origen `datos/`. Por ejemplo:
```text
datos_procesados/
├── clase_A/
│   ├── imagen1.npy
│   └── imagen2.npy
└── clase_B/
    ├── imagen3.npy
    └── imagen4.npy
```

### 2. Archivos Binarios `.npy` (Numpy Arrays)
En lugar de guardar las imágenes procesadas en formatos de imagen tradicionales (como JPEG), se exportan como archivos serializados binarios `.npy`.
* **Formato de datos**: Cada archivo contiene un arreglo de NumPy (`numpy.ndarray`) de tipo `float32`.
* **Dimensiones (Shape)**: Una matriz bidimensional de `(128, 128)`.
* **Rango de Valores**: Valores numéricos decimales continuos entre `0.0` (negro) y `1.0` (blanco).
* **Ventaja**: Estos archivos se cargan de forma extremadamente rápida y directa en Python utilizando `numpy.load()`, listos para ser aplanados (flattened) y pasados al modelo KNN.

### 3. Reporte y Verificación Visual
* **Resumen en Consola**: Se muestra un gráfico de barras textual en la consola que indica cuántas imágenes fueron procesadas con éxito por cada clase detectada.
* **Imagen de Verificación**: Se genera un archivo de imagen llamado `fase1_verificacion.png` en el directorio raíz. Este archivo muestra una comparación visual directa de algunas muestras: la imagen original al lado de su versión preprocesada en escala de grises de 128x128 píxeles.
