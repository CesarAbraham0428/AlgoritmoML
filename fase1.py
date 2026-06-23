# ============================================================
#  FASE 1 - Preprocesamiento de imágenes para KNN
#  Pasos: Escala de grises → Redimensionar → Normalizar
# ============================================================

from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────
TAMAÑO          = 128          # 128x128 píxeles (suficiente para KNN con pocas imágenes)
CARPETA_DATOS   = "datos"      # Carpeta con tus imágenes originales
CARPETA_SALIDA  = "datos_procesados"  # Carpeta donde se guardarán los resultados

# ─────────────────────────────────────────
#  FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ─────────────────────────────────────────
def procesar_imagen(ruta_imagen):
    """
    Aplica los pasos de la Fase 1 a una imagen:
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


# ─────────────────────────────────────────
#  PROCESAMIENTO DE TODAS LAS IMÁGENES
# ─────────────────────────────────────────
def fase1_preprocesar():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Detectar clases automáticamente desde las carpetas
    clases = [d for d in os.listdir(CARPETA_DATOS)
              if os.path.isdir(os.path.join(CARPETA_DATOS, d))]

    if not clases:
        print("No se encontraron carpetas de clases en:", CARPETA_DATOS)
        return

    print(f"Clases detectadas: {clases}")
    print(f"Tamaño objetivo: {TAMAÑO}x{TAMAÑO}\n")
    print("=" * 50)

    resumen = {}

    for clase in clases:
        ruta_clase        = os.path.join(CARPETA_DATOS, clase)
        ruta_salida_clase = os.path.join(CARPETA_SALIDA, clase)
        os.makedirs(ruta_salida_clase, exist_ok=True)

        # Listar imágenes válidas
        extensiones = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        imagenes = [f for f in os.listdir(ruta_clase)
                    if f.lower().endswith(extensiones)]

        print(f"\n Clase: {clase.upper()}  ({len(imagenes)} imágenes)")
        exitosas = 0

        for idx, nombre in enumerate(imagenes, 1):
            ruta_origen = os.path.join(ruta_clase, nombre)

            try:
                # Procesar imagen
                img_procesada = procesar_imagen(ruta_origen)

                # Guardar como archivo .npy (numpy array) original
                nombre_sin_ext = os.path.splitext(nombre)[0]
                nombre_npy = f"{nombre_sin_ext}.npy"
                ruta_destino = os.path.join(ruta_salida_clase, nombre_npy)
                np.save(ruta_destino, img_procesada)

                # Guardar variación horizontal reflejada (flip)
                img_flipped = np.fliplr(img_procesada)
                nombre_npy_flip = f"{nombre_sin_ext}_flip.npy"
                ruta_destino_flip = os.path.join(ruta_salida_clase, nombre_npy_flip)
                np.save(ruta_destino_flip, img_flipped)

                print(f"{idx:02d}. {nombre:30s} -> shape: {img_procesada.shape}  min: {img_procesada.min():.2f}  max: {img_procesada.max():.2f} (+flip)")
                exitosas += 2

            except Exception as e:
                print(f"{idx:02d}. {nombre} -> ERROR: {e}")

        resumen[clase] = (exitosas, len(imagenes) * 2)

    # ─────────────────────────────────────────
    #  RESUMEN FINAL
    # ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("RESUMEN FASE 1")
    print("=" * 50)
    total_imgs = 0
    for clase, (ok, total) in resumen.items():
        barra = "" * ok + "░" * (total - ok)
        print(f"  {clase:12s}: {barra}  {ok}/{total} procesadas")
        total_imgs += ok
    print(f"\n  Total procesadas: {total_imgs} imágenes")
    print(f"  Guardadas en:     {CARPETA_SALIDA}/")
    print("=" * 50)


# ─────────────────────────────────────────
#  VERIFICACIÓN VISUAL (opcional pero útil)
# ─────────────────────────────────────────
def verificar_resultado(n_muestras=2):
    """
    Muestra una comparación visual: original vs procesada
    para verificar que la Fase 1 funcionó correctamente.
    """
    clases = [d for d in os.listdir(CARPETA_SALIDA)
              if os.path.isdir(os.path.join(CARPETA_SALIDA, d))]

    fig, ejes = plt.subplots(len(clases), n_muestras * 2,
                              figsize=(4 * n_muestras * 2, 4 * len(clases)))

    # Asegura que ejes siempre sea 2D aunque haya 1 sola clase
    if len(clases) == 1:
        ejes = [ejes]

    for fila, clase in enumerate(clases):
        archivos_npy = os.listdir(os.path.join(CARPETA_SALIDA, clase))[:n_muestras]

        for col, archivo in enumerate(archivos_npy):
            # Imagen procesada (desde .npy)
            img_proc = np.load(os.path.join(CARPETA_SALIDA, clase, archivo))

            # Imagen original (busca el jpg/png equivalente)
            nombre_orig = os.path.splitext(archivo)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                ruta_orig = os.path.join(CARPETA_DATOS, clase, nombre_orig + ext)
                if os.path.exists(ruta_orig):
                    img_orig = Image.open(ruta_orig)
                    break

            # Columna ORIGINAL
            ax_orig = ejes[fila][col * 2]
            ax_orig.imshow(img_orig)
            ax_orig.set_title(f"{clase}\nOriginal", fontsize=9)
            ax_orig.axis('off')

            # Columna PROCESADA
            ax_proc = ejes[fila][col * 2 + 1]
            ax_proc.imshow(img_proc, cmap='gray', vmin=0, vmax=1)
            ax_proc.set_title(f"Procesada\n{TAMAÑO}x{TAMAÑO} | [0-1]", fontsize=9)
            ax_proc.axis('off')

    plt.suptitle("Fase 1 - Verificación: Original vs Procesada", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    ruta_carpeta_verificacion = "verificacion"
    os.makedirs(ruta_carpeta_verificacion, exist_ok=True)
    ruta_guardado = os.path.join(ruta_carpeta_verificacion, "fase1_verificacion.png")
    
    plt.savefig(ruta_guardado, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"\nVerificación guardada como: {ruta_guardado}")


# ─────────────────────────────────────────
#  EJECUCIÓN
# ─────────────────────────────────────────
if __name__ == "__main__":
    fase1_preprocesar()       # Procesa todas las imágenes
    verificar_resultado(n_muestras=2)  # Muestra comparación visual