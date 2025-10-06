# Laboratorio 3 - Análisis espectral de la voz

**Universidad Militar Nueva Granada**

**Asignatura:** Procesamiento Digital de Señales

**Estudiantes:** Dubrasca Martinez, Mariana Leyton, Maria Fernanda Castellanos

**Fecha:** 21 de septiembre de 2025

**Título de la práctica:** Convolucion, transformada y estadisticos descriptivos de la señal EOG.

# **Objetivos**

# **Procedimiento, método o actividades**

# **Parte A**

## **Código en Python (Google colab)**
<pre> ```
# Importación de las librerias a utilizar
!pip install wfdb                                                    # Instalación de la liberia wfdb
import wfdb                                                          # Liberia para analizar señales fisiologicas
import matplotlib.pyplot as plt                                      # Liberia para permitir visualizar las graficas de las señales
import os                                                            # Liberia para interactuar con el sistema operativo
from google.colab import files                                       # Liberia en Google colab para subir archivos desde el computador
import numpy as np

archivos = ["Mujer1.wav", "Mujer2.wav", "Mujer3.wav",
            "Hombre1.wav", "Hombre2.wav", "Hombre3.wav"]

# Recorremos cada archivo
for archivo in archivos:
    # Leer el archivo wav
    fs, data = wavfile.read(archivo)

    # Si es estéreo, tomamos solo un canal
    if data.ndim > 1:
        data = data[:, 0]

    # Crear el eje de tiempo
    tiempo = [t/fs for t in range(len(data))]

    # Graficar
    plt.figure(figsize=(10,4))
    plt.plot(tiempo, data)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Muestras digitalizadas")
    plt.title(f"Forma de onda del audio - {archivo}")
    plt.grid(True)
    plt.show()
  ```
</pre>

## **Gráfica Mujer 1**
<img width="1098" height="485" alt="image" src="https://github.com/user-attachments/assets/31643bb8-661c-4d50-9fbd-64e15ddc52d8" />

## **Gráfica Mujer 2**
<img width="1098" height="490" alt="image" src="https://github.com/user-attachments/assets/64c09af1-7c23-4c94-9972-168d4e00542e" />

## **Gráfica Mujer 3**
<img width="1095" height="481" alt="image" src="https://github.com/user-attachments/assets/35e69405-a53f-4b63-9127-d3ece5e4bbe9" />

## **Gráfica Hombre 1**
<img width="1093" height="503" alt="image" src="https://github.com/user-attachments/assets/2f3b0cf3-aa7e-4f5d-968f-20ac4d125ef1" />

## **Gráfica Hombre 2**
<img width="1099" height="491" alt="image" src="https://github.com/user-attachments/assets/145fd4eb-76af-4cbc-b45c-fbe978bfcec7" />

## **Gráfica Hombre 3**
<img width="1096" height="495" alt="image" src="https://github.com/user-attachments/assets/a10f4d21-553b-4853-bc44-ff7901ce6592" />

# **Parte B**

# **Parte C**
